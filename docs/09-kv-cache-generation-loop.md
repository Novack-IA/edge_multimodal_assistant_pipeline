# 09 — Laço de Geração, Contrato de Turno/Sessão e Injeção de Vídeo em Tempo Real

Este é o **coração do sistema** e a área mais delicada. Erros aqui vazam para latência,
consistência de contexto e estabilidade de longo prazo (NFR-R3). Leia inteiro antes de tocar
no laço.

> **Revisão 2026-09-03:** o KV cache de baixo nível (chaves/valores da atenção, página por
> página) **não é mais gerido por código nosso** — é gerido internamente pelo **vLLM**
> (PagedAttention + Automatic Prefix Caching). O que continua sendo **nosso**, e continua
> exigindo estado explícito documentado (CLAUDE.md invariante 5), é o **contrato de
> turno/sessão**: a lista de mensagens que enviamos a cada requisição, quando um frame de vídeo
> entra nela, e quando/o que é consolidado no histórico persistido. Este documento descreve
> esse contrato.

## 1. Conceitos

- **KV cache (nível vLLM):** a cada token, a atenção reusa as chaves/valores dos tokens
  anteriores em vez de recomputá-los. O vLLM gerencia isso com **PagedAttention** (páginas de
  memória, como memória virtual), **continuous batching** e **Automatic Prefix Caching (APC)**
  — prefixos de prompt idênticos entre requisições reusam KV automaticamente, sem recompute.
  Referências: [lfm2_tool_parser — vLLM Docs](https://docs.vllm.ai/en/latest/api/vllm/tool_parsers/lfm2_tool_parser/),
  [Tool Calling — vLLM Docs](https://docs.vllm.ai/en/latest/features/tool_calling/).
- **Contrato de turno/sessão (nível nosso):** a lista `messages` (OpenAI-style) que construímos
  a cada turno e enviamos ao vLLM. É **esse** objeto que tem ciclo de vida explícito — quando um
  bloco de imagens entra, quando é descartado, quando uma resposta é consolidada no histórico.

## 2. O desafio específico deste projeto

Mesmo delegando o KV cache de token ao vLLM, o contrato de turno/sessão não é trivial, pelos
mesmos três motivos de sempre — só que resolvidos em outra camada:

1. **Entrada de vídeo em tempo real:** frames mudam continuamente. Decidimos *quais* entram na
   mensagem do turno, *quando*, e garantimos que **nunca persistem** no histórico reenviado
   (senão o prefixo "estável" deixaria de ser estável, quebrando o Automatic Prefix Caching e
   estourando o teto de 32K tokens rapidinho).
2. **Barge-in:** interrupções podem acontecer no meio da geração. Como nunca escrevemos o turno
   no histórico até ele terminar com sucesso, "rebobinar" vira simplesmente **não commitar** —
   mais simples e mais seguro que truncamento manual de páginas, mas precisa ser um contrato
   explícito, não um acidente de implementação.
3. **Tool calling:** resultados de ferramenta são anexados **dentro do turno atual** (mensagens
   `assistant` com `tool_calls` + `tool` com o resultado), e só entram no histórico persistido
   se o turno inteiro terminar com sucesso.

## 3. Estrutura lógica da lista de mensagens por turno

```
[ SYSTEM ]                       ← estável, reenviado todo turno (APC do vLLM reaproveita)
[ HISTÓRICO CONSOLIDADO (texto) ]← cresce por turno; sujeito a janela deslizante; SEM imagens
[ USER: frames da janela + texto]← turno atual — frames aqui, nunca no histórico consolidado
[ ASSISTANT + TOOL* ]            ← resposta e tool calls do turno atual (staging, não commitado)
```

Decisão de projeto sobre **posição dos frames**: ficam **só na mensagem do turno atual**, nunca
no histórico persistido. Consequência:
- O prefixo `[SYSTEM][HISTÓRICO]` é **texto puro e estável** — o Automatic Prefix Caching do
  vLLM tem prefixo idêntico byte-a-byte entre turnos consecutivos (até crescer com o próximo
  commit), maximizando o reuso.
- Trocar os frames a cada turno só afeta o **sufixo** da requisição — nunca invalida o prefixo
  cacheado.
- O teto de contexto (32K, ver §6) nunca é inflado por imagens acumuladas de turnos antigos.

## 4. Política da "janela de vídeo"

Parâmetros no `config.yaml`:
- `video_fps` (default 2) — taxa de amostragem de frames.
- `video_window_frames` (default 4–8) — quantos frames são anexados por turno.
- `video_token_budget` — teto de tokens gastos com imagem por turno. **Diferente do plano
  original**, não há mais um `downsample_factor` que controlamos — quem decide tokens/imagem é
  o processor interno do LFM2.5-VL-3B (tiling 512×512 + thumbnail). Calibrar este orçamento
  **empiricamente**, medindo tokens reais por imagem/resolução em M1/M2 (ver [`05`](05-vision-encoder-coupling.md) §5).
- `frame_selection` = `uniform` | `keyframe` (por diferença de frame).

Regras:
1. Um **produtor de vídeo** roda contínuo (independe do estado da conversa — [`08`](08-orchestration-barge-in.md)),
   mantendo um **ring buffer** de frames brutos (sem encode — não há mais estágio de encoder
   nosso).
2. Na **entrada em THINKING**, o orquestrador tira um **snapshot** do ring buffer e o anexa como
   conteúdo de imagem na mensagem `user` daquele turno.
3. Durante a geração daquele turno, a janela de vídeo é **congelada** (não muda no meio da
   resposta). Frames novos entram só no **próximo** turno.

> **Não** re-anexar frames a cada chunk gerado. Os frames entram uma vez no prompt do turno; a
> geração autoregressiva do vLLM atende ao próprio KV cache interno. Atualização de frames =
> evento *entre* turnos, não *dentro* do turno.

### Variante avançada (opcional, M6): frames intra-turno
Se for necessário reagir a mudança visual no meio de uma resposta longa: pausar, abrir uma
**nova** requisição ao vLLM com o histórico do turno-até-agora + novos frames + instrução de
continuação. O Automatic Prefix Caching absorve parte do custo (o prefixo `[SYSTEM][HISTÓRICO]`
+ texto já gerado é reaproveitado), mas ainda é uma segunda requisição, não uma mutação in-place
do KV. Tratar como otimização futura, não MVP.

## 5. Ciclo de vida do contrato de turno/sessão

| Evento | Ação |
|--------|-------------------|
| **Boot** | Subir o vLLM, aguardar warm-up completo (NFR-R6, [`13`](13-vllm-deployment.md) §4), montar `[SYSTEM]` fixo. |
| **Novo turno (THINKING)** | Montar `messages = [system] + histórico_consolidado + [user: frames+texto]`. |
| **Chunk recebido (streaming)** | Repassar para TTS; checar `cancel_token` a cada chunk. |
| **Tool call** | Anexar `assistant(tool_calls=...)` + `tool(result)` **à lista local do turno** (não ao histórico persistido); nova requisição continua o raciocínio. |
| **Fim do turno (sucesso)** | **Commit**: consolidar `user(texto)` + `assistant(texto final)` no histórico persistido — **sem** as imagens e **sem** as mensagens `tool` intermediárias (opcionalmente resumidas em uma frase, se relevante para o histórico). |
| **Barge-in** | Fechar a conexão HTTP (aborta a geração no vLLM); **não commitar nada** do turno — o histórico persistido permanece exatamente como estava antes do turno começar. |
| **Janela de histórico cheia** | Truncar/resumir os turnos mais antigos (§6). |

### Barge-in: por que "não commitar" basta
Como o histórico persistido só é escrito **uma vez, no fim bem-sucedido do turno** (nunca
incrementalmente durante a geração), um turno abortado simplesmente nunca chega a esse commit.
Não há necessidade de truncar páginas de KV manualmente — isso é responsabilidade do vLLM, e ele
não recebe instrução para persistir nada além do que está dentro da requisição HTTP que
acabamos de abortar. Ainda assim, isso é um **contrato explícito** (CLAUDE.md invariante 5): o
código do commit deve estar num único ponto, só alcançado no caminho de sucesso — nunca "quase
sempre chamado exceto quando X".

## 6. Gestão de contexto longo (NFR-R3: 2 h sem vazar; NFR-R7: teto de 32K)

- **Teto real e menor:** o LFM2.5-VL-3B tem **32.768 tokens** de contexto — bem menor que os
  128K assumidos no plano original (que era para um backbone text-only puro). `history_max_turns`
  deve ser calibrado com folga para o bloco de frames + resposta do turno atual, não só para o
  texto do histórico.
- **Janela deslizante de histórico:** manter os últimos K turnos íntegros (texto); turnos mais
  antigos são **resumidos** (um passo barato do próprio LFM2.5-VL-3B, sem imagens) ou
  descartados. Configurável (`history_max_turns`, `summarize_old_turns`).
- **Frames nunca acumulam:** por construção (§3), frames só existem na mensagem do turno atual
  — nunca são escritos no histórico persistido. Isso elimina por completo a classe de vazamento
  "contexto de vídeo crescendo sem limite" que o plano original mitigava manualmente.
- **Orçamento calculado antes de enviar:** somar tokens estimados de `[system] + histórico +
  frames + texto]` **antes** de disparar a requisição; truncar/resumir se ultrapassar o teto —
  nunca descobrir o estouro pela resposta de erro do vLLM.
- **Monitoramento:** logar, por turno, tokens do prefixo estável, tokens de imagem, tokens de
  resposta e (via métricas do vLLM, ver [`13`](13-vllm-deployment.md) §7) ocupação do pool de
  KV cache. Teste de soak de 2 h no CI de device (M4).

## 7. Interação com continuous batching

O **scheduler do próprio vLLM** faz continuous batching — não é algo que implementamos. Mesmo
com um único usuário interativo, isso ajuda quando há sobreposição (ex.: um passo de resumo de
histórico rodando enquanto o turno principal gera). Nosso trabalho é só garantir que o turno
interativo tenha prioridade sobre tarefas de fundo (resumo, pré-aquecimento) — via fila/ordem de
disparo das requisições, já que o vLLM não expõe prioridade explícita de forma trivial nesta
versão; medir se isso é suficiente ou se é preciso enfileirar tarefas de fundo com atraso
deliberado.

## 8. Pseudo-código do laço de geração

```python
# services/llm/generation_loop.py (esboço normativo)
async def run_turn(session, user_text, frames, tools):
    turn_messages = [
        *session.system_and_history,        # prefixo estável — texto puro
        {
            "role": "user",
            "content": [*[frame_to_image_part(f) for f in frames],
                        {"type": "text", "text": user_text}],
        },
    ]
    response_text = ""
    async for event in llm_client.stream_chat(turn_messages, tools, session.sampling,
                                               cancel_token=turn.cancel_token):
        if turn.cancel_token.is_set():                    # BARGE-IN
            return                                         # nada commitado; sessão intacta
        if event.is_tool_call:
            result = await run_tool(event.tool_call, turn.cancel_token)   # cancelável
            if turn.cancel_token.is_set():
                return
            turn_messages.append(event.as_assistant_tool_call_message())
            turn_messages.append(result.as_tool_message())
            continue                                       # nova chamada com o resultado anexado
        response_text += event.token
        yield event.token                                  # stream p/ TTS (doc 07)

    session.commit(user_text, response_text)                # SÓ texto entra no histórico
```

## 9. Regras invioláveis (repetidas do [`CLAUDE.md`](../CLAUDE.md))
1. Frames entram **uma vez por turno**, na mensagem do turno; **nunca** são escritos no
   histórico persistido.
2. Barge-in fecha a conexão HTTP e **não commita nada** — a sessão permanece no último estado
   consolidado.
3. Só **texto** é consolidado no histórico entre turnos (resposta final, sem imagens nem
   mensagens `tool` intermediárias).
4. O prefixo `[SYSTEM][HISTÓRICO]` é reenviado idêntico a cada turno; o Automatic Prefix
   Caching do vLLM evita recompute — mas isso é uma propriedade do runtime, não desculpa para
   não versionar/documentar o contrato de mensagens.
5. Todo caminho (token, tool call, barge-in) **checa o cancel_token**.

## 10. Riscos

| Risco | Mitigação |
|-------|-----------|
| Reenviar o prefixo inteiro a cada turno parece caro | Automatic Prefix Caching do vLLM evita recompute do prefixo idêntico; medir hit rate real |
| Prefixo "quase idêntico" (ex.: timestamp dentro do system prompt) quebra o cache hit | manter o `[SYSTEM]` **literalmente estável** entre turnos — nada de dados variáveis nele |
| Vazamento de contexto em sessão longa | frames nunca persistem (§3); janela de histórico; soak test 2 h |
| Barge-in corrompe a sessão | commit só no caminho de sucesso (§5); nunca escrita incremental |
| Tool result infla o turno | limitar tamanho do resultado inserido; resumir se grande |
| Frames "velhos" na resposta | snapshot no início de THINKING; janela congelada durante a geração |
| Teto de 32K estourado sem aviso | calcular orçamento antes de enviar (§6); truncar/resumir preventivamente |
