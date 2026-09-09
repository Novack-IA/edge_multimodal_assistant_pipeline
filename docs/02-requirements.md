# 02 — Requisitos

Notação: **FR** = requisito funcional, **NFR** = não-funcional. Prioridade: **M** (must),
**S** (should), **C** (could).

> **Revisão 2026-09-03:** o cérebro passou de "LFM2.5-2.6B text-only + conector treinado por
> nós" para "**LFM2.5-VL-3B**, VLM pronto da Liquid AI, servido via **vLLM**". Requisitos abaixo
> foram atualizados e expandidos para refletir isso — em especial licenciamento (§4), orçamento
> de contexto (§1 e §2), quantização (§1 e §3), cold-start do vLLM (§3), e capacidades nativas
> do modelo que antes não existiam (§6).

## 1. Requisitos funcionais

### Áudio / STT
- **FR-1 (M)** O sistema detecta início e fim de fala do usuário via VAD, com endpointing
  configurável (silêncio sustentado default 500–800 ms).
- **FR-2 (M)** O STT transcreve pt-BR e emite **transcrições parciais** durante a fala e
  uma **final** no endpoint.
- **FR-3 (S)** O STT aceita configuração para também reconhecer inglês (multilíngue).
- **FR-4 (M)** O STT roda em GPU local (CTranslate2/CUDA), sem chamadas de rede.

### Visão
- **FR-5 (M)** O sistema captura vídeo da câmera continuamente e mantém uma **janela dos
  últimos N frames** amostrados (fps configurável, default 2), como imagens brutas — não há
  encoder próprio (ver [`05`](05-vision-encoder-coupling.md)).
- **FR-6 (M)** No momento de montar o turno, os frames selecionados são anexados como
  **conteúdo de imagem** na mensagem do usuário enviada ao vLLM; a codificação (SigLIP2 NaFlex
  + projetor) acontece dentro do modelo servido, não em código nosso.
- **FR-7 (M)** No momento de responder, o cérebro considera o estado visual **corrente**
  (não um frame velho de segundos atrás) — frames são anexados fresh a cada turno e nunca
  reaproveitados de um turno anterior (ver [`09`](09-kv-cache-generation-loop.md)).
- **FR-8 (C)** Suporte a compreensão temporal mais longa (ação ao longo do tempo) — fase M6. O
  modelo já suporta nativamente clipes curtos (poucos frames) como input multi-imagem; isso
  cobre parte do caso de uso sem trabalho adicional.
- **FR-8b (S, M5)** Quando o usuário pedir ("onde está o X?", "aponte para..."), o sistema pode
  retornar **coordenadas aproximadas** de objetos na cena (grounding), capacidade nativa do
  LFM2.5-VL-3B (RefCOCO P@1 87.9 — ver [`05`](05-vision-encoder-coupling.md) §4). Requer decidir
  um formato de saída falado (ex.: "à sua esquerda", não coordenadas em pixel) — golden rule:
  não inventar UX de coordenadas sem validar com o usuário do produto.
- **FR-8c (C, M6)** O sistema pode ler texto/telas/documentos apresentados à câmera (OCR/leitura
  de tela nativa do modelo, ScreenSpot-v2 80.7 médio). Não é objetivo central do assistente
  falado, mas a capacidade está disponível sem custo adicional de treino.

### Cérebro / diálogo / tools
- **FR-9 (M)** Um único modelo (**LFM2.5-VL-3B**, servido via vLLM) conduz o diálogo, com
  histórico de conversa.
- **FR-10 (M)** O modelo gera resposta em **streaming de tokens** via SSE (`/v1/chat/completions`
  com `stream=true`).
- **FR-11 (M)** O modelo realiza **tool calling** nativo (formato pythonic
  `<|tool_call_start|>[...]<|tool_call_end|>`), extraído pelo parser `lfm2` do vLLM em
  `tool_calls` estruturados; o orquestrador executa e devolve o resultado, e o modelo continua o
  raciocínio (padrão agentic).
- **FR-11b (S)** Tool calling deve funcionar tanto quando disparado por **texto** quanto quando
  a decisão depende do **conteúdo da imagem** (ex.: "liga a luz que está apagada" após ver a
  cena) — capacidade nativa a validar em M2.
- **FR-12 (M)** Ferramentas executam de forma assíncrona com timeout; falha de ferramenta é
  tratada graciosamente (o modelo é informado do erro via mensagem `tool` com `is_error=true`).
- **FR-12b (S)** A precisão/quantização do modelo servido (BF16/FP8/NVFP4/GGUF) é trocável via
  `config.yaml` sem mudança de código (ver [`13`](13-vllm-deployment.md) §5), para permitir
  degradação graciosa (FR-21) e ajuste fino de latência vs. qualidade.

### TTS / voz
- **FR-13 (M)** A saída é falada em **pt-BR**.
- **FR-14 (M)** A voz é **clonada** de uma amostra de referência (XTTS-v2), configurável.
- **FR-15 (M)** O TTS opera em **streaming** (sintetiza e toca antes de a resposta terminar).

### Interação / barge-in
- **FR-16 (M)** Durante a fala do assistente, se o usuário começar a falar, o sistema
  **interrompe imediatamente** o TTS e a geração do modelo (barge-in, abortando o stream HTTP
  do vLLM) e passa a ouvir.
- **FR-17 (M)** O sistema faz **turn-taking** correto: não fala por cima do usuário nem
  responde a ruído de fundo (VAD + confirmação de fala).
- **FR-18 (S)** Suporte a *backchannels* (o usuário dizer "aham", "sei" sem interromper) —
  distinguir de barge-in real. Fase M5.

### Sistema
- **FR-19 (M)** Um único `config.yaml` controla modelos, pesos, thresholds, quantização do vLLM
  e orçamentos.
- **FR-20 (M)** Logging estruturado com timestamps por fronteira de estágio.
- **FR-21 (S)** Modo de degradação (baixa fps/resolução/beam/quantização) quando a GPU satura.

## 2. Requisitos não-funcionais — orçamento de latência

Alvos medidos como **percentis (p50 / p95)** sob carga nominal. "Round-trip" = do fim da
fala do usuário até o **primeiro áudio** do assistente.

| Métrica | Alvo p50 | Alvo p95 | Onde é medido |
|---------|----------|----------|---------------|
| **NFR-L1** Barge-in stop (fala detectada → áudio para) | **≤ 120 ms** | ≤ 200 ms | [`08`](08-orchestration-barge-in.md) |
| **NFR-L2** Endpointing (fim real da fala → decisão de endpoint) | ≤ 600 ms | ≤ 900 ms | VAD |
| **NFR-L3** STT final (endpoint → transcrição final) | ≤ 300 ms | ≤ 600 ms | STT |
| **NFR-L4** TTFT do vLLM, **prompt multimodal** (mensagens prontas → 1º token) | ≤ 250 ms | ≤ 500 ms | [`06`](06-llm-brain-generation.md) |
| **NFR-L5** TTFB do TTS (1ª frase → 1º chunk de áudio) | ≤ 200 ms | ≤ 350 ms | TTS |
| **NFR-L6** **Round-trip conversacional** (fim da fala → 1º áudio) | **≤ 800 ms** | ≤ 1200 ms | e2e |
| **NFR-L7** Pré-processamento + prefill de imagem, dentro do TTFT (por frame anexado) | ≤ 40 ms | ≤ 80 ms | [`06`](06-llm-brain-generation.md) |
| **NFR-L8** Taxa de geração sustentada do vLLM | ≥ 30 tok/s | ≥ 20 tok/s | [`06`](06-llm-brain-generation.md) |

> **NFR-L4 revisado:** diferente do plano original, o TTFT agora inclui o custo de **prefill de
> tokens de imagem** (frames da janela de vídeo viram tokens no prompt via SigLIP2 NaFlex +
> tiling 512×512). Isso tende a ser mais caro que TTFT só-texto. O alvo é mantido como piso a
> validar em M1/M2 — se não bater, a alavanca é reduzir `window_frames`/resolução antes de
> relaxar o alvo (FR-21).
>
> **NFR-L7 revisado:** não existe mais um estágio de "encode de frame" isolado (era uma etapa
> TensorRT nossa); a codificação acontece dentro do forward do vLLM. Este NFR agora mede a
> fração do TTFT atribuível ao processamento de imagem (via profiling/logging estruturado), não
> um serviço separado.
>
> **NFR-L8 revisado:** a Liquid publicou ~20 tok/s em smartphone (Galaxy S26 Ultra), ~116 tok/s
> em laptop AMD Ryzen AI Max+ 395, ~228 tok/s em Apple M5 Max, e throughput de ~11K tok/s em H100
> sob alta concorrência. Nenhum desses é Jetson Thor. O alvo de ≥30 tok/s p50 é mantido como piso
> conservador — Jetson Thor deve superá-lo com folga por ser mais potente que os dispositivos
> mobile/laptop citados; **validar no device em M1** e revisar o alvo para cima se a folga for
> grande (não para baixo).
>
> O round-trip percebido é dominado por **NFR-L2 + NFR-L3 + NFR-L4 + NFR-L5** encadeados,
> mas o streaming permite sobreposição (o TTS da 1ª frase começa enquanto o vLLM ainda gera).

## 3. NFR — recursos e robustez

- **NFR-R1 (M)** Todos os modelos residentes cabem na memória da plataforma alvo com folga de
  ≥ 20% (ver alocação em [`03`](03-hardware-platform.md); LFM2.5-VL-3B varia de ~1.6 GB
  (GGUF Q4_0) a ~5.4 GB (BF16) conforme quantização escolhida — folgado nos 128 GB da Thor
  mesmo sem quantizar).
- **NFR-R2 (M)** Nenhum componente do laço quente faz I/O de rede — **incluindo** o servidor
  vLLM, que roda localmente no device (não é um serviço de nuvem; a API é HTTP local).
- **NFR-R3 (M)** O sistema roda de forma estável por ≥ 2 h contínuas sem vazamento de memória
  (ver [`09`](09-kv-cache-generation-loop.md); o pool de KV cache do vLLM tem tamanho fixo
  reservado, então "vazamento" aqui é sobre nosso histórico/sessão crescer sem limite, não
  sobre páginas do vLLM).
- **NFR-R4 (S)** Recuperação automática se um worker de modelo cair (supervisão + restart,
  incluindo o processo/container do vLLM).
- **NFR-R5 (M)** Determinismo reproduzível: versões de JetPack, CUDA, **versão do vLLM** e pesos
  fixadas e registradas (ver [`13`](13-vllm-deployment.md)).
- **NFR-R6 (M, NOVO)** O orquestrador só entra em estado "pronto para uso" após o servidor vLLM
  completar o **warm-up** (compilação JIT de kernels Triton/FlashInfer para a arquitetura
  Blackwell/SM_110a da Thor, esperada em **5–15 min na primeira subida**; subidas seguintes
  reusam cache local) e responder com sucesso a um turno dummy. Ver [`13`](13-vllm-deployment.md)
  §4. Falhar a inicialização silenciosamente aceitando turnos antes disso não é aceitável.
- **NFR-R7 (M, NOVO)** A gestão de histórico de conversa deve respeitar o **teto real de
  contexto do LFM2.5-VL-3B: 32.768 tokens** (bem menor que os 128K do plano original, que
  assumia o 2.6B text-only puro). O orçamento por turno (system + histórico + frames + texto +
  resposta) deve ser calculado e truncado/resumido **antes** de estourar esse teto, não depois
  de uma falha de requisição (ver [`09`](09-kv-cache-generation-loop.md) §6).

## 4. NFR — privacidade e licenciamento

- **NFR-P1 (M)** Áudio e vídeo brutos não saem do dispositivo.
- **NFR-P2 (M)** **Revisar licenças** de cada peso antes de uso comercial. Atenção especial:
  **XTTS-v2 usa a Coqui Public Model License (CPML)** — verificar termos para uso comercial
  (ver [`07`](07-tts-voice-cloning.md) e a nota de licenciamento). Ter um **plano B de TTS**
  com licença permissiva caso o uso seja comercial.
- **NFR-P3 (M)** Registrar a licença de LFM2.5-VL-3B, faster-whisper e (se aplicável no futuro,
  M6) datasets de fine-tuning de domínio.
- **NFR-P4 (M, NOVO)** **LFM2.5-VL-3B usa a LFM Open License v1.0 ("lfm1.0")** — baseada em
  Apache 2.0, com um **limiar de receita**: uso comercial é **gratuito** para organizações com
  receita anual **< US$10 milhões**; acima disso, o uso comercial requer licença paga (contatar
  `sales@liquid.ai`). Sem copyleft — pesos fine-tuned podem ficar proprietários. Organizações
  sem fins lucrativos (501(c)(3) ou equivalente) podem usar para fins não-comerciais/pesquisa
  sem limiar de receita. **Ação:** registrar a receita anual da organização usuária e
  revisitar esta licença antes de qualquer deploy comercial (mesma disciplina do NFR-P2 para o
  XTTS-v2).

## 5. Capacidades nativas herdadas do LFM2.5-VL-3B (não exigem treino)

Diferente do plano original — onde qualquer capacidade multimodal exigiria treino do
conector —, o modelo pronto já traz um conjunto de capacidades que **não custam trabalho de
treino adicional** para existir, só decisão de produto sobre expô-las ou não:

| Capacidade | Benchmark publicado | Status no produto |
|---|---|---|
| Descrição/pergunta sobre a cena (VQA) | — (uso central do produto) | **Adotado** — FR-6/FR-7 |
| Tool calling a partir de imagem | ToolSandbox 59.5, BFCL v4 32.5 | **Adotado** — FR-11b |
| Grounding (coordenadas de objetos) | RefCOCO-avg P@1 87.9 | Should, M5 — FR-8b |
| Leitura de tela/documento (OCR) | ScreenSpot-v2 80.7 médio | Could, M6 — FR-8c |
| Multi-imagem / clipe curto de vídeo | BLINK 61.5, MuirBench 58.3; 5 frames, TTFT 34 ms (H100) | **Adotado parcialmente** — janela de vídeo (FR-5) já usa isso |

Ver [`05`](05-vision-encoder-coupling.md) §4 para as fontes e números completos.

## 6. Critérios de aceitação (Definition of Done do MVP)

O MVP (fim de M4) está pronto quando, na plataforma alvo (Jetson AGX Thor):

1. O usuário conversa por voz em pt-BR e recebe respostas faladas com voz clonada.
2. O assistente responde corretamente a perguntas sobre o que a câmera vê *no momento*.
3. O usuário consegue interromper a fala do assistente e ser ouvido (barge-in dentro de
   NFR-L1), com o histórico de conversa preservado corretamente (nenhum turno abortado
   commitado — ver [`09`](09-kv-cache-generation-loop.md)).
4. Pelo menos **uma tool call real** funciona ponta a ponta via o parser `lfm2` do vLLM.
5. Round-trip p50 dentro de NFR-L6 em execução sustentada de 30 min.
6. A **quantização escolhida como default** foi validada contra o baseline BF16 (qualidade não
   degrada de forma perceptível em VQA/tool-calling/grounding — ver [`11`](11-testing-eval.md)).
7. Todos os benchmarks de latência ([`11`](11-testing-eval.md)) passam no CI, incluindo o
   cold-start do vLLM documentado e não bloqueando turnos prematuramente (NFR-R6).
8. A licença do LFM2.5-VL-3B (NFR-P4) foi registrada e confirmada compatível com o uso
   pretendido.
