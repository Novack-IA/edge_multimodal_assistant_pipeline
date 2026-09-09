# 06 — Cérebro: LFM2.5-VL-3B, Tool Calling e Serving via vLLM

## 1. O modelo

**LFM2.5-VL-3B** — VLM agentic on-device da Liquid AI, publicado **já treinado** em 12/ago/2026
(ver [`05`](05-vision-encoder-coupling.md) para arquitetura e capacidades nativas).

Características relevantes:
- **~3,1B** parâmetros (backbone LFM2.5-2.6B + SigLIP2 NaFlex 400M + projetor).
- **Contexto de 32.768 tokens** — teto real a respeitar (ver [`09`](09-kv-cache-generation-loop.md) §6).
- Footprint **~3 GB** em BF16 (menos quantizado — ver [`03`](03-hardware-platform.md) §4).
- **Não-reasoning**: responde direto, sem cadeia de pensamento visível — latência previsível,
  adequado ao orçamento conversacional (NFR-L4/L6).
- **Licença: LFM Open License v1.0** (`lfm1.0`) — uso comercial gratuito até US$10M de receita
  anual (ver [`02`](02-requirements.md) NFR-P4).
- **Tool calling nativo** a partir de **texto ou imagem** (ToolSandbox 59.5, BFCL v4 32.5).

Fontes: [LFM2.5-VL-3B — Liquid Docs](https://docs.liquid.ai/lfm/models/lfm25-vl-3b),
[LFM2.5-VL-3B — Liquid AI Blog](https://www.liquid.ai/blog/lfm2-5-vl-3b),
[LiquidAI/LFM2.5-VL-3B — Hugging Face](https://huggingface.co/LiquidAI/LFM2.5-VL-3B).

## 2. Runtime: vLLM

Servimos o modelo via **vLLM**, rodando como **processo/servidor separado** no device (container
oficial da NVIDIA ou via `thorllm` — ver [`13`](13-vllm-deployment.md) para o guia completo de
deploy/operação). O orquestrador fala com ele via **cliente HTTP assíncrono streaming** contra a
API OpenAI-compatible.

Comando de referência (detalhes e matriz de quantização em [`13`](13-vllm-deployment.md)):

```bash
vllm serve LiquidAI/LFM2.5-VL-3B \
  --enable-auto-tool-choice \
  --tool-call-parser lfm2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.5 \
  --host 127.0.0.1 --port 8000
```

Recursos que usamos, todos nativos do vLLM (não implementados por nós):
- **PagedAttention** (KV cache paginado) e **Automatic Prefix Caching** (reuso de prefixo
  idêntico entre requisições — cobre nosso `[system][histórico]` estável).
- **Continuous batching** (equivalente ao in-flight batching do plano original).
- **Streaming SSE** via `/v1/chat/completions` com `stream=true`.
- Suporte multimodal nativo (conteúdo de imagem na mensagem).

> **Primeira subida é lenta (5–15 min)** por JIT de kernels Triton/FlashInfer para a arquitetura
> Blackwell/SM_110a — ver [`13`](13-vllm-deployment.md) §4 e NFR-R6. Isso é operacional, não um
> bug de latência de turno.

## 3. Tool calling

- Formato nativo **pythonic**: `<|tool_call_start|>[nome_func(arg="valor")]<|tool_call_end|>`.
- O **parser `lfm2` do vLLM** (`--enable-auto-tool-choice --tool-call-parser lfm2`) converte isso
  automaticamente no array `tool_calls` padrão OpenAI — não escrevemos parsing próprio.
- Ferramentas ficam em `tools/`, cada uma com: nome, JSON schema de argumentos, função async,
  timeout, e tratamento de erro.
- Laço agentic (ReAct-like): o modelo emite `tool_calls` → orquestrador executa → resultado
  entra como mensagem `role="tool"` → nova requisição ao vLLM continua o raciocínio. Ver
  interação com o contrato de turno em [`09`](09-kv-cache-generation-loop.md).
- **Barge-in durante tool call:** se o usuário interromper enquanto uma ferramenta roda,
  cancelar a coroutine da ferramenta (ou descartar o resultado) e devolver o turno. Ferramentas
  com efeitos colaterais precisam ser idempotentes ou canceláveis.
- Validar cedo (M2) que o tool calling funciona **também quando disparado pelo conteúdo da
  imagem** (FR-11b), não só por texto — é um caso de teste diferente do "tool calling de texto"
  que a maioria dos exemplos da comunidade cobre.

## 4. Contrato de geração (streaming)

O cliente do LLM expõe um gerador assíncrono de eventos sobre a API streaming do vLLM:

```python
# services/llm/client.py (contrato)
async def stream_chat(
    messages: list[dict],   # OpenAI-style; imagens já embutidas no turno atual (doc 05 §3)
    tools: list[ToolSpec],
    sampling: SamplingCfg,
    cancel_token,            # setado no barge-in → fecha a conexão HTTP/aborta
) -> AsyncIterator[LlmToken | ToolCall]:
    """POST streaming em {server_url}/v1/chat/completions (stream=true).
    Cada chunk SSE vira um LlmToken; tool_calls do parser lfm2 viram ToolCall.
    Fecha a conexão imediatamente se cancel_token estiver setado."""
```

Requisitos:
- **TTFT ≤ 250 ms p50** (NFR-L4, agora incluindo prefill de imagem — ver [`02`](02-requirements.md) §2).
- **≥ 30 tok/s sustentado** (NFR-L8).
- Emitir tokens assim que chegam via SSE (sem esperar o evento final).
- Respeitar `cancel_token` **a cada chunk** — barge-in tem que fechar a conexão em ≤ 200 ms.

## 5. Amostragem

- Diálogo: `temperature` baixa (0.3–0.7), `top_p` ~0.9. Expor no config.
- Tool calling: preferir determinismo maior (temperatura menor) para chamadas confiáveis.
- `max_new_tokens` por turno limitado (ex.: 256) para não gerar monólogos que estouram
  latência de TTS; respostas faladas são curtas por natureza.
- **Não usar** variantes "Thinking" da família LFM2.5 (ex.: LFM2.5-1.2B-Thinking) — o
  LFM2.5-VL-3B é intencionalmente não-reasoning para manter latência baixa; não há necessidade
  de desabilitar um modo de raciocínio que ele não tem.

## 6. Riscos

| Risco | Mitigação |
|-------|-----------|
| Quantização (FP8/NVFP4) degrada tool calling/grounding | benchmark pós-quantização vs. baseline BF16 antes de trocar o default (ver [`11`](11-testing-eval.md)) |
| NVFP4 comunitário nunca foi benchmarked publicamente | medir qualidade e latência no device antes de adotar (não assumir "Blackwell-nativo" = "sempre melhor") |
| Contexto de 32K é bem menor que os 128K assumidos no plano original | `history_max_turns` mais conservador + frames nunca persistem no histórico (doc 09 §6) |
| Cold-start do vLLM (5–15 min JIT) atrasa disponibilidade | NFR-R6: warm-up explícito no boot, não turno-a-turno |
| vLLM em Jetson Thor é plataforma nova (poucos meses de maturidade pública) | pinar versão exata (NFR-R5); ter fallback documentado via `llama.cpp` ([`13`](13-vllm-deployment.md) §8) |
| Tool calling motivado por imagem não testado tão amplamente quanto por texto | suíte de teste dedicada em M2 (FR-11b) |
