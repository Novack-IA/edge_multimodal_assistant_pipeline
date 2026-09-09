# 01 — Arquitetura

## 1. Decisão central: Opção A revisada — cérebro único, pronto

Há **exatamente um modelo** no sistema: o **LFM2.5-VL-3B**, um VLM agentic publicado **já
treinado** pela Liquid AI em 12/ago/2026. Internamente ele é composto por um backbone
**LFM2.5-2.6B**, um encoder de visão **SigLIP2 NaFlex 400M** e um projetor multimodal — mas essa
composição já foi feita e treinada pela Liquid; nós **não construímos nem treinamos** esse
acoplamento. ~3,1B parâmetros no total, ~3 GB em memória (BF16), contexto de 32K tokens.

> **Por que isso mudou em relação ao plano original:** a versão anterior desta doc assumia que
> teríamos que treinar um conector para acoplar visão a um LFM2.5-2.6B text-only, replicando o
> padrão LFM2-VL com um backbone diferente. Pesquisa em 2026-09-03 mostrou que a Liquid **já
> publica esse exato tipo de acoplamento pronto e validado** — LFM2.5-VL-3B é literalmente
> "LFM2.5-2.6B + SigLIP2 NaFlex + projetor", já treinado, com tool calling nativo a partir de
> texto **ou** imagem (ToolSandbox 59.5, BFCL v4 32.5), grounding (RefCOCO P@1 87.9) e leitura
> de tela/documento (ScreenSpot-v2 80.7 médio). Reconstruir isso do zero duplicaria meses de
> trabalho de treino da própria Liquid para benefício incerto, e reintroduziria exatamente o
> risco que o plano antigo tentava mitigar (regressão de tool calling por fine-tune malfeito).
> Ver [`05`](05-vision-encoder-coupling.md) para os detalhes e [`12`](12-references.md) para as
> fontes.

**Consequência de engenharia:** o trabalho deixa de ser "treinar um conector" e passa a ser
**"servir bem um modelo pronto"** — escolher runtime, quantização e integrá-lo ao laço de
voz/vídeo/barge-in com a mesma disciplina de latência e estado explícito de antes. O runtime
escolhido é **vLLM** (não TensorRT-LLM — ver §7).

## 2. Diagrama de blocos

```
                         ┌──────────────────────────────────────────────┐
   🎤 microfone          │                ORQUESTRADOR (Pipecat)         │
        │                │   máquina de estados de turno + barge-in      │
        ▼                │                                               │
   ┌─────────┐   fala?   │   IDLE → LISTENING → THINKING → SPEAKING      │
   │ Silero  │──────────▶│              ▲            │         │          │
   │  VAD    │  barge-in │              │ interrupt  │         │          │
   └─────────┘───────────┼──────────────┘            │         │          │
        │ segmentos      └───────────────────────────┼─────────┼──────────┘
        ▼                                            │         │
   ┌──────────────┐  texto parcial/final             │         │
   │ faster-      │──────────────┐                    │         │
   │ whisper (STT)│              │                    │         │
   └──────────────┘              ▼                    │         │
                        ┌────────────────────────────────────┐  │
   📷 câmera            │   cliente HTTP (streaming) ──────▶  │  │
        │               │   vLLM :: LFM2.5-VL-3B (processo    │  │
        ▼               │   separado, doc 13)                 │  │
   ┌──────────┐ frames  │                                      │  │
   │ captura+ │────────▶│  mensagens OpenAI-style:              │  │
   │ amostragem│ PIL/   │   [system] [histórico texto]          │  │
   │ (1–4 fps)│ base64  │   [user: imagens da janela + texto]   │  │
   └──────────┘         │                                      │  │
                        │   (encoder de visão + tool calling    │  │
                        │    já embutidos no modelo servido)    │  │
                        │        │  streaming SSE + tool_calls  │  │
                        │        ▼                              │  │
                        │   texto (stream)  +  tool calls        │  │
                        └───────┬───────────────────┬───────────┘  │
                                │                   │              │
                                │            ┌──────▼──────┐       │
                                │            │ ferramentas │       │
                                │            │ (async)     │       │
                                │            └──────┬──────┘       │
                                │                   │ resultado    │
                                │◀──────────────────┘              │
                                ▼                                  │
                        ┌───────────────┐   chunks de áudio        │
                        │ XTTS-v2 (TTS) │─────────────────────────▶│──▶ 🔊
                        │ voz clonada   │   (interrompível)         │
                        └───────────────┘                           │
```

## 3. Fluxo de dados (caminho feliz)

1. **Áudio** entra em blocos (ex. 20–30 ms). O **Silero VAD** classifica fala/silêncio.
2. Ao detectar início de fala, o orquestrador vai para `LISTENING`. Áudio é enviado ao
   **faster-whisper**, que emite **transcrições parciais** e, no endpoint (silêncio
   sustentado), a **final**.
3. Em paralelo e continuamente, a **câmera** é amostrada. Frames selecionados ficam num **ring
   buffer** de imagens brutas (PIL/numpy) — **não há mais um estágio de encoder nosso**; a
   codificação de visão acontece dentro do próprio LFM2.5-VL-3B quando o vLLM processa a
   requisição.
4. No endpoint da fala, o orquestrador monta a lista de mensagens: `[system] + [histórico
   consolidado (texto)] + [user: frames da janela atual como conteúdo de imagem + texto
   transcrito]`, entra em `THINKING`, e abre uma requisição **streaming** (SSE) ao **vLLM**.
5. Se a resposta contém uma **tool call** (extraída pelo parser `lfm2` do vLLM), o orquestrador
   a executa (assíncrono) e devolve o resultado como mensagem `tool` na mesma sequência
   (padrão ReAct/agentic).
6. Tokens de texto vão sendo agrupados em frases/cláusulas e enviados ao **XTTS-v2**, que
   sintetiza **chunks de áudio em streaming**. O orquestrador vai para `SPEAKING` e toca o
   áudio.
7. Ao fim do turno, só o **texto** (usuário + resposta) é consolidado no histórico persistido —
   os frames do turno são descartados (ver [`09`](09-kv-cache-generation-loop.md)).
8. Se o **VAD** detectar fala do usuário durante `SPEAKING` ou `THINKING` → **barge-in**: aborta
   o stream HTTP do vLLM e a síntese do TTS, volta a `LISTENING`.

## 4. Concorrência e isolamento

- O **orquestrador** roda em asyncio (Pipecat).
- **STT e TTS** são workers pesados de GPU, cada um em processo/thread próprio com fila de
  entrada e stream de saída. O event loop nunca bloqueia em inferência.
- O **LFM2.5-VL-3B roda como servidor vLLM em processo separado** (container ou systemd, ver
  [`13`](13-vllm-deployment.md)), com seu próprio scheduler de continuous batching. O
  orquestrador fala com ele por um **cliente HTTP assíncrono streaming** contra a API
  OpenAI-compatible (`/v1/chat/completions`).
- A **captura de vídeo** roda como um produtor contínuo que atualiza um buffer compartilhado
  (a janela de vídeo, agora só frames brutos — muito mais leve que antes, sem GPU dedicada). O
  cliente do vLLM lê o snapshot dessa janela no momento em que monta a mensagem do turno.
- **Ferramentas** são coroutines; uma ferramenta lenta não pode travar o laço — usa timeout.

## 5. Alocação de GPU (visão geral; detalhe em [`03`](03-hardware-platform.md))

Todos os modelos residentes simultaneamente na memória unificada da Jetson Thor (128 GB):

| Modelo | Memória aprox. | Runtime |
|--------|------------------|---------|
| Silero VAD | ~pequeno (< 100 MB) | ONNX Runtime |
| faster-whisper large-v3 INT8 | ~1.5–3 GB | CTranslate2/CUDA |
| **LFM2.5-VL-3B** (servido pelo vLLM) | 1.6–5.4 GB conforme quantização (ver [`03`](03-hardware-platform.md) §4) + pool de KV cache reservado pelo vLLM | **vLLM** |
| XTTS-v2 | ~2 GB | PyTorch/CUDA |

Folgado na Thor (128 GB) mesmo em BF16 sem quantizar. O gargalo prático segue sendo **banda de
memória**, mas o LFM2.5-VL-3B é bem menor que o orçamento original (2.5–3 GB do 2.6B + 0.5–1 GB
do encoder separado ≈ igual ao 3B combinado hoje, só que já treinado).

## 6. Contratos entre estágios (resumo)

Definidos em `contracts/` (ver [`CLAUDE.md`](../CLAUDE.md)). Esboço revisado — `VisionEmbedding`
foi substituído por `VideoFrame` (frame bruto, não mais embedding pré-computado nosso), e o
transporte para o cérebro passa a ser uma lista de mensagens OpenAI-style em vez de embeddings
mesclados manualmente:

```python
# contracts/messages.py
@dataclass
class AudioChunk:      # microfone → VAD/STT
    pcm: bytes; sample_rate: int; t_capture: float

@dataclass
class VadEvent:        # VAD → orquestrador
    kind: Literal["speech_start", "speech_end"]; t: float

@dataclass
class Transcript:      # STT → orquestrador
    text: str; is_final: bool; t_start: float; t_end: float

@dataclass
class VideoFrame:      # captura → janela de vídeo (SEM encoder nosso — doc 05)
    image: "PIL.Image | np.ndarray"; frame_id: int; t: float

@dataclass
class LlmToken:        # vLLM → orquestrador (stream SSE)
    text: str; is_tool_call: bool; done: bool

@dataclass
class ToolCall:        # vLLM (via parser lfm2) → ferramentas
    name: str; arguments: dict

@dataclass
class ToolResult:       # ferramentas → vLLM (mensagem role="tool")
    name: str; content: str; is_error: bool; t: float

@dataclass
class AudioOut:        # TTS → playback (stream, interrompível)
    pcm: bytes; is_last: bool
```

## 7. Por que vLLM, não TensorRT-LLM

O plano original assumia TensorRT-LLM como runtime do LLM. Isso é revisado porque:

- **Suporte oficial e específico.** A NVIDIA publica um **container oficial de vLLM para Jetson
  Thor** (`ghcr.io/nvidia-ai-iot/vllm:...-tegra-aarch64...`), com quantização NVFP4 nativa de
  Blackwell documentada e medida no próprio blog técnico da NVIDIA. TensorRT-LLM em Jetson tem
  suporte inicial só validado publicamente até Orin (branch `v0.12.0-jetson`); não há caminho de
  export documentado para a arquitetura híbrida conv+attention do LFM2 no TensorRT-LLM.
- **Parser de tool calling dedicado.** O vLLM tem um parser (`lfm2`) feito especificamente para
  o formato pythonic `<|tool_call_start|>[func(...)]<|tool_call_end|>` usado pela família
  LFM2/LFM2.5 — ativado com `--enable-auto-tool-choice --tool-call-parser lfm2`. Isso é
  exatamente o contrato de tool calling que o modelo já fala nativamente.
- **PagedAttention + continuous batching + Automatic Prefix Caching já vêm prontos** — as
  mesmas propriedades que o plano original ia implementar manualmente sobre TensorRT-LLM
  (paged KV cache, in-flight batching, reuso de prefixo) são o núcleo do próprio vLLM.
- **Multimodal de primeira classe.** A API `/v1/chat/completions` do vLLM aceita conteúdo de
  imagem diretamente na mensagem; o processor multimodal do modelo roda dentro do servidor.

TensorRT-LLM fica registrado como opção a reavaliar só se o vLLM não atingir os orçamentos de
latência no device (ver [`13`](13-vllm-deployment.md) §8 para o fallback via `llama.cpp`, que é
o caminho mais maduro hoje fora do vLLM).

## 8. Por que não a Opção B (dois modelos)

Registrado para posteridade: dois LLMs (VL "olhos" + agentic "cérebro") somam latência de um
hop extra e duplicam pressão de memória/banda — justamente o recurso escasso na Jetson. A
Opção A concentra tudo num modelo. Diferente do plano original, hoje **nem pagamos o custo do
treino do conector offline** — a Liquid já pagou esse custo e publicou o resultado.
