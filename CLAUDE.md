# CLAUDE.md — Instruções para o agente de codificação

Este arquivo orienta o **Claude Code** (ou qualquer agente) trabalhando neste repositório.
Leia-o inteiro antes de escrever qualquer código.

> **Decisão de arquitetura revisada em 2026-09-03** (pesquisa web): o cérebro do sistema não é
> mais um LLM text-only com um conector de visão treinado por nós. É o **LFM2.5-VL-3B**, um VLM
> agentic **pronto**, publicado pela Liquid AI em 12/ago/2026 (backbone LFM2.5-2.6B + SigLIP2
> NaFlex 400M + projetor, ~3,1B parâmetros, já treinado com tool calling nativo a partir de texto
> **ou** imagem). Ver [`docs/01`](docs/01-architecture.md) e [`docs/05`](docs/05-vision-encoder-coupling.md)
> para o histórico da decisão anterior e por que foi substituída.

## O que este projeto é

Um assistente de voz+visão multimodal em tempo real que roda **inteiramente on-device** em
NVIDIA Jetson (AGX Thor). O usuário fala, o sistema vê a cena por câmera, raciocina, opcionalmente
chama ferramentas, e responde falando em português do Brasil com uma voz clonada. O usuário pode
**interromper a fala do sistema a qualquer momento** (barge-in).

## Invariantes que você NUNCA deve violar

1. **Um único modelo (Opção A revisada).** Há exatamente um modelo no laço: o **LFM2.5-VL-3B**,
   servido via **vLLM**. Nunca introduza um segundo LLM/VLM "descritor de imagem". A visão já
   vem acoplada e treinada pela Liquid — **não treinamos conector nem fazemos fine-tune no
   caminho crítico do MVP** (fine-tune de domínio é opcional, M6 — ver [`docs/05`](docs/05-vision-encoder-coupling.md)).
2. **Tudo local por padrão.** Nenhum componente do laço quente pode depender de rede/nuvem — o
   próprio servidor vLLM roda **no device**. Chamadas externas só são permitidas dentro de
   *ferramentas* (tool calls) explicitamente declaradas, e devem ser assíncronas e não bloquear
   o laço.
3. **Streaming ponta a ponta.** STT emite parciais; o vLLM emite o primeiro token via streaming
   SSE o quanto antes; o TTS sintetiza o primeiro trecho antes de a resposta terminar. Nada de
   "esperar o áudio inteiro" em nenhum estágio.
4. **Barge-in é sagrado.** Detecção de fala do usuário durante o TTS deve **interromper a
   síntese e a geração do vLLM em < 200 ms** (abortando a requisição HTTP em streaming) e
   devolver o turno ao usuário. Ver [`docs/08`](docs/08-orchestration-barge-in.md) e
   [`docs/09`](docs/09-kv-cache-generation-loop.md).
5. **O contrato de turno/sessão é estado explícito, não implícito** — mesmo com o KV cache de
   baixo nível delegado ao vLLM (PagedAttention + Automatic Prefix Caching). O ciclo de vida da
   **lista de mensagens** (histórico consolidado, bloco de frames do turno, commit/descarte em
   barge-in) é definido em [`docs/09`](docs/09-kv-cache-generation-loop.md). Não delegue isso a
   comportamento default de biblioteca sem documentar.
6. **Orçamento de latência é um requisito, não uma aspiração.** Todo PR que toca o laço quente
   deve rodar o benchmark de latência ([`docs/11`](docs/11-testing-eval.md)) e não regredir os
   alvos de [`docs/02`](docs/02-requirements.md).

## Ordem de leitura recomendada

1. [`docs/00-project-overview.md`](docs/00-project-overview.md) — contexto
2. [`docs/02-requirements.md`](docs/02-requirements.md) — o que "pronto" significa
3. [`docs/01-architecture.md`](docs/01-architecture.md) — como as peças se encaixam
4. [`docs/09-kv-cache-generation-loop.md`](docs/09-kv-cache-generation-loop.md) — o coração do sistema
5. Specs de componente ([`04`](docs/04-stt-faster-whisper.md), [`05`](docs/05-vision-encoder-coupling.md),
   [`06`](docs/06-llm-brain-generation.md), [`07`](docs/07-tts-voice-cloning.md),
   [`08`](docs/08-orchestration-barge-in.md), [`13`](docs/13-vllm-deployment.md)) conforme a tarefa

## Convenções de código

- **Linguagem:** Python 3.10+ para orquestração e serviços; C++/CUDA só onde um kernel
  custom for justificado por profiling (hoje, nenhum é esperado — vLLM cobre o LLM/VLM).
- **Concorrência:** `asyncio` no orquestrador (Pipecat é asyncio-native). Cada modelo pesado
  (STT, TTS) roda em seu próprio worker/processo com fila; o **LLM/VLM roda como servidor vLLM
  separado**, acessado por um cliente HTTP assíncrono streaming. O event loop nunca bloqueia
  em inferência.
- **Config:** um único `config.yaml` tipado (pydantic-settings). Sem constantes mágicas
  espalhadas. Modelos, caminhos de pesos, thresholds de VAD, tamanhos de janela de vídeo,
  quantização do vLLM — tudo no config.
- **Contratos entre estágios:** definidos como dataclasses/`TypedDict` em um módulo `contracts/`.
  Um estágio nunca depende do formato interno de outro.
- **Logging estruturado** com timestamps de alta resolução em cada fronteira de estágio (para
  reconstruir a cascata de latência offline).
- **Testes:** todo componente do laço quente tem (a) teste unitário de contrato e (b) teste de
  latência com orçamento (falha o CI se estourar). Ver [`docs/11`](docs/11-testing-eval.md).

## Estrutura de repositório sugerida

```
edge-multimodal-assistant/
├── CLAUDE.md
├── README.md
├── config.yaml
├── docs/                      # esta documentação
├── contracts/                 # dataclasses de mensagens entre estágios
├── services/
│   ├── vad/                   # Silero VAD
│   ├── stt/                   # faster-whisper worker
│   ├── vision/                # captura + amostragem de frames (SEM encoder próprio — doc05)
│   ├── llm/                   # cliente vLLM (OpenAI-compatible) + laço agentic + contrato de turno (doc09)
│   └── tts/                   # XTTS-v2 streaming
├── orchestrator/               # pipeline Pipecat + máquina de estados
├── tools/                      # ferramentas expostas ao tool calling
├── deployment/                 # launch config do vLLM, systemd/container, matriz de quantização (doc13)
├── training/                   # OPCIONAL (M6): fine-tune LoRA de domínio — não é caminho crítico do MVP
├── benchmarks/                 # scripts de latência/throughput
└── tests/
```

## Regra de ouro

Quando uma decisão não estiver coberta aqui ou nos `docs/`, **pare e pergunte** em vez de
improvisar — especialmente qualquer coisa que toque no laço de geração, contrato de
turno/sessão ou barge-in. Uma escolha errada nessas três áreas contamina o sistema inteiro.
