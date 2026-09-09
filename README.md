# Edge Multimodal Assistant — Documentação de Requisitos

Assistente multimodal **on-device** para NVIDIA Jetson (**AGX Thor**) que:

- **Ouve** o usuário (STT com `faster-whisper` + VAD)
- **Vê** em tempo real (LFM2.5-VL-3B — VLM agentic **pronto**, visão nativa)
- **Fala** em português do Brasil com **clonagem de voz** (XTTS-v2 streaming)
- **Raciocina e chama ferramentas** (tool calling nativo do LFM2.5-VL-3B, texto ou imagem)
- Permite **interrupção do TTS pelo usuário** (barge-in) e mantém latência conversacional

> **Decisão de arquitetura travada (revisada em 2026-09-03):** Opção A — **um único cérebro**,
> agora o **LFM2.5-VL-3B** da Liquid AI, servido via **vLLM**. É um VLM agentic já treinado
> (backbone LFM2.5-2.6B + SigLIP2 NaFlex 400M), com tool calling nativo a partir de texto ou
> imagem — **não treinamos um conector de visão**. A decisão anterior (treinar um conector para
> acoplar visão a um 2.6B text-only) foi substituída ao descobrir que a Liquid já publica esse
> exato acoplamento pronto, validado e com parser de tool calling dedicado no vLLM. Ver
> [`docs/01-architecture.md`](docs/01-architecture.md).

## Para quem é esta pasta

Esta documentação foi escrita para ser consumida por um agente de codificação
(**Claude Code**) trabalhando no repositório, e por engenheiros humanos revisando o
projeto. Ela é **normativa e extensiva**: descreve requisitos, contratos de interface,
orçamentos de latência, o laço de geração, o contrato de turno/sessão e o plano de
desenvolvimento faseado.

**Comece por [`CLAUDE.md`](CLAUDE.md)** — ele diz ao agente como navegar estes documentos
e quais invariantes nunca violar.

## Índice da documentação

| Documento | Conteúdo |
|-----------|----------|
| [`docs/00-project-overview.md`](docs/00-project-overview.md) | Visão, objetivos, escopo, glossário |
| [`docs/01-architecture.md`](docs/01-architecture.md) | Arquitetura Opção A (revisada), diagrama de blocos, fluxo de dados |
| [`docs/02-requirements.md`](docs/02-requirements.md) | Requisitos funcionais (FR) e não-funcionais (NFR), orçamento de latência |
| [`docs/03-hardware-platform.md`](docs/03-hardware-platform.md) | Jetson Thor, JetPack, CUDA, vLLM, alocação de memória |
| [`docs/04-stt-faster-whisper.md`](docs/04-stt-faster-whisper.md) | STT streaming com faster-whisper + Silero VAD |
| [`docs/05-vision-encoder-coupling.md`](docs/05-vision-encoder-coupling.md) | Capacidades multimodais nativas do LFM2.5-VL-3B (sem treino no MVP) |
| [`docs/06-llm-brain-generation.md`](docs/06-llm-brain-generation.md) | LFM2.5-VL-3B, tool calling, serving via vLLM |
| [`docs/07-tts-voice-cloning.md`](docs/07-tts-voice-cloning.md) | XTTS-v2 clonagem de voz + streaming em pt-BR |
| [`docs/08-orchestration-barge-in.md`](docs/08-orchestration-barge-in.md) | Pipecat, turn-taking, barge-in, máquina de estados |
| [`docs/09-kv-cache-generation-loop.md`](docs/09-kv-cache-generation-loop.md) | Laço de geração, contrato de turno/sessão, injeção de frames de vídeo |
| [`docs/10-roadmap-milestones.md`](docs/10-roadmap-milestones.md) | Plano de desenvolvimento faseado (M0–M6) |
| [`docs/11-testing-eval.md`](docs/11-testing-eval.md) | Estratégia de testes, benchmarks de latência, avaliação |
| [`docs/12-references.md`](docs/12-references.md) | Todas as documentações oficiais citadas |
| [`docs/13-vllm-deployment.md`](docs/13-vllm-deployment.md) | Deploy e operação do vLLM no Jetson Thor |

## Stack resumido

| Função | Componente | Runtime alvo |
|--------|-----------|--------------|
| VAD / endpointing | Silero VAD | ONNX Runtime (CUDA/CPU) |
| STT | faster-whisper (`large-v3` / `large-v3-turbo`, INT8) | CTranslate2 (CUDA) |
| Cérebro + visão + tool calling | **LFM2.5-VL-3B** (VLM agentic pronto, ~3,1B params) | **vLLM** (BF16/FP8/NVFP4) |
| TTS pt-BR + voice cloning | XTTS-v2 (`inference_stream`) | PyTorch/CUDA (+ DeepSpeed opcional) |
| Orquestração | Pipecat | Python asyncio |

## Status

`M0 — Bootstrap concluído`. Estrutura de repositório, `config.yaml` tipado (pydantic), contratos
(`contracts/`), serviços (`services/{vad,stt,vision,llm,tts}`), orquestrador com máquina de
estados + barge-in (`orchestrator/`), registro de ferramentas (`tools/`), deploy do vLLM
data-driven (`deployment/`), harness de benchmarks (`benchmarks/`) e suíte de testes de contrato
(34 testes, mocks, sem GPU) — tudo em pé e passando (`pytest -q`, `ruff check .`).

Os backends pesados (faster-whisper, XTTS-v2, silero-vad, vLLM/pesos do LFM2.5-VL-3B) usam
import lazy e **não foram baixados/instalados automaticamente** — isso é uma ação pesada
(múltiplos GB, potencialmente horas de JIT do vLLM) que fica para quando alguém decidir rodar
M1 de verdade no device. Ver `pyproject.toml` (extras `vad`/`stt`/`tts`) e `deployment/README.md`.

Próximo portão: **M1** — subir cada componente isolado no device e medir latência/memória reais
(docs/10).
