# 03 — Plataforma de Hardware e Runtime

## 1. Plataforma alvo

| | **AGX Thor (128 GB)** — única plataforma alvo |
|---|---|
| GPU | Blackwell, 2560 CUDA / 96 Tensor, arquitetura **SM_110a** |
| Compute IA | ~2.070 TFLOPS (FP4, sparse) |
| Memória | 128 GB LPDDR5X, > 270 GB/s |
| CPU | 14× Arm Neoverse-V3AE |
| Potência | 40–130 W |
| FP4 nativo | **sim** (Blackwell) — usado por **NVFP4**, formato de quantização nativo |

> **Verificado no device real em 2026-09-03:** `nvidia-smi` reporta GPU `NVIDIA Thor`, driver
> 595.78, **CUDA 13.2**; `nv_tegra_release` reporta build **R39** (linha JetPack 7.x); 122 GB de
> RAM livres de 128 GB total. Este repositório é desenvolvido e testado diretamente na
> plataforma alvo — não há mais um cenário "Orin como fallback" no plano ativo (o Orin fica
> registrado como historicamente considerado, mas fora do escopo corrente).

Fontes: [Jetson AGX Thor vs AGX Orin — Forecr](https://www.forecr.io/blogs/all/nvidia-jetson-orin-family-vs-thor-what-you-need-to-know),
[Introducing NVIDIA Jetson Thor — NVIDIA Developer Blog](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/).

## 2. Software base

- **JetPack 7.x** é o SDK oficial (Linux for Tegra + CUDA + cuDNN + TensorRT + multimídia).
  Fonte: [JetPack — NVIDIA](https://developer.nvidia.com/embedded/jetpack).
- **vLLM** é o runtime de inferência do LFM2.5-VL-3B em Jetson Thor (ver [`13`](13-vllm-deployment.md)
  para o guia completo de deploy). Dois caminhos documentados:
  1. **Container oficial da NVIDIA**: `ghcr.io/nvidia-ai-iot/vllm:...-tegra-aarch64-...` —
     confirmar a tag exata compatível com JetPack 7.x/CUDA 13.x no momento do deploy (a NVIDIA
     publica atualizações frequentes; a tag `r36.4-tegra-aarch64-cu126-22.04` foi vista em
     pesquisa mas pode não ser a mais recente).
  2. **`thorllm`** ([GitHub](https://github.com/ms1design/thorllm)) — wrapper comunitário que
     instala vLLM via wheel pré-compilada `cu130`, cria venv isolado e configura um serviço
     systemd; útil para gestão via CLI/TUI sem depender de containers.
- **Jetson AI Lab** tem tutoriais de referência para modelos on-device:
  [Jetson AI Lab — Tutorials](https://www.jetson-ai-lab.com/tutorials/).
- **TensorRT-LLM** não é mais o caminho primário (ver [`01`](01-architecture.md) §7) — fica
  registrado como fallback a reavaliar, não como dependência ativa.

> **Nota de versão:** confirme a matriz de compatibilidade JetPack ↔ CUDA ↔ vLLM no device real
> antes de fixar versões em produção. Registrar as versões exatas em `config.yaml` e no README
> de deploy (NFR-R5).

## 3. Runtimes por componente

| Componente | Runtime | Precisão alvo |
|-----------|---------|----------------|
| Silero VAD | ONNX Runtime (CUDA EP ou CPU) | FP32/INT8 |
| faster-whisper | CTranslate2 (CUDA) | **INT8** (ou int8_float16) |
| **LFM2.5-VL-3B** (visão + linguagem + tool calling, tudo em um) | **vLLM** (fallback: `llama.cpp` GGUF — ver [`13`](13-vllm-deployment.md) §8) | BF16 (default inicial) → FP8/NVFP4 após validação (M2) |
| XTTS-v2 | PyTorch + CUDA (DeepSpeed opcional p/ acelerar) | FP16 |

## 4. Orçamento de memória (estimativa, Opção A revisada)

Valores por checkpoint do **LFM2.5-VL-3B**, conforme quantização (tamanhos de arquivo reais,
confirmados via Hugging Face em 2026-09-03; validar footprint em runtime — inclui overhead do
runtime, não só os pesos):

| Quantização | Tamanho do checkpoint | Runtime | Observação |
|---|---|---|---|
| BF16 / F16 | 5.4 GB | vLLM | Default recomendado para começar (M1/M2) — sem risco de qualidade |
| Q8_0 (GGUF) | 2.87 GB | llama.cpp (fallback) | |
| Q6_K (GGUF) | 2.22 GB | llama.cpp (fallback) | |
| Q5_K_M (GGUF) | 1.94 GB | llama.cpp (fallback) | |
| Q4_K_M (GGUF) | 1.67 GB | llama.cpp (fallback) | |
| Q4_0 (GGUF) | 1.59 GB | llama.cpp (fallback) | menor footprint, mais risco de qualidade |
| FP8 (community, LLM Compressor+AutoRound) | ~2.7–3 GB (estimado, não confirmado) | vLLM | camadas de convolução do LIV **não** são quantizadas (necessário p/ rodar no vLLM) |
| NVFP4 (community, Blackwell-nativo) | ~1.5–2 GB (estimado, não confirmado) | vLLM | **não benchmarked publicamente** — validar qualidade e latência no device antes de adotar como default |

Orçamento total no device (todos os modelos residentes, cenário BF16 conservador):

| Item | Memória |
|------|---------|
| faster-whisper large-v3 INT8 | ~1.5–3 GB |
| **LFM2.5-VL-3B** (BF16) + pool de KV cache do vLLM | ~5.4 GB + reservar 2–4 GB de KV |
| XTTS-v2 | ~2 GB |
| Silero VAD | < 0.1 GB |
| Buffers de vídeo/áudio/framework | ~1–2 GB |
| **Total aproximado** | **~13–17 GB / 128 GB** |

Conclusão: folga enorme mesmo sem quantizar. A motivação para quantizar (FP8/NVFP4) não é
"caber na memória" — é **latência/throughput**; validar ganho real antes de trocar o default
(ver [`13`](13-vllm-deployment.md) §5).

## 5. Câmera e áudio

- **Câmera:** CSI (via `nvarguscamerasrc`/GStreamer) ou USB/UVC (`v4l2`). Preferir pipeline
  GStreamer com aceleração NVMM para evitar cópias CPU↔GPU. Amostrar frames no fps de
  trabalho (default 2), não no fps nativo da câmera.
- **Áudio:** captura em 16 kHz mono PCM (formato esperado por VAD e STT). Playback do TTS em
  paralelo, com **cancelamento de eco/AEC** recomendado (senão o VAD escuta o próprio TTS e
  dispara barge-in falso — ver [`08`](08-orchestration-barge-in.md)).

## 6. Modo de energia

Fixar o device em modo de máxima performance para benchmarks (`nvpmodel` + `jetson_clocks`).
Documentar o modo de energia junto de cada número de latência — potência afeta clocks e,
portanto, latência (Thor: 40–130 W).
