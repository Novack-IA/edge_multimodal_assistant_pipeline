# 12 — Referências

Documentações oficiais e fontes citadas ao longo desta pasta. Verificar sempre a versão
correspondente ao device (JetPack/CUDA/vLLM/Transformers) antes de implementar.

## Modelo VLM pronto: LFM2.5-VL-3B (Liquid AI) — decisão central, revisada 2026-09-03
- LFM2.5-VL-3B — [Liquid Docs](https://docs.liquid.ai/lfm/models/lfm25-vl-3b)
- LFM2.5-VL-3B: A Better and Faster Vision-Language Model for the Edge — [Liquid AI Blog](https://www.liquid.ai/blog/lfm2-5-vl-3b)
- LFM2-VL-3B: A New Efficient Vision-Language for the Edge (predecessor) — [Liquid AI Blog](https://www.liquid.ai/blog/lfm2-vl-3b-a-new-efficient-vision-language-for-the-edge)
- Agentic Robotics at the Edge: Liquid AI, AMD e Robotec.ai com LFM2-VL-3B — [Liquid AI Blog](https://www.liquid.ai/blog/agentic-robotics-at-the-edge-liquid-ai-amd-and-robotec-ai-demonstrate-lfm2-vl-3b-in-embedded-autonomy)
- LiquidAI/LFM2.5-VL-3B (pesos) — [Hugging Face](https://huggingface.co/LiquidAI/LFM2.5-VL-3B)
- LiquidAI/LFM2.5-VL-3B-GGUF (quantizações llama.cpp) — [Hugging Face](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-GGUF)
- LiquidAI/LFM2.5-VL-3B-ONNX — [Hugging Face](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-ONNX)
- LFM License — LFM Open License v1.0 (licenciamento, limiar de receita) — [Liquid AI](https://www.liquid.ai/lfm-license)
- Model License — [Liquid Docs](https://docs.liquid.ai/lfm/help/model-license)
- LFM2 Technical Report (arquitetura do backbone LFM2.5-2.6B) — [arXiv 2511.23404](https://arxiv.org/abs/2511.23404)
- LFM2-VL — [Hugging Face Transformers docs](https://huggingface.co/docs/transformers/model_doc/lfm2_vl) (referência de arquitetura interna)
- SigLIP2 — [Hugging Face Transformers docs](https://huggingface.co/docs/transformers/model_doc/siglip2) (encoder de visão usado internamente pelo LFM2.5-VL-3B)

## Runtime de serving: vLLM (substitui TensorRT-LLM como caminho primário)
- vLLM — [Liquid Docs (guia de deploy)](https://docs.liquid.ai/deployment/gpu-inference/vllm)
- llama.cpp (fallback) — [Liquid Docs](https://docs.liquid.ai/deployment/on-device/llama-cpp)
- lfm2_tool_parser (parser de tool calling dedicado à família LFM2/LFM2.5) — [vLLM Docs](https://docs.vllm.ai/en/latest/api/vllm/tool_parsers/lfm2_tool_parser/)
- Tool Calling — [vLLM Docs](https://docs.vllm.ai/en/latest/features/tool_calling/)
- Unlock Faster, Smarter Edge Models with 7x Gen AI Performance on Jetson AGX Thor (vLLM + NVFP4 no Thor) — [NVIDIA Technical Blog](https://developer.nvidia.com/blog/unlock-faster-smarter-edge-models-with-7x-gen-ai-performance-on-nvidia-jetson-agx-thor/)
- thorllm — vLLM manager para Jetson Thor (wheel pré-compilada, systemd) — [GitHub](https://github.com/ms1design/thorllm)
- FP4 Quantization with NVFP4 — [LLM Compressor / vLLM Docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)

## STT / VAD
- faster-whisper — [GitHub SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Silero VAD — [GitHub snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- WhisperLive (referência de streaming) — [GitHub collabora/WhisperLive](https://github.com/collabora/WhisperLive)
- whisper_streaming (referência de streaming) — [GitHub](https://github.com/wonyx/whisper_streaming)

## TTS (voice cloning)
- XTTS — [coqui-tts Read the Docs](https://coqui-tts.readthedocs.io/en/latest/models/xtts.html)
- Synthesizing speech (inference/streaming) — [coqui-tts docs](https://coqui-tts.readthedocs.io/en/latest/inference.html)
- coqui/XTTS-v2 (pesos + licença CPML) — [Hugging Face](https://huggingface.co/coqui/XTTS-v2)

## Orquestração / barge-in
- Pipecat — [Docs](https://docs.pipecat.ai/)
- Pipecat Evals — [Docs](https://docs.pipecat.ai/pipecat/fundamentals/evaluations/overview)
- Voice AI Barge-In and Turn-Taking (2026) — [FutureAGI](https://futureagi.com/blog/voice-ai-barge-in-turn-taking-2026/)
- Adaptive interruption handling — [LiveKit Docs](https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/)
- Barge-in on-device — [EdgeAI Blog](https://www.runedge.ai/blog/barge-in-interruption-handling-on-device-voice)

## Hardware / Jetson Thor
- Introducing NVIDIA Jetson Thor — [NVIDIA Developer Blog](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)
- Jetson AGX Thor vs AGX Orin — [Forecr](https://www.forecr.io/blogs/all/nvidia-jetson-orin-family-vs-thor-what-you-need-to-know)
- Jetson Thor — [NVIDIA](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)
- JetPack — [NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack)
- Jetson AI Lab — Tutorials — [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/)

## Histórico (runtime/arquitetura descartados na revisão de 2026-09-03)
Mantidos por rastreabilidade — não são mais dependências ativas do projeto:
- TensorRT-LLM — [Docs](https://nvidia.github.io/TensorRT-LLM/) (era o runtime primário assumido; ver [`01` §7](01-architecture.md))
- KV Cache Reuse Optimizations (TensorRT-LLM) — [NVIDIA Technical Blog](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)
- LFM2.5-2.6B (backbone text-only puro, era o "cérebro" antes de adotarmos o VLM pronto) — [Liquid AI Blog](https://www.liquid.ai/blog/lfm2-5-2-6b)

> **Nota:** versões de bibliotecas e blogs evoluem. Ao implementar, confirmar no model card / na
> doc da versão exata (especialmente a **tag do container vLLM para Jetson Thor**, o **template
> de chat/tool-calling** do LFM2.5-VL-3B, e a matriz de compatibilidade JetPack↔CUDA↔vLLM).
