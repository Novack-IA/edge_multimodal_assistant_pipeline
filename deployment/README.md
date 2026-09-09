# Deployment

Guia operacional completo: [`docs/13-vllm-deployment.md`](../docs/13-vllm-deployment.md).

## vLLM (LFM2.5-VL-3B)

```bash
# Instalar (device real, fora do escopo deste repo — ver docs/13 §2):
#   - container oficial: ghcr.io/nvidia-ai-iot/vllm:<tag>-tegra-aarch64-...
#   - ou thorllm: https://github.com/ms1design/thorllm

python deployment/serve_vllm.py            # imprime o comando derivado de config.yaml
python deployment/serve_vllm.py --exec     # sobe de verdade

# systemd (produção):
sudo cp deployment/vllm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm
```

`deployment/serve_vllm.py` lê `llm.*` de `config.yaml` e nunca hardcoda
quantização/porta — trocar a precisão default é só editar o YAML (FR-12b).

## Fallback: llama.cpp

Se o vLLM não bater os orçamentos de latência (docs/13 §8):

```bash
llama-mtmd-cli -hf LiquidAI/LFM2.5-VL-3B-GGUF:Q4_K_M
```

Trocar `llm.runtime: llama_cpp` no config ainda não tem um client dedicado
implementado em `services/llm/` — hoje só `vllm` (OpenAI-compatible) é
suportado; ver risco registrado em docs/10.

## Warm-up / cold-start (NFR-R6)

`orchestrator.Orchestrator.boot()` só marca o sistema como pronto depois de:
1. `GET {server_url}/models` responder 200 (health check simples via API OpenAI-compatible).
2. Um turno dummy (`LlmClient.warmup`) ser respondido com sucesso.
3. Uma síntese dummy de TTS (`TtsEngine.warmup`) completar.

A primeira subida do vLLM nesta arquitetura pode levar 5–15 min (JIT
Triton/FlashInfer para SM_110a) — `cold_start.vllm_warmup_budget_s` no
config é o teto de espera antes de `boot()` desistir.
