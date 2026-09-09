# Benchmarks de latência (docs/11)

Rodam no device alvo, em modo de energia fixo (`nvpmodel`/`jetson_clocks`),
reportando p50/p95 — nunca só média. Falham (assert) se estourarem
`config.yaml: latency_budgets_ms`.

| Script | Mede | NFR |
|---|---|---|
| `bench_llm_ttft.py` | mensagens prontas → 1º token do vLLM | NFR-L4 (requer vLLM de pé) |
| `bench_barge_in.py` | fração software: cancel_token → stream retorna | NFR-L1 (fração VAD/playback requer M3/M4) |

## Ainda não implementados (requerem componentes de M1/M3/M4 no device)

- `bench_endpoint` (NFR-L2), `bench_stt_final` (NFR-L3): precisam de
  `services/stt` com pesos baixados e áudio de referência pt-BR.
- `bench_tts_ttfb` (NFR-L5): precisa dos pesos XTTS-v2 + `assets/reference_voice.wav`.
- `bench_roundtrip` (NFR-L6): precisa do `Orchestrator` fim-a-fim com mic/câmera reais (M3/M4).
- `bench_vision_prefill` (NFR-L7): fração do TTFT atribuível a imagem — via
  métricas Prometheus do vLLM (docs/13 §7), comparando prompt com/sem frames.
- `bench_llm_throughput` (NFR-L8): tok/s sustentado, requer vLLM de pé sob carga.
- `bench_cold_start` (NFR-R6): tempo de `Orchestrator.boot()` na primeira subida do vLLM.

Ver `benchmarks/harness.py` (`LatencyBench`) — todo benchmark novo usa o
mesmo harness para consistência de relatório e de gate de CI.
