# 11 — Testes e Avaliação

Princípio: **latência e barge-in são requisitos com número**, então têm teste automatizado que
falha o CI quando regride. Qualidade (STT/visão/voz) tem avaliação medida, não "no olho".

## 1. Camadas de teste

### 1.1 Testes de contrato (unit)
Cada serviço tem teste do seu contrato (`contracts/`): entradas/saídas, tipos, comportamento em
cancelamento. Rodam com **mocks** dos modelos (rápidos, sem GPU) no CI de PR.

### 1.2 Testes de latência (device)
Rodam na Jetson alvo, em modo de energia fixo (`nvpmodel`/`jetson_clocks`), reportando
**p50/p95**. Falham se estourarem os alvos de [`02-requirements.md`](02-requirements.md):

| Teste | Mede | Alvo |
|-------|------|------|
| `bench_barge_in` | fala detectada → áudio para | NFR-L1 ≤ 120/200 ms |
| `bench_endpoint` | fim de fala → decisão | NFR-L2 |
| `bench_stt_final` | endpoint → transcrição final | NFR-L3 |
| `bench_llm_ttft` | mensagens prontas (com imagens) → 1º token do vLLM | NFR-L4 |
| `bench_tts_ttfb` | frase → 1º chunk | NFR-L5 |
| `bench_roundtrip` | fim da fala → 1º áudio (e2e) | NFR-L6 |
| `bench_vision_prefill` | fração do TTFT atribuível ao processamento de imagem (não é mais um estágio isolado — ver [`06`](06-llm-brain-generation.md)) | NFR-L7 |
| `bench_llm_throughput` | tok/s sustentado do vLLM | NFR-L8 |
| `bench_cold_start` | tempo de warm-up do vLLM até 1º turno dummy bem-sucedido | NFR-R6 (informativo, não bloqueia CI de turno) |

Cada bench roda N iterações, descarta warm-up, e grava a distribuição (não só a média).

### 1.3 Teste de soak (estabilidade)
`soak_2h`: conversa sintética contínua por ≥ 2 h monitorando **páginas de KV cache**, memória
de GPU e latência ao longo do tempo. Falha se houver crescimento monotônico (vazamento —
NFR-R3) ou degradação de latência. Ver [`09` §6](09-kv-cache-generation-loop.md).

## 2. Avaliação de qualidade

### 2.1 STT
- **WER** em um conjunto pt-BR representativo (sotaques, ruído de fundo do ambiente alvo).
- Comparar `large-v3` vs `large-v3-turbo` vs INT8/int8_float16.

### 2.2 Visão (o cérebro multimodal)
- **VQA/captioning** em conjunto de validação (inclui pt-BR).
- **Regressão de quantização (crítico):** bateria de VQA/grounding/tool-calling comparando
  **BF16 (baseline) vs. a precisão escolhida como default** (FP8/NVFP4/GGUF) — o portão de M2 é
  *não regredir* de forma perceptível. Ver [`05` §8](05-vision-encoder-coupling.md).
- **Sanity-check contra benchmarks publicados:** confirmar que nosso deploy (prompt template,
  parser `lfm2`, quantização) reproduz, dentro de margem razoável, os números que a Liquid
  publicou (ScreenSpot-v2 80.7, RefCOCO P@1 87.9, ToolSandbox 59.5, BFCL v4 32.5 — ver
  [`05` §4](05-vision-encoder-coupling.md)). Um desvio grande indica bug de integração, não do
  modelo.
- Frescor visual: teste que muda a cena e verifica que a resposta reflete o frame corrente
  (valida a política de janela de vídeo de [`09`](09-kv-cache-generation-loop.md), em particular
  que frames de turnos antigos nunca vazam via o histórico consolidado).

### 2.3 TTS
- **MOS** (avaliação subjetiva) da voz clonada em pt-BR; similaridade de locutor vs. a amostra
  de referência.
- Verificar pronúncia de números, siglas e estrangeirismos comuns no domínio.

### 2.4 Barge-in / turn-taking (comportamental)
- Suite de cenários: interromper no meio de frase, interromper durante tool call, ruído de
  fundo (não deve interromper), backchannel (M5, não deve interromper), TTS não dispara
  barge-in (AEC).
- Cada cenário verifica **estado final correto** + **latência** + **ausência de efeito colateral
  duplicado** (tools idempotentes).

## 3. Metodologia de latência

- **Timestamps por fronteira de estágio** (logging estruturado, [`CLAUDE.md`](../CLAUDE.md)) →
  reconstruir a cascata offline e atribuir cada ms a um estágio.
- Sempre reportar **percentis** (p50/p95), nunca só média — a cauda é o que o usuário sente.
- Registrar, junto de cada número: versão de JetPack/TensorRT, pesos, modo de energia,
  precisão (FP8/FP4), fps/resolução de vídeo (NFR-R5).

## 4. Gate de CI (resumo)
- PR normal: testes de contrato + lint (sem GPU).
- PR que toca o laço quente (`services/llm`, `services/tts`, `orchestrator`, `services/vision`):
  **exige** rodar os benchmarks de latência no runner de device e não regredir.
- Merge para `main`: soak curto (ex.: 20 min) automático; soak de 2 h agendado (nightly).

## 5. Ferramentas
- Framework de avaliação de pipeline de voz (ex.: os *evals* do Pipecat —
  [Pipecat Evals](https://docs.pipecat.ai/pipecat/fundamentals/evaluations/overview)) para
  cenários conversacionais end-to-end.
- Harness próprio para os benchmarks de latência (precisa de timestamps de device).
