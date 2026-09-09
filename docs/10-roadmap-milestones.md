# 10 — Roadmap e Milestones

Plano faseado. Cada milestone tem **entregável verificável** e um **portão** (não avança sem
passar). Datas ficam a critério do time; a **ordem** é normativa.

> **Revisão 2026-09-03:** M2 foi redefinido. No plano original, era o item de maior risco do
> cronograma (treinar um conector preservando tool calling). Com o LFM2.5-VL-3B pronto, M2 vira
> **validação de um modelo já treinado**, não treino — risco e duração muito menores.

## M0 — Bootstrap (fundação)
**Entregáveis:** repositório com esta `docs/`, `config.yaml` tipado, `contracts/` com as
dataclasses de mensagem, esqueleto de `services/` e `orchestrator/`, CI básico.
**Portão:** o esqueleto sobe, os contratos compilam, os testes de contrato (mock) passam.

## M1 — Componentes isolados no device
Colocar cada modelo rodando **sozinho** na Jetson Thor, medindo latência/memória reais.
**Entregáveis:**
- faster-whisper INT8 transcrevendo pt-BR (WER + NFR-L3) — [`04`](04-stt-faster-whisper.md).
- Silero VAD com endpointing calibrado (NFR-L2).
- **vLLM servindo o LFM2.5-VL-3B** (BF16) no device — subida, warm-up medido (NFR-R6),
  TTFT/tok-s em prompt multimodal (NFR-L4/L7/L8) — [`06`](06-llm-brain-generation.md),
  [`13`](13-vllm-deployment.md).
- XTTS-v2 clonando voz e fazendo streaming pt-BR (NFR-L5) — [`07`](07-tts-voice-cloning.md).
- **Resolver licença do XTTS-v2 (CPML)** e **registrar a licença do LFM2.5-VL-3B (LFM1.0,
  limiar de receita)** para o uso pretendido (NFR-P2, NFR-P4).
**Portão:** cada componente atinge seu orçamento de latência isolado; footprints de memória
somados cabem na plataforma (NFR-R1).

## M2 — Multimodalidade nativa validada (sem treino)
Antes era a peça de pesquisa/treino do projeto; agora é **validação e calibração** de um modelo
pronto — [`05`](05-vision-encoder-coupling.md), [`06`](06-llm-brain-generation.md).
**Entregáveis:**
- Confirmar que o parser `lfm2` do vLLM extrai `tool_calls` corretamente, inclusive quando
  disparado pelo conteúdo da imagem (FR-11b).
- Benchmark de quantização: BF16 (baseline) vs. FP8 vs. NVFP4 vs. GGUF, em qualidade
  (VQA/grounding/tool-calling, comparando contra os números publicados pela Liquid — doc 05 §4)
  e latência. Escolher a precisão default.
- Calibrar `video_token_budget`/`window_frames` com medições reais de tokens-por-imagem.
- Validar o teto de contexto de 32K com uma janela de vídeo real + histórico típico (NFR-R7).
**Portão:** o modelo responde corretamente sobre uma imagem estática **e** mantém tool-calling
dentro do esperado pelos benchmarks publicados (ToolSandbox/BFCL), na precisão escolhida como
default.

## M3 — Laço conversacional de voz (sem vídeo, sem barge-in)
Fechar o loop de voz ponta a ponta — [`08`](08-orchestration-barge-in.md).
**Entregáveis:**
- Pipecat orquestrando VAD → STT → cliente vLLM → TTS em turnos.
- **AEC** funcionando (pré-requisito do barge-in) — [`08` §4](08-orchestration-barge-in.md).
- Máquina de estados IDLE/LISTENING/THINKING/SPEAKING.
- Contrato de sessão (histórico texto-only, commit no fim do turno) — [`09`](09-kv-cache-generation-loop.md).
**Portão:** conversa por voz fluida em pt-BR, round-trip p50 dentro de NFR-L6 (ainda sem
vídeo/barge-in).

## M4 — MVP: vídeo em tempo real + barge-in + 1 tool
O MVP dos critérios de aceitação — [`02` §6](02-requirements.md).
**Entregáveis:**
- Janela de vídeo + anexação de frames por turno — [`09`](09-kv-cache-generation-loop.md).
- **Barge-in** dentro de NFR-L1 (≤ 200 ms), abortando a requisição HTTP do vLLM sem commit
  parcial.
- **Uma tool call real** ponta a ponta via parser `lfm2`.
- **Soak test de 2 h** sem vazamento de contexto/memória (NFR-R3).
**Portão:** todos os critérios de aceitação do MVP passam; benchmarks de latência verdes no CI.

## M5 — Robustez e naturalidade
**Entregáveis:**
- Distinção **barge-in vs. backchannel** (FR-18).
- **Modo de degradação** sob saturação de GPU (FR-21) — inclui trocar quantização/precisão do
  vLLM em runtime, e opcionalmente cair para LFM2.5-VL-1.6B ([`05`](05-vision-encoder-coupling.md) §6).
- Supervisão/restart de workers, **incluindo o processo/container do vLLM** (NFR-R4).
- Amostragem de vídeo por keyframe (economia de tokens).
- Endurecimento de tools (timeouts, idempotência, cancelamento).
- **Grounding** exposto como capacidade de produto (FR-8b).
**Portão:** sessões longas estáveis; interrupções e ruído tratados sem falsos positivos.

## M6 — Upgrades opcionais
**Candidatos (priorizar por necessidade real):**
- **Fine-tuning de domínio via LoRA** (movido de M2 no plano original para cá — só entra se a
  avaliação de M2 mostrar necessidade real) — [`05` §7](05-vision-encoder-coupling.md).
- Leitura de tela/documento como feature de produto (FR-8c).
- Temporalidade de vídeo mais longa que o suportado nativamente pelo modelo.
- Frames intra-turno — [`09` §4](09-kv-cache-generation-loop.md).
- Backchannels ativos (o assistente dá sinais de escuta).
- Multi-idioma na saída.

## Dependências (resumo)

```
M0 ─▶ M1 ─▶ M2 ─┐
            └────▶ M3 ─▶ M4 ─▶ M5 ─▶ M6
(M2 e M3 podem correr em paralelo após M1; M4 exige ambos)
```

## Riscos de cronograma (topo, revisado)
1. **Quantização degradar tool-calling/grounding** (M2) — risco menor que "treinar conector
   preservando agentic" do plano original, mas ainda real; benchmark antes de fixar default.
2. **Maturidade operacional do vLLM em Jetson Thor** — plataforma e integração recentes (poucos
   meses de histórico público); ter fallback documentado (`llama.cpp`, [`13`](13-vllm-deployment.md) §8).
3. **Licença do XTTS-v2** (M1) pode forçar troca de motor de TTS — a interface agnóstica
   ([`07` §6](07-tts-voice-cloning.md)) contém o estrago.
4. **Teto de contexto de 32K** menor que o assumido no plano original — pode exigir
   `history_max_turns` mais conservador do que o confortável (M2/M3).
