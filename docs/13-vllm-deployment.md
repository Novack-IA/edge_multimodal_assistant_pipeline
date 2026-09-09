# 13 — Deploy e Operação do vLLM (Servindo o LFM2.5-VL-3B)

Documento novo (2026-09-03), companheiro de [`06`](06-llm-brain-generation.md). Cobre a
operação do servidor vLLM no Jetson Thor — o que o cliente do orquestrador assume que já está
rodando.

## 1. Por que vLLM

Resumo da decisão (detalhe em [`01` §7](01-architecture.md)): suporte oficial da NVIDIA para
Jetson Thor, parser de tool calling dedicado à família LFM2 (`lfm2`), PagedAttention +
continuous batching + Automatic Prefix Caching prontos, API multimodal OpenAI-compatible.

## 2. Caminhos de instalação no device

### Opção A — Container oficial NVIDIA
```
ghcr.io/nvidia-ai-iot/vllm:<tag>-tegra-aarch64-<cuda>-<ubuntu>
```
Confirmar a **tag exata** compatível com a versão de JetPack/CUDA do device no momento do
deploy — a NVIDIA publica atualizações frequentes; não fixar a tag encontrada em pesquisa sem
revalidar.

### Opção B — `thorllm` (wrapper comunitário)
[GitHub ms1design/thorllm](https://github.com/ms1design/thorllm) — instala vLLM de uma wheel
pré-compilada `cu130` (fallback: build from source), cria venv isolado, configura serviço
systemd, oferece CLI/TUI de gestão. Suporta templates de modelo em YAML (tensor parallelism,
quantização, attention backend — FlashInfer).

### Requisitos confirmados neste device (verificado 2026-09-03)
- JetPack 7.x (build `R39` via `nv_tegra_release`).
- CUDA 13.2 (`nvidia-smi`).
- Python 3.12.3.
- GPU `NVIDIA Thor`, arquitetura Blackwell **SM_110a**.
- 122 GB RAM livre de 128 GB.

## 3. Comando de subida (referência)

```bash
vllm serve LiquidAI/LFM2.5-VL-3B \
  --enable-auto-tool-choice \
  --tool-call-parser lfm2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.5 \
  --host 127.0.0.1 --port 8000
```

- `--gpu-memory-utilization` controla quanto da memória é reservado para o pool de KV cache do
  vLLM — calibrar junto com o orçamento de [`03` §4](03-hardware-platform.md).
- `--max-model-len 32768` reflete o teto real de contexto do modelo (ver [`09` §6](09-kv-cache-generation-loop.md)) —
  não configurar um valor maior "só para garantir", isso não muda a capacidade real do modelo.
- Para servir uma variante quantizada, apontar para o repositório do checkpoint FP8/NVFP4 (ver
  §5) ou usar `--quantization <método>` conforme suportado pela versão do vLLM instalada.

## 4. Warm-up / cold start (NFR-R6)

A **primeira** subida do vLLM nesta arquitetura faz compilação JIT de kernels Triton/FlashInfer
para SM_110a — **5 a 15 minutos**, "parece travado" mas não está. Os artefatos compilados ficam
em cache local (tipicamente `~/.cache/vllm` ou equivalente) e subidas seguintes são rápidas.

**Regra de operação:** o orquestrador **não** deve aceitar turnos de usuário antes de:
1. O processo/container do vLLM reportar saúde (`/health` ou equivalente da API).
2. Um **turno dummy** (prompt mínimo, sem imagem) ter sido respondido com sucesso.

Só então o sistema entra em estado "pronto". Isso é parte do contrato de boot, não um detalhe
de implementação a critério de quem escrever o código.

## 5. Quantização — matriz de opções

| Precisão | Runtime | Tamanho (checkpoint) | Quando usar |
|---|---|---|---|
| **BF16** | vLLM | 5.4 GB | **Default para começar** (M1/M2) — sem risco de qualidade, memória sobra na Thor |
| FP8 (community, LLM Compressor+AutoRound) | vLLM | ~2.7–3 GB (estimado) | Após benchmark confirmar não-regressão (M2); camadas de convolução do LIV ficam não-quantizadas de propósito (necessário para rodar no vLLM) |
| NVFP4 (community, Blackwell-nativo) | vLLM | ~1.5–2 GB (estimado) | **Não benchmarked publicamente** — só adotar após medir qualidade e latência no device; Blackwell (Thor) é o alvo nativo do formato, mas isso não substitui a medição |
| GGUF Q8_0 / Q6_K / Q5_K_M / Q4_K_M / Q4_0 | `llama.cpp` (fallback, não vLLM) | 2.87 / 2.22 / 1.94 / 1.67 / 1.59 GB | Se vLLM não atingir os orçamentos de latência, ou como runtime mais leve para a variante 1.6B (degradação, [`05` §6](05-vision-encoder-coupling.md)) |

Trocar a precisão deve ser possível só editando `config.yaml` (FR-12b) — o cliente do
orquestrador não deve saber ou se importar com qual precisão está por trás da API.

## 6. Integração com o orquestrador

Cliente HTTP assíncrono (`httpx`/`aiohttp`) contra `{server_url}/v1/chat/completions` com
`stream=true`. Contrato completo em [`06` §4](06-llm-brain-generation.md). Barge-in cancela
fechando a conexão do stream (`cancel_token` compartilhado, [`08`](08-orchestration-barge-in.md)).

## 7. Observabilidade

O vLLM expõe métricas (formato Prometheus) — TTFT, tok/s, ocupação do pool de KV cache,
tamanho das filas de requisição. Cruzar essas métricas com o logging estruturado do resto do
sistema (CLAUDE.md) para reconstruir a cascata de latência ponta a ponta, não só o que acontece
dentro do vLLM.

## 8. Fallback: `llama.cpp`

Se o vLLM não atingir os orçamentos de latência no device (ou em caso de instabilidade da
integração, dado que é uma plataforma nova — ver [`10` riscos](10-roadmap-milestones.md)),
`llama.cpp` com suporte multimodal (`llama-mtmd-cli`/servidor, flag `--mmproj`) é o caminho mais
maduro publicamente para rodar LFM2-VL fora do vLLM:

```
llama-mtmd-cli -hf LiquidAI/LFM2.5-VL-3B-GGUF:Q4_K_M
```

A interface do cliente do LLM no orquestrador deve ficar **agnóstica de runtime** (mesmo
princípio usado para o TTS, [`07` §6](07-tts-voice-cloning.md)) — trocar vLLM por `llama.cpp`
não deve exigir mudanças fora de `services/llm/`.

## 9. Riscos

| Risco | Mitigação |
|-------|-----------|
| Tag do container oficial desatualizada/incompatível | revalidar no momento do deploy; não fixar tag de pesquisa sem checar |
| Cold-start de 5–15 min interpretado como falha | NFR-R6 — gate explícito de "pronto", documentado e logado |
| NVFP4/FP8 comunitários sem benchmark público | medir no device antes de adotar como default |
| vLLM em Thor é integração recente | pinar versão exata (NFR-R5); fallback `llama.cpp` documentado (§8) |
| `--max-model-len` configurado maior que o real do modelo mascarando estouro de contexto | manter em 32768, alinhado ao teto real (doc 09 §6) |
