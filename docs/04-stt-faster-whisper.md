# 04 — STT: faster-whisper + Silero VAD

> **Revisão 2026-09-09 (validado no device real):** o engine **default** deixou de ser
> faster-whisper e passou a ser **Parakeet-TDT-0.6B-v3 (NVIDIA, via NeMo)** — ver §9. Motivo:
> o wheel pip do CTranslate2 para aarch64 é **CPU-only** (não existe build CUDA publicado para
> essa combinação de plataforma), então "faster-whisper na GPU da Jetson" descrito abaixo nunca
> foi verdade neste device específico. Parakeet-TDT é PyTorch nativo — usa o mesmo torch+CUDA já
> validado no resto do sistema (vLLM, XTTS-v2) — e mede RTFx 57–90× no Thor. As seções 1–8 abaixo
> ficam como contexto histórico da decisão original e continuam válidas para `engine:
> faster_whisper` (fallback documentado, roda em CPU).

## 1. Escolha

**faster-whisper** (implementação do Whisper sobre **CTranslate2**), rodando na GPU da Jetson,
com quantização **INT8**. É até ~4× mais rápido que o Whisper de referência com uso de memória
menor, e suporta transcrição com timestamps de palavra.

- Repositório oficial: [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)

**Silero VAD** faz VAD/endpointing e é o gatilho de barge-in.

- Repositório oficial: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)

## 2. Modelo e precisão

- Modelo alvo: `large-v3` (qualidade) ou `large-v3-turbo` (latência). Avaliar ambos em pt-BR
  no device durante M1.
- `compute_type="int8"` (ou `int8_float16` se a qualidade cair). Isso é o que dá o ganho de
  latência/memória em Jetson.
- `device="cuda"`.

## 3. Contrato de streaming

O Whisper é, por natureza, um modelo **de janela** (processa ~30 s de contexto), não um
reconhecedor online token-a-token. Estratégia de streaming pragmática:

1. **VAD segmenta.** O Silero VAD marca `speech_start`/`speech_end`.
2. **Transcrições parciais.** Durante a fala, transcreva janelas deslizantes crescentes do
   áudio acumulado do turno atual e emita `Transcript(is_final=False)`. Essas parciais são
   para *feedback/latência percebida*, não para lógica final.
3. **Transcrição final.** No `speech_end` (endpoint), transcreva o segmento completo do turno
   e emita `Transcript(is_final=True)`. **É essa que aciona o cérebro.**

> Referências de arquitetura de streaming sobre faster-whisper (para inspiração de
> implementação, não como dependência): [WhisperLive — Collabora](https://github.com/collabora/WhisperLive),
> [whisper_streaming](https://github.com/wonyx/whisper_streaming). Avalie se vale usar um
> desses como base ou implementar o loop mínimo internamente.

## 4. VAD / endpointing (Silero)

- Roda em blocos de 30 ms a 16 kHz.
- Parâmetros a expor no `config.yaml`:
  - `vad_threshold` (probabilidade de fala; default ~0.5)
  - `min_silence_ms` para endpoint (default 500–800 ms) → controla NFR-L2
  - `min_speech_ms` (evita disparar em cliques/ruído curto)
  - `speech_pad_ms` (padding antes/depois do segmento)
- **Duplo uso:** o mesmo VAD que faz endpointing em `LISTENING` roda **também durante
  `SPEAKING`** para detectar barge-in (ver [`08`](08-orchestration-barge-in.md)). Idealmente
  em uma instância dedicada de baixa latência.

## 5. Esboço de implementação

```python
# services/stt/worker.py
from faster_whisper import WhisperModel

model = WhisperModel(
    "large-v3",            # ou "large-v3-turbo"
    device="cuda",
    compute_type="int8",   # chave para Jetson
)

def transcribe_segment(pcm_f32, language="pt", partial=False):
    segments, info = model.transcribe(
        pcm_f32,
        language=language,
        beam_size=1 if partial else 5,   # parcial = mais rápido
        vad_filter=False,                 # o VAD já é externo (Silero)
        word_timestamps=not partial,
        condition_on_previous_text=False, # evita drift em turnos curtos
    )
    return "".join(s.text for s in segments)
```

## 6. Considerações de latência

- **Parciais com `beam_size=1`** e janelas curtas para não competir com a GPU do LLM.
- A transcrição **final** compete com o LLM pela GPU; sequencie: `speech_end` → final →
  então libere o LLM. O overlap desejável é do lado do TTS, não do STT.
- Meça NFR-L3 (endpoint → final) isoladamente. Se estourar, teste `large-v3-turbo` ou reduza
  `beam_size` da final para 3.
- **AEC (cancelamento de eco)** no áreas antes do VAD/STT evita que o TTS contamine o STT.

## 7. Riscos e mitigação

| Risco | Mitigação |
|-------|-----------|
| Latência da final alta em `large-v3` | usar `large-v3-turbo`; INT8; beam menor |
| VAD dispara com o próprio TTS | AEC; instância de VAD ciente do estado `SPEAKING` |
| Drift de contexto entre turnos | `condition_on_previous_text=False` em turnos curtos |
| Sotaques/ruído pt-BR | avaliar WER em dados representativos no M1; considerar Riva pt-BR como plano B |

## 8. Plano B (histórico)

Se faster-whisper não atingisse NFR-L3 em pt-BR no device, avaliar **NVIDIA Riva ASR**
(Conformer/Parakeet, TensorRT-native, streaming real). Isso aconteceu — mas por uma razão
diferente da esperada (não foi WER/latência de modelo, foi ausência de wheel CUDA para
CTranslate2 em aarch64) — e a escolha final não foi Riva (serviço/servidor à parte, mais
operacional) e sim consumir o Parakeet-TDT diretamente via NeMo. Ver §9.

## 9. Engine default: Parakeet-TDT-0.6B-v3 via NeMo (revisão 2026-09-09)

### 9.1 Por que trocar

`faster-whisper` (docs §1-8 acima) foi a escolha original porque CTranslate2 é ~4× mais rápido
que o Whisper de referência **quando há um build CUDA**. No Jetson AGX Thor (aarch64 + JetPack
7.x + CUDA 13.2), o `pip install faster-whisper` traz um CTranslate2 pré-compilado que é
**CPU-only** — compilar CTranslate2 com CUDA para ARM a partir do código-fonte é conhecido por
ser difícil e não há wheel pip publicado para essa combinação. Rodar `large-v3` em CPU não bate
NFR-L3 de forma confiável.

### 9.2 O que foi avaliado

| Modelo | Runtime | Multilíngue (pt) | Observação |
|---|---|---|---|
| **Parakeet-TDT-0.6B-v3** (NVIDIA) | NeMo / PyTorch+CUDA | 25 idiomas, WER pt publicado 6.16 | **Escolhido** — ver §9.3 |
| Canary-1B-v2 (NVIDIA) | NeMo / PyTorch+CUDA | 25 idiomas europeus | Maior (1B) e "low throughput" vs. Parakeet (fonte: pesquisa web) — não testado no device |
| Whisper large-v3 via `transformers` (PyTorch puro) | PyTorch+CUDA | Multilíngue, pt-BR ok | Alternativa viável não testada — evita CTranslate2 mantendo o formato Whisper |
| NVIDIA Riva ASR | TensorRT / serviço à parte | Conformer/Parakeet | Mais operacional (servidor dedicado); não adotado por ora — Parakeet direto via NeMo é mais simples de embutir no mesmo processo |

### 9.3 Validação real no device (Jetson AGX Thor)

Instalado via `pip install nemo_toolkit[asr]` (resolve limpo em aarch64/py3.12 — sem gap de
wheel, ao contrário do CTranslate2). Teste round-trip: XTTS-v2 sintetizou uma frase pt-BR
conhecida, Parakeet-TDT transcreveu de volta:

- **Texto original:** "O assistente multimodal precisa entender comandos de voz em português
  com clareza."
- **Transcrito:** "O assistente multimodal precisa entender comandos de voz em português, com
  clareza." (diferença: só uma vírgula — sem erro de palavra)
- **Latência (chamadas aquecidas, modelo já em `cuda:0`):** ~90-96ms para 5.3s de áudio, ~65-77ms
  para 2.0s de áudio. **RTFx ≈ 57-90×** — bem dentro de NFR-L3 (≤300/600ms).
- Modelo carrega em `cuda:0` automaticamente via `ASRModel.from_pretrained(...).to("cuda")`,
  mesmo padrão de dispositivo já validado com vLLM e XTTS-v2 no resto do sistema.

### 9.4 Contrato

Mesma interface de `FasterWhisperStt` (`transcribe(pcm_f32, partial) -> Transcript`, roda em
executor) — ver `services/stt/parakeet.py`. Trocável via `config.yaml: stt.engine` (`parakeet`
default | `faster_whisper` fallback), mesma fábrica `services/stt/create_stt_engine` usada pelo
orquestrador — nenhum código fora de `services/stt/` precisa saber qual engine está ativo.

### 9.5 Riscos

| Risco | Mitigação |
|---|---|
| `nemo_toolkit[asr]` é uma árvore de dependências grande (lightning, hydra, wandb, etc.) | instalação validada no device sem conflito; monitorar em upgrades futuros |
| WER publicado (6.16) é sobre português europeu, não pt-BR | teste round-trip com voz sintética pt-BR saiu limpo; avaliar com falantes reais no M1/M2 |
| Parakeet-TDT não é nativamente streaming (transcreve o segmento inteiro) | mesma estratégia de §3 (parciais em janela crescente + final no endpoint) já cobre isso |
