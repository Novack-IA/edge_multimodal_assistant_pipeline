# 04 — STT: faster-whisper + Silero VAD

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

## 8. Plano B

Se faster-whisper não atingir NFR-L3 em pt-BR no device, avaliar **NVIDIA Riva ASR**
(Conformer/Parakeet, TensorRT-native, streaming real). Fica registrado como alternativa, mas o
padrão do projeto é faster-whisper conforme pedido.
