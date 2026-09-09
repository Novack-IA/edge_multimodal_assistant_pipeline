# 07 — TTS: XTTS-v2 (Clonagem de Voz + Streaming em pt-BR)

## 1. Escolha

**Coqui XTTS-v2** — TTS multilíngue com **clonagem de voz** a partir de uma amostra de
referência e **streaming de baixa latência**. Atende aos três requisitos: pt-BR, voice
cloning (foco pedido) e streaming interrompível.

- Docs: [XTTS — coqui-tts (Read the Docs)](https://coqui-tts.readthedocs.io/en/latest/models/xtts.html),
  [Synthesizing speech — coqui-tts](https://coqui-tts.readthedocs.io/en/latest/inference.html)
- Pesos: [coqui/XTTS-v2 — Hugging Face](https://huggingface.co/coqui/XTTS-v2)

**Idiomas:** XTTS-v2 suporta 17 idiomas **incluindo português (`pt`)**.

**Latência:** a documentação reporta *"streaming inference com < 200 ms de latência"* — base
para NFR-L5.

## 2. ⚠️ Licenciamento (bloqueio para uso comercial)

XTTS-v2 é distribuído sob a **Coqui Public Model License (CPML)**, que **restringe uso
comercial**. Antes de qualquer deploy comercial:

- **NFR-P2:** revisar a CPML e confirmar se o uso pretendido é permitido.
- Se for comercial e a CPML não permitir, avaliar **plano B** com licença permissiva e
  clonagem de voz — candidatos a investigar: modelos de voice cloning open com licença
  Apache/MIT, ou solução de TTS licenciável (incl. **NVIDIA Riva TTS**). Manter a interface do
  worker de TTS **agnóstica de motor** para permitir a troca sem tocar no orquestrador.

> Decisão travada pelo usuário: **foco em voice cloning**. XTTS-v2 é o default técnico; a
> pendência de licença é um item de risco a resolver em M1, não um bloqueio de arquitetura.

## 3. Clonagem de voz (fluxo)

Extrair os *latents* do locutor **uma vez** a partir da amostra de referência e reutilizar em
todas as sínteses:

```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig(); config.load_json("XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="XTTS-v2/", use_deepspeed=True)  # deepspeed acelera
model.cuda()

# Extrai latents da voz de referência (fazer no boot, cachear)
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=["referencia_voz.wav"],
)
```

- `speaker_wav` aceita **um ou vários** áudios de referência sem afetar o runtime.
- Cachear `gpt_cond_latent` e `speaker_embedding` (não recomputar por frase).

## 4. Streaming (interrompível)

```python
chunks = model.inference_stream(
    text_da_frase,          # sintetizar por frase/cláusula, não a resposta toda
    "pt",                   # português
    gpt_cond_latent,
    speaker_embedding,
    # temperature=0.65, repetition_penalty=2.0, top_k=50, top_p=0.8, speed=1.0
)
for chunk in chunks:
    if cancel_token.is_set():   # BARGE-IN: parar imediatamente
        break
    play(chunk)                 # enfileira no playback
```

Parâmetros (defaults da doc): `temperature=0.65`, `length_penalty=1.0`,
`repetition_penalty=2.0`, `top_k=50`, `top_p=0.8`, `speed=1.0`. Expor no `config.yaml`.

## 5. Integração com o laço

- **Segmentar por frase/cláusula.** O LLM emite tokens em streaming; o orquestrador acumula
  até um limite de frase (pontuação ou N tokens) e dispara `inference_stream` daquele trecho.
  Isso minimiza TTFB (NFR-L5) e torna a interrupção granular.
- **Fila de playback interrompível.** O áudio sintetizado vai para uma fila de reprodução que
  pode ser **esvaziada instantaneamente** no barge-in (ver [`08`](08-orchestration-barge-in.md)).
- **`use_deepspeed=True`** para reduzir latência de inferência quando disponível na Jetson;
  medir ganho real no device.
- **Aquecimento (warm-up):** rodar uma síntese dummy no boot para pagar custos de
  init/CUDA graphs antes do primeiro turno real.

## 6. Contrato do worker (agnóstico de motor)

```python
# services/tts/worker.py (contrato)
async def synthesize_stream(
    text: str, cancel_token,
) -> AsyncIterator[AudioOut]:
    """Sintetiza `text` em pt-BR com a voz clonada, em chunks PCM.
    Deve checar cancel_token entre chunks e abortar em barge-in."""
```

Manter esta interface estável permite trocar XTTS-v2 por outro motor (plano B de licença) sem
mexer no orquestrador.

## 7. Riscos

| Risco | Mitigação |
|-------|-----------|
| **Licença CPML** bloqueia uso comercial | resolver em M1; interface agnóstica p/ trocar motor |
| Latência de streaming acima de NFR-L5 | DeepSpeed; segmentar por frase; warm-up; medir no device |
| Qualidade pt-BR da voz clonada | avaliar amostras de referência; múltiplos áudios de ref |
| TTS "vaza" no microfone e dispara barge-in falso | AEC + gating de VAD ciente do estado (doc 08) |
