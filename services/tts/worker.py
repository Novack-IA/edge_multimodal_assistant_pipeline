"""TTS streaming com clonagem de voz — XTTS-v2 (docs/07).

`TtsEngine` é o contrato agnóstico de motor (docs/07 §6): trocar XTTS-v2 por
outro engine (plano B de licença — CPML, ver docs/02 NFR-P2) não deve exigir
mudanças no orquestrador, só uma nova implementação deste Protocol.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

from config import TtsConfig
from contracts.messages import AudioOut


class TtsEngine(Protocol):
    async def synthesize_stream(self, text: str, cancel_token: asyncio.Event) -> AsyncIterator[AudioOut]: ...

    async def warmup(self) -> None: ...


class XttsV2Engine:
    """Contrato: docs/07 §6. `checkpoint_dir` aponta para os pesos XTTS-v2
    baixados no device (`coqui/XTTS-v2` no Hugging Face) — não versionado
    aqui, é dado de deployment."""

    def __init__(self, cfg: TtsConfig, checkpoint_dir: str | Path = "XTTS-v2") -> None:
        self._cfg = cfg
        self._checkpoint_dir = Path(checkpoint_dir)
        self._model = None
        self._gpt_cond_latent = None
        self._speaker_embedding = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "TTS (coqui) não instalado — rode `pip install -e .[tts]` no device "
                "antes de sintetizar de verdade (docs/07). Revisar CPML antes de uso "
                "comercial (docs/02 NFR-P2)."
            ) from exc

        config = XttsConfig()
        config.load_json(str(self._checkpoint_dir / "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=str(self._checkpoint_dir), use_deepspeed=self._cfg.use_deepspeed)
        model.cuda()
        self._model = model

        # Extrai latents da voz de referência uma vez e cacheia (docs/07 §3).
        self._gpt_cond_latent, self._speaker_embedding = model.get_conditioning_latents(
            audio_path=[self._cfg.speaker_wav],
        )

    async def warmup(self) -> None:
        """Síntese dummy no boot — paga custo de init/CUDA graphs (docs/07 §5)."""
        cancel_token = asyncio.Event()
        async for _ in self.synthesize_stream("oi", cancel_token):
            pass

    async def synthesize_stream(self, text: str, cancel_token: asyncio.Event) -> AsyncIterator[AudioOut]:
        """Sintetiza `text` em pt-BR com a voz clonada, em chunks PCM.
        Checa `cancel_token` entre chunks e aborta em barge-in (docs/07 §4)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        def _produce() -> None:
            self._ensure_loaded()
            chunks = self._model.inference_stream(
                text,
                self._cfg.language,
                self._gpt_cond_latent,
                self._speaker_embedding,
                temperature=self._cfg.temperature,
                repetition_penalty=self._cfg.repetition_penalty,
                top_k=self._cfg.top_k,
                top_p=self._cfg.top_p,
                speed=self._cfg.speed,
            )
            for chunk in chunks:
                if cancel_token.is_set():  # BARGE-IN: parar imediatamente
                    break
                pcm = chunk.detach().cpu().numpy().tobytes() if hasattr(chunk, "detach") else bytes(chunk)
                loop.call_soon_threadsafe(queue.put_nowait, pcm)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        # Inferência bloqueante roda em thread própria — o event loop nunca
        # bloqueia em inferência (CLAUDE.md "Concorrência").
        threading.Thread(target=_produce, daemon=True).start()

        while True:
            pcm = await queue.get()
            if pcm is None or cancel_token.is_set():
                yield AudioOut(pcm=b"", is_last=True)
                return
            yield AudioOut(pcm=pcm, is_last=False)
