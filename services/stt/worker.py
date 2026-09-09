"""STT streaming com faster-whisper (docs/04).

O event loop nunca bloqueia em inferência (CLAUDE.md "Concorrência"): a
chamada síncrona do CTranslate2 roda em executor.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from config import SttConfig
from contracts.messages import Transcript


class FasterWhisperStt:
    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper não instalado — rode `pip install -e .[stt]` no device "
                "antes de transcrever de verdade (docs/04)."
            ) from exc
        self._model = WhisperModel(
            self._cfg.model,
            device=self._cfg.device,
            compute_type=self._cfg.compute_type,
        )

    async def transcribe(self, pcm_f32: np.ndarray, *, partial: bool) -> Transcript:
        """Transcreve um segmento de áudio do turno atual (docs/04 §3).

        `partial=True`: parciais durante a fala, feedback de latência apenas.
        `partial=False`: transcrição final no endpoint — é essa que aciona o cérebro.
        """
        self._ensure_loaded()
        t_start = time.time()
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._transcribe_sync, pcm_f32, partial)
        return Transcript(text=text, is_final=not partial, t_start=t_start, t_end=time.time())

    def _transcribe_sync(self, pcm_f32: np.ndarray, partial: bool) -> str:
        segments, _info = self._model.transcribe(
            pcm_f32,
            language=self._cfg.language,
            beam_size=self._cfg.partial_beam_size if partial else self._cfg.final_beam_size,
            vad_filter=False,  # o VAD já é externo (Silero) — doc04 §5
            word_timestamps=not partial,
            condition_on_previous_text=False,  # evita drift em turnos curtos
        )
        return "".join(s.text for s in segments)
