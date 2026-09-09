"""STT via NVIDIA Parakeet-TDT (NeMo) — engine default (docs/04 §9).

Escolhido no lugar do faster-whisper/CTranslate2 porque o wheel pip do
CTranslate2 para aarch64 é **CPU-only** (não existe build CUDA publicado
para essa combinação de plataforma — só há builds x86_64). O Parakeet-TDT
é PyTorch nativo: usa o mesmo torch+CUDA já validado no resto do sistema
(vLLM, XTTS-v2), sem gap de wheel.

Medido no device (Jetson AGX Thor, GPU): RTFx ~57-90x em chamadas
aquecidas, ~65-95ms por transcrição de um turno de 2-5s de fala — bem
dentro do orçamento de NFR-L3 (≤300/600ms). Transcrição pt-BR correta em
teste round-trip contra o próprio XTTS-v2.

Mesmo contrato de `FasterWhisperStt` (docs/04 §3): `transcribe(pcm_f32,
partial)` roda em executor — o event loop nunca bloqueia em inferência
(CLAUDE.md "Concorrência").
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from config import SttConfig
from contracts.messages import Transcript


class ParakeetStt:
    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "nemo_toolkit[asr] não instalado — rode `pip install -e .[stt-gpu]` no "
                "device antes de transcrever de verdade (docs/04 §9)."
            ) from exc
        model = nemo_asr.models.ASRModel.from_pretrained(self._cfg.parakeet_model)
        model = model.to(self._cfg.device).eval()
        self._model = model

    async def transcribe(self, pcm_f32: np.ndarray, *, partial: bool) -> Transcript:
        """Transcreve um segmento de áudio do turno atual (docs/04 §3).

        `partial=True`: parciais durante a fala, feedback de latência apenas.
        `partial=False`: transcrição final no endpoint — é essa que aciona o cérebro.
        """
        self._ensure_loaded()
        t_start = time.time()
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._transcribe_sync, pcm_f32)
        return Transcript(text=text, is_final=not partial, t_start=t_start, t_end=time.time())

    def _transcribe_sync(self, pcm_f32: np.ndarray) -> str:
        # NeMo transcreve a partir de arquivo (ou manifest) — grava um WAV
        # temporário a 16kHz (taxa de treino do Parakeet); descartado após o uso.
        fd, path_str = tempfile.mkstemp(suffix=".wav")
        import os

        os.close(fd)
        path = Path(path_str)
        try:
            sf.write(path, pcm_f32, self._sample_rate(pcm_f32), subtype="FLOAT")
            result = self._model.transcribe([str(path)])
            hyp = result[0]
            return hyp.text if hasattr(hyp, "text") else str(hyp)
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _sample_rate(_pcm_f32: np.ndarray) -> int:
        # Contrato do resto do laço (doc04 §5, config.audio.sample_rate) já
        # entrega áudio a 16kHz — ver services/vad e captura de microfone.
        return 16000
