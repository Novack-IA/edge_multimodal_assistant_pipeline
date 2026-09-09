"""Silero VAD — endpointing e gatilho de barge-in (docs/04 §4, docs/08 §4).

Import do pacote `silero-vad` é lazy: o módulo carrega sem GPU/onnxruntime
instalado (útil para testes de contrato); só falha ao instanciar
`SileroEndpointDetector` de verdade sem o extra `vad` instalado.
"""

from __future__ import annotations

import numpy as np

from config import VadConfig
from contracts.messages import AudioChunk, VadEvent


class SileroEndpointDetector:
    """Consome AudioChunk mono 16-bit PCM e emite VadEvent em speech_start/speech_end.

    Uma instância é usada em LISTENING para endpointing (NFR-L2) e outra,
    dedicada e de baixa latência, roda em paralelo durante SPEAKING para
    detectar barge-in (docs/04 §4, docs/08 §3) — não compartilhar estado
    entre as duas.
    """

    def __init__(self, cfg: VadConfig, *, sample_rate: int = 16000) -> None:
        self._cfg = cfg
        self._sample_rate = sample_rate
        self._model = None
        self._iterator = None

    def _ensure_loaded(self) -> None:
        if self._iterator is not None:
            return
        try:
            from silero_vad import VADIterator, load_silero_vad
        except ImportError as exc:  # pragma: no cover — exercitado só sem o extra instalado
            raise RuntimeError(
                "silero-vad não instalado — rode `pip install -e .[vad]` no device "
                "antes de usar SileroEndpointDetector de verdade (docs/04)."
            ) from exc
        self._model = load_silero_vad(onnx=True)
        self._iterator = VADIterator(
            self._model,
            sampling_rate=self._sample_rate,
            threshold=self._cfg.threshold,
            min_silence_duration_ms=self._cfg.min_silence_ms,
            speech_pad_ms=self._cfg.speech_pad_ms,
        )

    def process_chunk(self, chunk: AudioChunk) -> VadEvent | None:
        """Retorna um VadEvent se este chunk cruzou uma fronteira speech/silence."""
        self._ensure_loaded()
        pcm_f32 = np.frombuffer(chunk.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        event = self._iterator(pcm_f32, return_seconds=False)
        if event is None:
            return None
        if "start" in event:
            return VadEvent(kind="speech_start", t=chunk.t_capture)
        if "end" in event:
            return VadEvent(kind="speech_end", t=chunk.t_capture)
        return None

    def reset(self) -> None:
        """Zera o estado interno (chamar ao trocar de turno/sessão)."""
        if self._iterator is not None:
            self._iterator.reset_states()
