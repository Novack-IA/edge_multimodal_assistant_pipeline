from config import SttConfig
from services.stt.parakeet import ParakeetStt
from services.stt.worker import FasterWhisperStt

__all__ = ["FasterWhisperStt", "ParakeetStt", "create_stt_engine"]


def create_stt_engine(cfg: SttConfig) -> FasterWhisperStt | ParakeetStt:
    """Fábrica trocável por config (docs/04 §9) — mesmo padrão de
    `services/tts` (engine agnóstico) e `llm.runtime` (docs/13 §5)."""
    if cfg.engine == "parakeet":
        return ParakeetStt(cfg)
    return FasterWhisperStt(cfg)
