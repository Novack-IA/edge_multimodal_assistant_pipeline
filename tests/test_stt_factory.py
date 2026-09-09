"""docs/04 §9: engine de STT trocável por config, sem GPU necessária para o teste
(a fábrica só decide a classe — carregar o modelo é lazy, ver `_ensure_loaded`)."""

from config import SttConfig
from services.stt import ParakeetStt, create_stt_engine
from services.stt.worker import FasterWhisperStt


def test_default_engine_is_parakeet():
    engine = create_stt_engine(SttConfig())
    assert isinstance(engine, ParakeetStt)


def test_faster_whisper_fallback_selectable():
    engine = create_stt_engine(SttConfig(engine="faster_whisper"))
    assert isinstance(engine, FasterWhisperStt)
