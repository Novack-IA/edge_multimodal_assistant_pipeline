"""M0 gate: os contratos compilam e a config carrega e valida (CLAUDE.md)."""

from config import load_config


def test_load_config_example():
    cfg = load_config()
    assert cfg.platform.device == "thor"
    assert cfg.audio.sample_rate == 16000


def test_context_length_matches_model_ceiling():
    # NFR-R7 / docs/09 §6: teto real do LFM2.5-VL-3B é 32768 — não é "só um número
    # grande para garantir" (docs/13 §9 riscos).
    cfg = load_config()
    assert cfg.llm.context_length == 32768


def test_latency_budgets_present():
    cfg = load_config()
    budgets = cfg.latency_budgets_ms
    assert budgets.barge_in_p50 <= budgets.barge_in_p95
    assert budgets.roundtrip_p50 <= budgets.roundtrip_p95
