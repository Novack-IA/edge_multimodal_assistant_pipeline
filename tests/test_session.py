"""Invariantes do contrato de turno/sessão (docs/09 §9) — a área mais
sensível do sistema (CLAUDE.md invariante 5)."""

from config import LlmConfig
from services.llm.session import Session


def make_session(**overrides) -> Session:
    cfg = LlmConfig(**overrides)
    return Session(system_prompt="system fixo", cfg=cfg)


def test_system_and_history_never_contains_images():
    session = make_session()
    session.commit("olá", "oi, tudo bem?")
    for msg in session.system_and_history:
        assert isinstance(msg["content"], str)  # nunca lista de content parts (imagens)


def test_commit_is_the_only_write_path():
    session = make_session()
    assert session.history == []
    session.commit("pergunta", "resposta")
    assert len(session.history) == 1
    assert session.history[0].user_text == "pergunta"
    assert session.history[0].assistant_text == "resposta"


def test_history_window_respects_history_max_turns():
    session = make_session(history_max_turns=2)
    for i in range(5):
        session.commit(f"pergunta {i}", f"resposta {i}")
    # +1 pela mensagem system
    assert len(session.system_and_history) == 1 + 2 * 2
    # deve conter só os 2 turnos mais recentes
    assert session.system_and_history[1]["content"] == "pergunta 3"


def test_system_prompt_is_stable_prefix_for_apc():
    # docs/09 §3: [SYSTEM] literalmente estável entre turnos, senão quebra o
    # Automatic Prefix Caching do vLLM.
    session = make_session()
    before = session.system_and_history[0]
    session.commit("a", "b")
    after = session.system_and_history[0]
    assert before == after


def test_fits_context_true_when_small():
    session = make_session(context_length=32768)
    assert session.fits_context("oi", n_image_frames=4, video_token_budget=256, reserve_for_response=256)


def test_fits_context_false_when_budget_exceeded():
    session = make_session(context_length=1000)
    assert not session.fits_context(
        "x" * 10, n_image_frames=10, video_token_budget=500, reserve_for_response=256
    )
