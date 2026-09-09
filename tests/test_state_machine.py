"""Máquina de estados + barge-in (docs/08 §2/§7)."""

import asyncio

import pytest

from contracts.messages import TurnState, VadEvent
from orchestrator.state_machine import ConversationStateMachine


@pytest.mark.asyncio
async def test_idle_to_listening_on_speech_start():
    sm = ConversationStateMachine()
    await sm.on_vad_event(VadEvent(kind="speech_start", t=0.0))
    assert sm.state == TurnState.LISTENING


@pytest.mark.asyncio
async def test_listening_to_thinking_on_speech_end():
    sm = ConversationStateMachine()
    await sm.on_vad_event(VadEvent(kind="speech_start", t=0.0))
    await sm.on_vad_event(VadEvent(kind="speech_end", t=1.0))
    assert sm.state == TurnState.THINKING


@pytest.mark.asyncio
async def test_barge_in_from_thinking_returns_to_listening():
    sm = ConversationStateMachine()
    sm.enter(TurnState.THINKING)
    old_token = sm.turn.cancel_token
    await sm.on_vad_event(VadEvent(kind="speech_start", t=2.0))
    assert sm.state == TurnState.LISTENING
    assert old_token.is_set()  # o turno abortado observa o cancelamento
    assert not sm.turn.cancel_token.is_set()  # novo token limpo para o próximo turno


@pytest.mark.asyncio
async def test_barge_in_from_speaking_calls_callback():
    called = asyncio.Event()

    async def on_barge_in():
        called.set()

    sm = ConversationStateMachine(on_barge_in=on_barge_in)
    sm.enter(TurnState.SPEAKING)
    await sm.on_vad_event(VadEvent(kind="speech_start", t=0.0))
    assert sm.state == TurnState.LISTENING
    assert called.is_set()


@pytest.mark.asyncio
async def test_speech_start_in_idle_is_not_a_barge_in():
    triggered = False

    async def on_barge_in():
        nonlocal triggered
        triggered = True

    sm = ConversationStateMachine(on_barge_in=on_barge_in)
    await sm.on_vad_event(VadEvent(kind="speech_start", t=0.0))
    assert not triggered
    assert sm.state == TurnState.LISTENING
