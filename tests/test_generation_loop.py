"""Laço de geração (docs/09 §8): tool calling ponta a ponta, commit só no
sucesso, nada commitado em barge-in (as 5 regras invioláveis de docs/09 §9)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from config import LlmConfig, ToolsConfig
from contracts.messages import LlmToken, SamplingCfg, ToolCall, ToolSpec
from services.llm.generation_loop import run_turn
from services.llm.session import Session
from tools.registry import ToolRegistry


class FakeLlmClient:
    """Primeira chamada emite uma tool call; a segunda (após o resultado
    injetado) emite a resposta final em texto — imita o padrão ReAct do
    parser lfm2 (docs/06 §3)."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    async def stream_chat(
        self, messages: list[dict], tools: list[ToolSpec], sampling: SamplingCfg, cancel_token: asyncio.Event
    ) -> AsyncIterator:
        self.calls.append(messages)
        if len(self.calls) == 1:
            yield ToolCall(id="call_1", name="get_current_time", arguments={})
        else:
            for ch in ["Ag", "ora", " são 10h."]:
                if cancel_token.is_set():
                    return
                yield LlmToken(text=ch, is_tool_call=False, done=False)
            yield LlmToken(text="", is_tool_call=False, done=True)


def make_session_and_registry():
    session = Session(system_prompt="sys", cfg=LlmConfig())
    registry = ToolRegistry(ToolsConfig())

    spec = ToolSpec(name="get_current_time", description="", parameters={}, timeout_ms=1000)

    @registry.register(spec)
    async def _get_current_time() -> str:
        return "10:00"

    return session, registry


@pytest.mark.asyncio
async def test_run_turn_executes_tool_then_commits_text_only():
    session, registry = make_session_and_registry()
    client = FakeLlmClient()
    cancel_token = asyncio.Event()

    chunks = []
    async for text in run_turn(
        session, client, "que horas são?", frames=[], tools=registry.specs(), tool_registry=registry,
        cancel_token=cancel_token,
    ):
        chunks.append(text)

    assert "".join(chunks) == "Agora são 10h."
    assert len(session.history) == 1
    assert session.history[0].user_text == "que horas são?"
    assert session.history[0].assistant_text == "Agora são 10h."

    # segunda chamada ao "vLLM" já deve carregar o resultado da ferramenta
    second_call_messages = client.calls[1]
    assert any(m.get("role") == "tool" and m.get("content") == "10:00" for m in second_call_messages)


@pytest.mark.asyncio
async def test_run_turn_barge_in_commits_nothing():
    session, registry = make_session_and_registry()

    class SlowClient:
        async def stream_chat(self, messages, tools, sampling, cancel_token):
            yield LlmToken(text="Ol", is_tool_call=False, done=False)
            cancel_token.set()  # barge-in acontece no meio da geração
            yield LlmToken(text="á, tudo bem?", is_tool_call=False, done=False)

    cancel_token = asyncio.Event()
    chunks = []
    async for text in run_turn(
        session, SlowClient(), "oi", frames=[], tools=[], tool_registry=registry, cancel_token=cancel_token
    ):
        chunks.append(text)

    assert chunks == ["Ol"]  # parou assim que cancel_token foi setado
    assert session.history == []  # docs/09 §9 regra 2: nada commitado em barge-in
