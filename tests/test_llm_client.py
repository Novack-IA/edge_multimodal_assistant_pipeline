"""Parsing do SSE do vLLM e respeito ao cancel_token (docs/06 §4, NFR-L4/barge-in).

Usa `httpx.MockTransport` — nenhum servidor vLLM real é necessário para
este teste de contrato (docs/11 §1.1).
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from config import LlmConfig
from contracts.messages import LlmToken, SamplingCfg, ToolCall
from services.llm.client import LlmClient


def sse_body(*events: dict) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def make_client(transport: httpx.MockTransport) -> LlmClient:
    cfg = LlmConfig()
    http_client = httpx.AsyncClient(transport=transport, base_url=cfg.server_url)
    return LlmClient(cfg, http_client=http_client)


@pytest.mark.asyncio
async def test_stream_chat_yields_text_tokens():
    body = sse_body(
        {"choices": [{"delta": {"content": "Ol"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "á"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    client = make_client(transport)

    events = []
    async for e in client.stream_chat([], tools=[], sampling=SamplingCfg(), cancel_token=asyncio.Event()):
        events.append(e)

    texts = [e.text for e in events if isinstance(e, LlmToken) and e.text]
    assert "".join(texts) == "Olá"
    assert any(isinstance(e, LlmToken) and e.done for e in events)
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_chat_yields_tool_call_on_finish():
    body = sse_body(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "function": {"name": "get_current_time", "arguments": ""}}
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, "finish_reason": None}
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    client = make_client(transport)

    events = [e async for e in client.stream_chat([], tools=[], sampling=SamplingCfg(), cancel_token=asyncio.Event())]

    tool_calls = [e for e in events if isinstance(e, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_current_time"
    assert tool_calls[0].arguments == {}
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_chat_stops_when_cancelled():
    # Muitos eventos — se o cancel_token não for respeitado, o teste veria todos.
    body = sse_body(*({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]} for _ in range(50)))
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    client = make_client(transport)

    cancel_token = asyncio.Event()
    cancel_token.set()  # já cancelado antes de começar a consumir — pior caso
    events = [e async for e in client.stream_chat([], tools=[], sampling=SamplingCfg(), cancel_token=cancel_token)]
    assert events == []
    await client.aclose()
