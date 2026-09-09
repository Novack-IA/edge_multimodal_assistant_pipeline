"""Timeout, cancelamento (barge-in) e erro gracioso em tool calling (docs/06 §3, FR-12)."""

import asyncio

import pytest

from config import ToolsConfig
from contracts.messages import ToolCall, ToolSpec
from tools.registry import ToolRegistry


def make_registry() -> ToolRegistry:
    return ToolRegistry(ToolsConfig(timeout_ms=100))


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_result_not_raise():
    registry = make_registry()
    result = await registry.run(ToolCall(id="1", name="nope", arguments={}), asyncio.Event())
    assert result.is_error
    assert "desconhecida" in result.content


@pytest.mark.asyncio
async def test_successful_tool_call():
    registry = make_registry()
    spec = ToolSpec(name="echo", description="", parameters={}, timeout_ms=1000)

    @registry.register(spec)
    async def _echo(msg: str) -> str:
        return msg

    result = await registry.run(ToolCall(id="1", name="echo", arguments={"msg": "oi"}), asyncio.Event())
    assert not result.is_error
    assert result.content == "oi"


@pytest.mark.asyncio
async def test_tool_exception_becomes_error_result():
    registry = make_registry()
    spec = ToolSpec(name="boom", description="", parameters={}, timeout_ms=1000)

    @registry.register(spec)
    async def _boom() -> str:
        raise ValueError("deu ruim")

    result = await registry.run(ToolCall(id="1", name="boom", arguments={}), asyncio.Event())
    assert result.is_error
    assert "deu ruim" in result.content


@pytest.mark.asyncio
async def test_tool_timeout():
    registry = ToolRegistry(ToolsConfig(timeout_ms=20))
    spec = ToolSpec(name="slow", description="", parameters={}, timeout_ms=20)

    @registry.register(spec)
    async def _slow() -> str:
        await asyncio.sleep(1)
        return "nunca chega"

    result = await registry.run(ToolCall(id="1", name="slow", arguments={}), asyncio.Event())
    assert result.is_error
    assert "timeout" in result.content.lower()


@pytest.mark.asyncio
async def test_tool_cancelled_by_barge_in():
    registry = make_registry()
    spec = ToolSpec(name="slow", description="", parameters={}, timeout_ms=5000)

    @registry.register(spec)
    async def _slow() -> str:
        await asyncio.sleep(5)
        return "nunca chega"

    cancel_token = asyncio.Event()

    async def trigger_cancel():
        await asyncio.sleep(0.02)
        cancel_token.set()

    asyncio.ensure_future(trigger_cancel())
    with pytest.raises(asyncio.CancelledError):
        await registry.run(ToolCall(id="1", name="slow", arguments={}), cancel_token)
