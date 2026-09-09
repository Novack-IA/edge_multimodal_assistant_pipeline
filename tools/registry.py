"""Registro de ferramentas expostas ao tool calling (docs/06 §3).

Cada ferramenta: nome, JSON schema de argumentos, função async, timeout,
tratamento de erro. Ferramentas com efeitos colaterais devem ser
idempotentes ou canceláveis — barge-in cancela a coroutine em voo
(docs/06 §3 "Barge-in durante tool call", docs/08 §2).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from config import ToolsConfig
from contracts.messages import ToolCall, ToolResult, ToolSpec

ToolFn = Callable[..., Awaitable[Any]]


class ToolRegistry:
    def __init__(self, cfg: ToolsConfig) -> None:
        self._cfg = cfg
        self._tools: dict[str, tuple[ToolSpec, ToolFn]] = {}

    def register(self, spec: ToolSpec) -> Callable[[ToolFn], ToolFn]:
        def decorator(fn: ToolFn) -> ToolFn:
            self._tools[spec.name] = (spec, fn)
            return fn

        return decorator

    def specs(self) -> list[ToolSpec]:
        return [spec for spec, _fn in self._tools.values()]

    async def run(self, call: ToolCall, cancel_token: asyncio.Event) -> ToolResult:
        """Executa `call` com timeout e cancelamento cooperativo. Nunca
        levanta — falha vira `ToolResult(is_error=True)` (docs FR-12)."""
        entry = self._tools.get(call.name)
        if entry is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"ferramenta desconhecida: {call.name}",
                is_error=True,
                t=time.time(),
            )
        spec, fn = entry
        try:
            content = await self._run_cancellable(fn, call.arguments, spec.timeout_ms, cancel_token)
            return ToolResult(
                tool_call_id=call.id, name=call.name, content=str(content), is_error=False, t=time.time()
            )
        except asyncio.CancelledError:
            raise  # barge-in: propaga, quem chamou já retorna sem commitar
        except Exception as exc:  # noqa: BLE001 — erro de ferramenta vira mensagem `tool`, não crash
            return ToolResult(
                tool_call_id=call.id, name=call.name, content=str(exc), is_error=True, t=time.time()
            )

    @staticmethod
    async def _run_cancellable(
        fn: ToolFn, arguments: dict[str, Any], timeout_ms: int, cancel_token: asyncio.Event
    ) -> Any:
        call_task = asyncio.ensure_future(fn(**arguments))
        cancel_wait = asyncio.ensure_future(cancel_token.wait())
        try:
            done, pending = await asyncio.wait(
                {call_task, cancel_wait},
                timeout=timeout_ms / 1000,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            pass
        if call_task in done:
            for p in pending:
                p.cancel()
            return call_task.result()
        for p in pending:
            p.cancel()
        if cancel_wait in done:
            raise asyncio.CancelledError("barge-in durante tool call")
        raise TimeoutError(f"ferramenta excedeu timeout de {timeout_ms}ms")
