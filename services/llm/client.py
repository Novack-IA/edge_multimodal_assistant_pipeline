"""Cliente HTTP assíncrono streaming contra o vLLM (docs/06 §4, docs/13 §6).

Fala com `{server_url}/chat/completions` (API OpenAI-compatible,
`stream=true`) e devolve um gerador assíncrono de `LlmToken | ToolCall`. O
parser `lfm2` do vLLM já converte a saída pythonic do modelo em
`tool_calls` estruturados — não fazemos parsing próprio (docs/06 §3).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from config import LlmConfig
from contracts.messages import LlmToken, OpenAIMessage, SamplingCfg, ToolCall, ToolSpec


class LlmClient:
    def __init__(self, cfg: LlmConfig, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._cfg = cfg
        self._client = http_client or httpx.AsyncClient(
            base_url=cfg.server_url,
            timeout=httpx.Timeout(60.0, connect=5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def warmup(self) -> None:
        """Turno dummy pós-boot — não aceitar turnos de usuário antes disso (NFR-R6, docs/13 §4)."""
        dummy_cancel = asyncio.Event()
        dummy_sampling = SamplingCfg(max_new_tokens=4)
        messages: list[OpenAIMessage] = [{"role": "user", "content": "oi"}]
        async for _event in self.stream_chat(messages, tools=[], sampling=dummy_sampling, cancel_token=dummy_cancel):
            pass

    async def stream_chat(
        self,
        messages: list[OpenAIMessage],
        tools: list[ToolSpec],
        sampling: SamplingCfg,
        cancel_token: asyncio.Event,
    ) -> AsyncIterator[LlmToken | ToolCall]:
        """POST streaming em /chat/completions (stream=true).

        Cada chunk SSE de texto vira um `LlmToken`; ao fechar uma tool call
        (`finish_reason == "tool_calls"`), emite `ToolCall`s acumuladas.
        Fecha a conexão imediatamente se `cancel_token` estiver setado —
        é assim que o barge-in aborta a geração (docs/08 §3, NFR-L1).
        """
        payload: dict = {
            "model": self._cfg.backbone,
            "messages": messages,
            "stream": True,
            "temperature": sampling.tool_temperature if tools else sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_new_tokens,
        }
        if tools:
            payload["tools"] = [t.as_openai_tool() for t in tools]
            payload["tool_choice"] = "auto"

        tool_call_accum: dict[int, dict] = {}
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if cancel_token.is_set():
                    return  # sai do `async with` -> fecha a conexão HTTP (barge-in)
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                for event in self._events_from_delta(delta, tool_call_accum):
                    yield event

                finish_reason = choice.get("finish_reason")
                if finish_reason == "tool_calls":
                    for idx in sorted(tool_call_accum):
                        acc = tool_call_accum[idx]
                        try:
                            args = json.loads(acc["arguments"]) if acc["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        yield ToolCall(id=acc["id"], name=acc["name"], arguments=args)
                    tool_call_accum.clear()
                elif finish_reason:
                    yield LlmToken(text="", is_tool_call=False, done=True)

    @staticmethod
    def _events_from_delta(delta: dict, tool_call_accum: dict[int, dict]) -> list[LlmToken]:
        events: list[LlmToken] = []
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                idx = tc.get("index", 0)
                acc = tool_call_accum.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    acc["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["arguments"] += fn["arguments"]
            events.append(LlmToken(text="", is_tool_call=True, done=False))
        elif delta.get("content"):
            events.append(LlmToken(text=delta["content"], is_tool_call=False, done=False))
        return events
