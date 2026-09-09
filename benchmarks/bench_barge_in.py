#!/usr/bin/env python3
"""NFR-L1 — barge-in stop (fala detectada -> geração/TTS param), docs/11.

Mede só a fração *software* do orçamento: tempo entre `cancel_token.set()`
e o loop de streaming do LlmClient observar o cancelamento e retornar (a
fração de detecção VAD + flush de playback exige hardware real de
áudio/mic — M3/M4, ver docs/08 §3 para a decomposição completa do orçamento).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.harness import LatencyBench
from config import load_config
from contracts.messages import LlmToken


class _FakeStreamingClient:
    """Simula um stream SSE lento o bastante para observarmos o cancelamento
    no meio, sem depender de um servidor vLLM real estar de pé."""

    async def stream_chat(self, messages, tools, sampling, cancel_token):
        for _ in range(1000):
            if cancel_token.is_set():
                return
            await asyncio.sleep(0.005)
            yield LlmToken(text="a", is_tool_call=False, done=False)


async def main(n: int = 30) -> None:
    cfg = load_config()
    client = _FakeStreamingClient()
    bench = LatencyBench(name="bench_barge_in (fração software: cancel -> stream retorna)")

    async def one_run() -> None:
        cancel_token = asyncio.Event()
        gen = client.stream_chat([], [], None, cancel_token)

        async def consume():
            async for _ in gen:
                pass

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.02)  # deixa o stream "andar" um pouco antes de interromper
        import time

        t0 = time.perf_counter()
        cancel_token.set()
        await task
        bench.samples_ms.append((time.perf_counter() - t0) * 1000)

    for _ in range(n):
        await one_run()
    print(bench.report())
    bench.assert_within_budget(p50_ms=cfg.latency_budgets_ms.barge_in_p50, p95_ms=cfg.latency_budgets_ms.barge_in_p95)


if __name__ == "__main__":
    asyncio.run(main())
