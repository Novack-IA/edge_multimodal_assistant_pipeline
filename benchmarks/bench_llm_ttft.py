#!/usr/bin/env python3
"""NFR-L4 — TTFT do vLLM (mensagens prontas, com imagens -> 1º token), docs/11.

Requer um servidor vLLM real respondendo em `llm.server_url` (docs/13). Sem
ele, imprime instruções em vez de números fabricados — latência é medida,
não estimada (CLAUDE.md invariante 6).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.harness import LatencyBench
from config import load_config
from contracts.messages import SamplingCfg
from services.llm.client import LlmClient


async def main(n: int = 20) -> None:
    cfg = load_config()
    client = LlmClient(cfg.llm)
    if not await client.health_check():
        print(
            f"vLLM não está respondendo em {cfg.llm.server_url} — suba o servidor primeiro "
            "(deployment/serve_vllm.py) antes de rodar este benchmark (docs/13)."
        )
        await client.aclose()
        raise SystemExit(1)

    bench = LatencyBench(name="bench_llm_ttft")
    messages = [
        {"role": "system", "content": "Você é um assistente conversacional em pt-BR, respostas curtas."},
        {"role": "user", "content": "Descreva em uma frase o que você vê agora."},
    ]
    sampling = SamplingCfg(max_new_tokens=8)

    async def one_call() -> None:
        cancel_token = asyncio.Event()
        async for _event in client.stream_chat(messages, tools=[], sampling=sampling, cancel_token=cancel_token):
            return  # mede só até o primeiro evento — TTFT

    await bench.run_async(one_call, n=n, warmup=2)
    print(bench.report())
    bench.assert_within_budget(p50_ms=cfg.latency_budgets_ms.llm_ttft_p50)
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
