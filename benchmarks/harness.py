"""Harness de latência — p50/p95, nunca só média (docs/11 §3).

Cada `bench_*.py` usa `LatencyBench` para coletar N amostras (descartando
warm-up) e falha (`assert_within_budget`) se os percentis estourarem os
alvos de `config.yaml: latency_budgets_ms` (docs/02 §2).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class LatencyBench:
    name: str
    samples_ms: list[float] = field(default_factory=list)

    @contextmanager
    def measure(self):
        t0 = time.perf_counter()
        yield
        self.samples_ms.append((time.perf_counter() - t0) * 1000)

    async def run_async(
        self, fn: Callable[[], Awaitable[None]], *, n: int, warmup: int = 1
    ) -> None:
        for _ in range(warmup):
            await fn()
        for _ in range(n):
            t0 = time.perf_counter()
            await fn()
            self.samples_ms.append((time.perf_counter() - t0) * 1000)

    def percentile(self, p: float) -> float:
        if not self.samples_ms:
            raise ValueError(f"bench {self.name!r} não coletou nenhuma amostra")
        return statistics.quantiles(self.samples_ms, n=100, method="inclusive")[int(p) - 1]

    @property
    def p50(self) -> float:
        return self.percentile(50) if len(self.samples_ms) > 1 else self.samples_ms[0]

    @property
    def p95(self) -> float:
        return self.percentile(95) if len(self.samples_ms) > 1 else self.samples_ms[0]

    def report(self) -> str:
        return (
            f"{self.name}: n={len(self.samples_ms)} "
            f"p50={self.p50:.1f}ms p95={self.p95:.1f}ms "
            f"min={min(self.samples_ms):.1f}ms max={max(self.samples_ms):.1f}ms"
        )

    def assert_within_budget(self, *, p50_ms: float | None = None, p95_ms: float | None = None) -> None:
        """Falha o CI se estourar os alvos de docs/02 §2 (chamar do teste de latência)."""
        if p50_ms is not None:
            assert self.p50 <= p50_ms, f"{self.name} p50={self.p50:.1f}ms > orçamento {p50_ms}ms"
        if p95_ms is not None:
            assert self.p95 <= p95_ms, f"{self.name} p95={self.p95:.1f}ms > orçamento {p95_ms}ms"
