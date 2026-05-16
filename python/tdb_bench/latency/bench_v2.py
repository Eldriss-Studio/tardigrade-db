"""Latency-bench v2 primitives — Strategy + Template Method harness.

Decomposes the existing ad-hoc ``experiments/latency_benchmark.py``
script into testable primitives so the harness can be unit-tested
against a mock engine without GPU dependency. The real GPU run lives
in ``experiments/latency_benchmark_v2.py`` and supplies concrete
``IngestProbe`` and ``QueryWorkload`` implementations.

## Pattern stack

* **Strategy** — :class:`Timer`, :class:`QueryWorkload`,
  :class:`IngestProbe` are protocols with multiple
  implementations (real engine vs stub).
* **Template Method** — :class:`LatencyBenchRunner` sequences
  ``ingest → warmup → measured → report`` for every corpus size in
  :data:`CORPUS_SIZES`. Subclasses can override ``_collect_scale``
  for different sweep dimensions without rewriting the orchestration.
* **Repository / Value Object** — :class:`LatencyReport` and
  :class:`ScaleResult` carry the result shape; serialize to JSON for
  the positioning doc.

## Constants

Named in this module so callers don't pass magic numbers. Bumping the
defaults here propagates to every consumer.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol


# Corpus sizes the positioning doc quotes. Pinned here so the audit
# trail is reproducible: anyone running the bench gets the same scales.
CORPUS_SIZES: list[int] = [100, 1000, 5000]

# Per-scale warmup; runs before the measured queries to fill the SLB
# cache and trigger any one-time CUDA/cuBLAS allocations so the
# percentile distribution reflects steady-state latency.
WARMUP_QUERIES: int = 10

# Per-scale measured queries; the percentile distribution is computed
# from these latency samples. 100 gives p99 a 1% resolution.
MEASURED_QUERIES: int = 100

# Percentile shape reported per scale. p50/p95/p99 are standard ops
# latency-distribution markers in the positioning narrative.
LATENCY_PERCENTILES: list[int] = [50, 95, 99]


# ─── Strategy: Timer ──────────────────────────────────────────────────────


@dataclass
class TimerResult:
    """Output of a single ``Timer.measure()`` context."""

    elapsed_seconds: float = 0.0


class Timer(Protocol):
    """Strategy for measuring elapsed wall time around a code block."""

    @contextmanager
    def measure(self) -> Iterator[TimerResult]:
        ...  # pragma: no cover


class WallClockTimer:
    """Default Timer impl backed by :func:`time.perf_counter`."""

    @contextmanager
    def measure(self) -> Iterator[TimerResult]:
        result = TimerResult()
        start = time.perf_counter()
        try:
            yield result
        finally:
            result.elapsed_seconds = time.perf_counter() - start


# ─── Strategy: QueryWorkload ──────────────────────────────────────────────


@dataclass
class WorkloadResult:
    """Output of a single workload run."""

    latencies_ms: list[float] = field(default_factory=list)
    recall_at_k: float = 0.0


class QueryWorkload(Protocol):
    """Strategy: runs N queries and returns latencies + recall."""

    def run(self) -> WorkloadResult: ...  # pragma: no cover


class StubQueryWorkload:
    """Deterministic test double for the QueryWorkload Strategy.

    Generates ``num_queries`` of identical latency and reports
    ``recall_hits / num_queries`` as recall@k. Used by unit tests and
    as a fallback when the real engine isn't available.
    """

    def __init__(
        self, *, num_queries: int, recall_hits: int, simulated_latency_ms: float = 5.0
    ) -> None:
        if num_queries < 0:
            raise ValueError("num_queries must be >= 0")
        if not 0 <= recall_hits <= num_queries:
            raise ValueError(f"recall_hits {recall_hits} not in [0, {num_queries}]")
        self._n = num_queries
        self._hits = recall_hits
        self._lat = simulated_latency_ms

    def run(self) -> WorkloadResult:
        latencies = [self._lat for _ in range(self._n)]
        recall = self._hits / self._n if self._n else 0.0
        return WorkloadResult(latencies_ms=latencies, recall_at_k=recall)


# ─── Strategy: IngestProbe ────────────────────────────────────────────────


class IngestProbe(Protocol):
    """Strategy: ingest ``count`` items, return wall-time seconds."""

    def ingest(self, count: int) -> float: ...  # pragma: no cover


# ─── Repository: PercentileLatency ────────────────────────────────────────


class PercentileLatency:
    """Pure-function helper: percentile dict from a list of samples."""

    @staticmethod
    def from_samples_ms(
        samples: list[float], percentiles: list[int] | None = None
    ) -> dict[str, float]:
        ps = percentiles if percentiles is not None else LATENCY_PERCENTILES
        if not samples:
            return {f"p{q}": 0.0 for q in ps}
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        return {f"p{q}": _percentile(sorted_samples, q, n) for q in ps}


def _percentile(sorted_samples: list[float], q: int, n: int) -> float:
    """Linear-interpolation percentile (NumPy ``linear`` method)."""
    if n == 1:
        return sorted_samples[0]
    rank = (q / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_samples[lo]
    frac = rank - lo
    return sorted_samples[lo] + frac * (sorted_samples[hi] - sorted_samples[lo])


# ─── Value Objects: ScaleResult + LatencyReport ───────────────────────────


@dataclass
class ScaleResult:
    """One row of the latency table — measured at a single corpus size."""

    corpus_size: int
    ingest_seconds: float
    recall_at_k: float
    latency_percentiles_ms: dict[str, float]
    measured_query_count: int


@dataclass
class LatencyReport:
    """The full bench output — one ScaleResult per corpus size."""

    scales: list[ScaleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scales": [
                {
                    "corpus_size": s.corpus_size,
                    "ingest_seconds": s.ingest_seconds,
                    "recall_at_k": s.recall_at_k,
                    "latency_percentiles_ms": s.latency_percentiles_ms,
                    "measured_query_count": s.measured_query_count,
                }
                for s in self.scales
            ]
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )


# ─── Template Method: LatencyBenchRunner ──────────────────────────────────


_WorkloadFactory = Callable[..., QueryWorkload]


class LatencyBenchRunner:
    """Sequences ingest → warmup → measured → report per corpus size."""

    def __init__(
        self,
        *,
        ingest_probe: IngestProbe,
        workload_factory: _WorkloadFactory,
        corpus_sizes: list[int] = CORPUS_SIZES,
    ) -> None:
        self._ingest = ingest_probe
        self._factory = workload_factory
        self._sizes = corpus_sizes

    def run(self) -> LatencyReport:
        scales = [self._collect_scale(size) for size in self._sizes]
        return LatencyReport(scales=scales)

    def _collect_scale(self, corpus_size: int) -> ScaleResult:
        ingest_seconds = self._ingest.ingest(corpus_size)
        # Warmup — runs but discarded for percentile computation.
        warmup = self._build_workload(corpus_size, phase="warmup")
        _ = warmup.run()
        # Measured — drives the percentile table.
        measured = self._build_workload(corpus_size, phase="measured")
        result = measured.run()
        return ScaleResult(
            corpus_size=corpus_size,
            ingest_seconds=ingest_seconds,
            recall_at_k=result.recall_at_k,
            latency_percentiles_ms=PercentileLatency.from_samples_ms(result.latencies_ms),
            measured_query_count=len(result.latencies_ms),
        )

    def _build_workload(self, corpus_size: int, *, phase: str) -> QueryWorkload:
        """Factory invocation; supports both 1-arg and (size, phase) factories."""
        try:
            return self._factory(corpus_size, phase=phase)  # type: ignore[call-arg]
        except TypeError:
            return self._factory(corpus_size)
