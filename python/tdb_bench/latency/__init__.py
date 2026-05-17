"""Latency benchmark primitives for the positioning reframe.

Strategy + Template Method harness that captures ingest time and
query-latency percentiles at fixed corpus sizes. The primitives are
unit-tested with mocks; the real GPU bench lives in
``experiments/latency_benchmark_v2.py``.
"""

from .bench_v2 import (
    CORPUS_SIZES,
    LATENCY_PERCENTILES,
    MEASURED_QUERIES,
    WARMUP_QUERIES,
    LatencyBenchRunner,
    LatencyReport,
    PercentileLatency,
    ScaleResult,
    StubQueryWorkload,
    Timer,
    WallClockTimer,
    WorkloadResult,
)

__all__ = [
    "CORPUS_SIZES",
    "LATENCY_PERCENTILES",
    "MEASURED_QUERIES",
    "WARMUP_QUERIES",
    "LatencyBenchRunner",
    "LatencyReport",
    "PercentileLatency",
    "ScaleResult",
    "StubQueryWorkload",
    "Timer",
    "WallClockTimer",
    "WorkloadResult",
]
