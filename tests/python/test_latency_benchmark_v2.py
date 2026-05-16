"""ATDD: latency_benchmark_v2 primitives (Track A, slice A1).

Unit-tests the Strategy + Template Method primitives that drive
the latency-benchmark harness against a mock engine. The harness
itself runs against real Qwen3-0.6B in
``experiments/latency_benchmark_v2.py``; this file pins the
contracts.

Slices covered: A1.1 (recall canary primitive), A1.2 (latency
distribution capture), A1.3 (end-to-end ingest timing).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tdb_bench.latency.bench_v2 import (
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
)


# ─── A1 building blocks: timer + percentile primitives ────────────────────


class TestTimer:
    def test_wall_clock_timer_records_elapsed_seconds(self):
        timer = WallClockTimer()
        with timer.measure() as result:
            x = sum(range(100))  # microseconds
            assert x == 4950
        assert result.elapsed_seconds > 0
        assert result.elapsed_seconds < 1.0

    def test_substitutable_for_protocol(self):
        timer: Timer = WallClockTimer()
        with timer.measure() as r:
            pass
        assert r.elapsed_seconds >= 0


class TestPercentileLatency:
    def test_p50_p95_p99_from_samples(self):
        # 100 samples from 1ms to 100ms — p50 ≈ 50, p95 ≈ 95, p99 ≈ 99.
        samples = [float(i) for i in range(1, 101)]
        p = PercentileLatency.from_samples_ms(samples, percentiles=[50, 95, 99])
        assert 49 <= p["p50"] <= 51
        assert 94 <= p["p95"] <= 96
        assert 98 <= p["p99"] <= 100

    def test_empty_samples_returns_zero(self):
        # Degenerate input must not crash the runner mid-bench.
        p = PercentileLatency.from_samples_ms([], percentiles=[50, 95, 99])
        assert p == {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def test_pinned_percentiles_match_constants(self):
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        p = PercentileLatency.from_samples_ms(samples)  # uses LATENCY_PERCENTILES default
        assert set(p.keys()) == {f"p{q}" for q in LATENCY_PERCENTILES}


# ─── A1.1 — recall canary primitive ───────────────────────────────────────


class TestStubQueryWorkload:
    def test_records_per_query_latency_and_recall(self):
        # Stub workload generates synthetic queries; the primitive
        # captures latencies in ms and recall@k as a 0..1 ratio.
        workload = StubQueryWorkload(
            num_queries=10,
            recall_hits=8,  # 8/10 = 0.8
            simulated_latency_ms=5.0,
        )
        result = workload.run()

        assert len(result.latencies_ms) == 10
        assert all(abs(x - 5.0) < 0.1 for x in result.latencies_ms)
        assert result.recall_at_k == pytest.approx(0.8)

    def test_recall_at_k_in_valid_range(self):
        for hits in (0, 5, 10):
            r = StubQueryWorkload(num_queries=10, recall_hits=hits).run()
            assert 0.0 <= r.recall_at_k <= 1.0


# ─── A1.2 + A1.3 — full runner via Template Method ───────────────────────


class _FakeIngestProbe:
    """Records ingest count and returns a deterministic elapsed time."""

    def __init__(self, ingest_seconds_per_call: float = 0.05) -> None:
        self._t = ingest_seconds_per_call
        self.calls: list[int] = []

    def ingest(self, count: int) -> float:
        self.calls.append(count)
        return self._t


def _stub_workload(corpus_size: int, recall: float = 0.95) -> StubQueryWorkload:
    return StubQueryWorkload(
        num_queries=MEASURED_QUERIES,
        recall_hits=int(MEASURED_QUERIES * recall),
        simulated_latency_ms=2.0 + 0.001 * corpus_size,  # tiny growth with scale
    )


class TestLatencyBenchRunner:
    def test_runs_all_corpus_sizes(self):
        probe = _FakeIngestProbe()
        runner = LatencyBenchRunner(
            ingest_probe=probe,
            workload_factory=_stub_workload,
            corpus_sizes=CORPUS_SIZES,
        )

        report = runner.run()

        assert isinstance(report, LatencyReport)
        assert len(report.scales) == len(CORPUS_SIZES)
        for scale in report.scales:
            assert isinstance(scale, ScaleResult)
            assert scale.corpus_size in CORPUS_SIZES

    def test_each_scale_has_p50_p95_p99(self):
        probe = _FakeIngestProbe()
        runner = LatencyBenchRunner(
            ingest_probe=probe,
            workload_factory=_stub_workload,
            corpus_sizes=[100],
        )

        report = runner.run()

        only = report.scales[0]
        for q in LATENCY_PERCENTILES:
            assert f"p{q}" in only.latency_percentiles_ms
            assert only.latency_percentiles_ms[f"p{q}"] > 0

    def test_largest_scale_reports_ingest_seconds(self):
        # A1.3: end-to-end ingest is timed for at least the largest
        # corpus size in the run. The runner asks the ingest probe
        # for each scale's elapsed time and records it.
        probe = _FakeIngestProbe(ingest_seconds_per_call=0.5)
        runner = LatencyBenchRunner(
            ingest_probe=probe,
            workload_factory=_stub_workload,
            corpus_sizes=[100, 1000],
        )

        report = runner.run()

        # Both scales recorded their ingest_seconds; largest is the
        # one quoted in the positioning doc.
        for s in report.scales:
            assert s.ingest_seconds == pytest.approx(0.5, abs=0.01)
        assert probe.calls == [100, 1000]

    def test_warmup_queries_run_but_not_counted(self):
        # The workload factory is called twice per scale (warmup +
        # measured); only the measured latencies feed the percentiles.
        # Captured via probing the call counter on the workload.
        seen: list[tuple[int, int]] = []

        def factory(corpus_size: int, *, phase: str = "measured") -> StubQueryWorkload:
            n = WARMUP_QUERIES if phase == "warmup" else MEASURED_QUERIES
            seen.append((corpus_size, n))
            return StubQueryWorkload(
                num_queries=n,
                recall_hits=int(n * 0.9),
                simulated_latency_ms=1.0,
            )

        runner = LatencyBenchRunner(
            ingest_probe=_FakeIngestProbe(),
            workload_factory=factory,
            corpus_sizes=[100],
        )

        report = runner.run()

        # Two calls per scale: one warmup, one measured.
        assert seen == [(100, WARMUP_QUERIES), (100, MEASURED_QUERIES)]
        # Only the measured latencies populate the percentiles
        # (otherwise len(latencies) would be 10+100=110).
        assert report.scales[0].measured_query_count == MEASURED_QUERIES


# ─── Repository: LatencyReport JSON round-trip ────────────────────────────


class TestLatencyReportSerialization:
    def test_to_json_round_trip(self, tmp_path: Path):
        runner = LatencyBenchRunner(
            ingest_probe=_FakeIngestProbe(),
            workload_factory=_stub_workload,
            corpus_sizes=[100],
        )
        report = runner.run()

        out = tmp_path / "report.json"
        report.write_json(out)

        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert "scales" in loaded
        assert loaded["scales"][0]["corpus_size"] == 100
        assert "latency_percentiles_ms" in loaded["scales"][0]
        assert "ingest_seconds" in loaded["scales"][0]
