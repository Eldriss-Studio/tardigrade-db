#!/usr/bin/env python3
"""Engine retrieval latency at fixed corpus sizes.

Measures TardigradeDB's retrieval path in isolation: the engine receives a pre-computed query key and returns top-k packs. That's the path consumers exercise when the LLM running in their process already produced the query KV as a side effect of its forward pass — retrieval becomes a dot product over the in-memory index, with no extra model spend.

This is **just the engine measurement.** Earlier versions of the script also reported a "cold" path (encode-from-scratch) and a "text-RAG baseline" (embedding API + vector DB), both simulated with `time.sleep()` calls using guessed values for Qwen3-0.6B prefill and OpenAI embedding latencies. Those simulated columns were retracted on 2026-05-17 — the warm-path engine number is the honest measurement; comparators can come back when we actually time real models in a follow-up experiment.

Output: `target/engine-retrieval-latency.json` plus a Markdown table appended to `docs/positioning/latency_first.md` when `--append-doc` is passed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "python"))

import numpy as np  # noqa: E402

import tardigrade_db  # noqa: E402

# ─── Configurable constants (no magic values) ───────────────────────────

# Corpus sizes to sweep. Matches the existing latency-bench-v2
# scales for direct comparison.
DEFAULT_CORPUS_SIZES: tuple[int, ...] = (100, 1_000, 5_000)

# Hidden-state dimensionality. Defaults to Qwen3-0.6B's hidden size;
# override via CLI for other capture models.
DEFAULT_HIDDEN_DIM: int = 1024

# Number of measured queries per scenario. Matches v2 bench harness.
MEASURED_QUERIES: int = 100

# Warm-up queries discarded before measurement. Lets the SLB and OS
# page cache warm up.
WARMUP_QUERIES: int = 10

# Percentiles reported. Mirrors v2 harness convention.
LATENCY_PERCENTILES: tuple[int, ...] = (50, 95, 99)

# Output path for the JSON report.
DEFAULT_OUTPUT_PATH: str = "target/engine-retrieval-latency.json"

# Markdown table append target (positioning doc).
DEFAULT_DOC_PATH: str = "docs/positioning/latency_first.md"


# ─── Result types ───────────────────────────────────────────────────────


@dataclass
class PercentileLatency:
    """p50/p95/p99 plus sample count, all in milliseconds."""

    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> "PercentileLatency":
        if not samples_ms:
            return cls(n=0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0)
        sorted_samples = sorted(samples_ms)
        return cls(
            n=len(sorted_samples),
            p50_ms=_percentile(sorted_samples, 50),
            p95_ms=_percentile(sorted_samples, 95),
            p99_ms=_percentile(sorted_samples, 99),
        )


def _percentile(sorted_samples: list[float], pct: int) -> float:
    if not sorted_samples:
        return 0.0
    idx = max(0, min(len(sorted_samples) - 1, math.ceil(pct / 100.0 * len(sorted_samples)) - 1))
    return sorted_samples[idx]


@dataclass
class ScenarioResult:
    corpus_size: int
    latency: PercentileLatency
    mean_ms: float
    stdev_ms: float


@dataclass
class Report:
    corpus_sizes: list[int]
    hidden_dim: int
    scenarios: list[ScenarioResult]
    notes: str

    def to_dict(self) -> dict:
        out = asdict(self)
        # Flatten the nested PercentileLatency for JSON convenience.
        out["scenarios"] = [
            {
                **{k: v for k, v in asdict(s).items() if k != "latency"},
                "p50_ms": s.latency.p50_ms,
                "p95_ms": s.latency.p95_ms,
                "p99_ms": s.latency.p99_ms,
                "n": s.latency.n,
            }
            for s in self.scenarios
        ]
        return out


# ─── Engine + workload ──────────────────────────────────────────────────


def _build_engine(
    corpus_size: int, hidden_dim: int,
) -> tuple[object, list[np.ndarray], tempfile.TemporaryDirectory]:
    """Populate a fresh engine with ``corpus_size`` synthetic packs.

    Returns ``(engine, ingested_keys, tmpdir)``. The caller must keep ``tmpdir`` alive for the engine's lifetime — the PyO3 ``Engine`` class does not accept extra Python attributes, so we return the keep-alive explicitly.
    """
    tmpdir = tempfile.TemporaryDirectory(prefix="tdb_retrieval_latency_")
    engine = tardigrade_db.Engine(tmpdir.name)
    rng = np.random.default_rng(seed=42)
    keys: list[np.ndarray] = []
    for _ in range(corpus_size):
        k = rng.standard_normal(hidden_dim).astype(np.float32)
        v = rng.standard_normal(16).astype(np.float32)
        keys.append(k)
        engine.mem_write(
            owner=1, layer=0, key=k, value=v, salience=1.0, parent_cell_id=None,
        )
    engine.flush()
    return engine, keys, tmpdir


def _measure_retrieval(engine, keys: list[np.ndarray]) -> list[float]:
    """Time ``MEASURED_QUERIES`` retrievals after a warm-up. Returns per-query latencies in milliseconds."""
    rng = np.random.default_rng(seed=0)
    samples: list[float] = []
    indices = [int(rng.integers(0, len(keys))) for _ in range(WARMUP_QUERIES + MEASURED_QUERIES)]
    for i, idx in enumerate(indices):
        t0 = time.perf_counter()
        engine.mem_read(query_key=keys[idx], k=5, owner=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= WARMUP_QUERIES:
            samples.append(elapsed_ms)
    return samples


# ─── Orchestration ──────────────────────────────────────────────────────


def run_benchmark(corpus_sizes: list[int], hidden_dim: int) -> Report:
    scenarios: list[ScenarioResult] = []
    for size in corpus_sizes:
        engine, keys, _tmp = _build_engine(size, hidden_dim)
        samples = _measure_retrieval(engine, keys)
        scenarios.append(
            ScenarioResult(
                corpus_size=size,
                latency=PercentileLatency.from_samples(samples),
                mean_ms=statistics.fmean(samples) if samples else 0.0,
                stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
            ),
        )
        # Keep tmpdir alive until this scenario's measurement
        # finishes; drop afterwards so disk doesn't fill up across
        # many sweeps.
        del _tmp
    return Report(
        corpus_sizes=list(corpus_sizes),
        hidden_dim=hidden_dim,
        scenarios=scenarios,
        notes=(
            "Engine retrieval only — no LLM forward pass, no embedding-API call, "
            "no network. The query key is pre-computed; the engine returns top-5 "
            "matches. This is the retrieval cost a consumer pays when the LLM "
            "running in their process already produced the query KV. Cold-path "
            "and text-RAG comparators that previously appeared here were "
            "simulated with guessed sleep values and have been retracted; "
            "real-model comparators will land in a follow-up."
        ),
    )


def _render_markdown(report: Report) -> str:
    lines: list[str] = [
        "",
        "### Engine retrieval latency (warm-path only)",
        "",
        "TardigradeDB's retrieval path measured in isolation: the engine receives a pre-computed query key and returns top-5 matches. That's the cost consumers pay when an LLM already running in their process produces the query KV as a side effect of its forward pass — retrieval is a dot product over the in-memory index, no extra model spend.",
        "",
        f"Notes: {report.notes}",
        "",
        "| Corpus | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | stdev (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in report.scenarios:
        lines.append(
            f"| {s.corpus_size} | "
            f"{s.latency.p50_ms:.2f} | {s.latency.p95_ms:.2f} | {s.latency.p99_ms:.2f} | "
            f"{s.mean_ms:.2f} | {s.stdev_ms:.2f} |",
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CORPUS_SIZES),
        help="Corpus sizes to sweep (default: 100 1000 5000).",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help=f"Hidden-state dimensionality (default: {DEFAULT_HIDDEN_DIM}).",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--append-doc",
        action="store_true",
        help=f"Append the Markdown table to {DEFAULT_DOC_PATH}.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_benchmark(args.corpus_sizes, args.hidden_dim)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    # `print` is OK here — this is a CLI script.
    print(f"wrote {out_path}")
    print(_render_markdown(report))
    if args.append_doc:
        doc_path = Path(DEFAULT_DOC_PATH)
        with doc_path.open("a", encoding="utf-8") as f:
            f.write(_render_markdown(report))
        print(f"appended to {doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
