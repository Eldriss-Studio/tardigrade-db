#!/usr/bin/env python3
"""Shared-LLM-path latency benchmark (M1.4).

Quantifies TardigradeDB's architectural differentiator: when an LLM
is already running in-process and produces a query KV as a natural
side effect of its forward pass, retrieval is free — just a dot
product over the in-memory index. Compare three retrieval paths
end-to-end at three corpus sizes:

* **Cold path** — fresh model forward pass to encode the query,
  then engine retrieval. The shape of what the current bench
  measures: encode-from-scratch + retrieve. The cost is dominated
  by the model forward pass.

* **Warm path** — the LLM is already running and just produced a
  prefill. Reuse that hidden-state slice as the query key. Cost is
  retrieval only — the model spend is amortised over the LLM's
  primary job.

* **Text RAG baseline** — what a typical RAG pipeline pays: an
  external embedding-API call (network + remote model) plus a
  vector-DB lookup. Stand-in cost numbers cited inline so the
  comparison is auditable.

The script is pure-synthetic: it does not load a real LLM. Model
and network costs are simulated with documented sleep values
sourced from public benchmarks (Qwen3-0.6B prefill latencies,
OpenAI embedding API p50). The *shape* of the comparison and the
relative ordering are what matter; absolute numbers should be
re-measured with real models when integrating with a specific
consumer's stack.

Output: ``target/shared-llm-latency.json`` plus a Markdown table
appended to ``docs/positioning/latency_first.md``.
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

# Simulated model-prefill latency, milliseconds. From the public
# `Qwen3-0.6B` benchmark page for a short query (~20 tokens) on
# CPU. GPU would be ~5-10× faster; the foot-print of the win is
# unchanged. Document the assumption explicitly in the JSON output
# so reviewers can recompute with their own numbers.
SIMULATED_COLD_PREFILL_MS: float = 50.0

# Simulated text-RAG embedding-API round-trip, milliseconds. OpenAI
# `text-embedding-3-small` p50 over a fast network is ~50ms. Local
# embedding (BGE-base on CPU) is ~30-100ms. Pick the optimistic
# end so the comparison doesn't appear to cherry-pick.
SIMULATED_RAG_EMBED_MS: float = 50.0

# Simulated vector-DB query latency, milliseconds. Pinecone p50 for
# 10K vectors at 768d is ~5ms. We use 5ms as a low-end estimate so
# the comparison is fair to text-RAG.
SIMULATED_RAG_VECTORDB_MS: float = 5.0

# Output path for the JSON report.
DEFAULT_OUTPUT_PATH: str = "target/shared-llm-latency.json"

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
    name: str
    corpus_size: int
    latency: PercentileLatency
    mean_ms: float
    stdev_ms: float


@dataclass
class Report:
    corpus_sizes: list[int]
    hidden_dim: int
    simulated_cold_prefill_ms: float
    simulated_rag_embed_ms: float
    simulated_rag_vectordb_ms: float
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


# ─── Engine + workloads ─────────────────────────────────────────────────


def _build_engine(
    corpus_size: int, hidden_dim: int,
) -> tuple[object, list[np.ndarray], tempfile.TemporaryDirectory]:
    """Populate a fresh engine with ``corpus_size`` synthetic packs.

    Returns ``(engine, ingested_keys, tmpdir)``. The caller must
    keep ``tmpdir`` alive for the engine's lifetime — the PyO3
    ``Engine`` class does not accept extra Python attributes, so we
    return the keep-alive explicitly.
    """
    tmpdir = tempfile.TemporaryDirectory(prefix="tdb_shared_llm_")
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


def _sleep_ms(ms: float) -> None:
    """Busy-wait the requested milliseconds. We use a busy loop
    rather than ``time.sleep`` so the simulated latency is accurate
    even when the OS scheduler quantum is large."""
    target = time.perf_counter() + (ms / 1000.0)
    while time.perf_counter() < target:
        pass


def _run_cold(engine, keys: list[np.ndarray]) -> list[float]:
    """Cold path: simulate full model prefill, then retrieve."""
    rng = np.random.default_rng(seed=0)
    samples: list[float] = []
    # Pre-pick query indices so timing is consistent across paths.
    indices = [int(rng.integers(0, len(keys))) for _ in range(WARMUP_QUERIES + MEASURED_QUERIES)]
    for i, idx in enumerate(indices):
        t0 = time.perf_counter()
        _sleep_ms(SIMULATED_COLD_PREFILL_MS)  # simulated prefill
        engine.mem_read(query_key=keys[idx], k=5, owner=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= WARMUP_QUERIES:
            samples.append(elapsed_ms)
    return samples


def _run_warm(engine, keys: list[np.ndarray]) -> list[float]:
    """Warm path: query KV reused — measure retrieval only."""
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


def _run_rag_baseline(engine, keys: list[np.ndarray]) -> list[float]:
    """Text-RAG baseline: simulated embedding API + simulated
    vector-DB lookup. Uses the engine only as a stand-in for the
    final vector index (it's a similar cost). The point of the
    measurement is the embedding-API overhead a text-RAG consumer
    cannot avoid."""
    rng = np.random.default_rng(seed=0)
    samples: list[float] = []
    indices = [int(rng.integers(0, len(keys))) for _ in range(WARMUP_QUERIES + MEASURED_QUERIES)]
    for i, idx in enumerate(indices):
        t0 = time.perf_counter()
        _sleep_ms(SIMULATED_RAG_EMBED_MS)
        _sleep_ms(SIMULATED_RAG_VECTORDB_MS)
        # The engine call here is a stand-in for the vector-DB
        # query that a real RAG pipeline would issue; pinned to
        # the engine for apples-to-apples on the same hardware.
        engine.mem_read(query_key=keys[idx], k=5, owner=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= WARMUP_QUERIES:
            samples.append(elapsed_ms)
    return samples


# ─── Orchestration ──────────────────────────────────────────────────────


def run_benchmark(
    corpus_sizes: list[int],
    hidden_dim: int,
) -> Report:
    scenarios: list[ScenarioResult] = []
    for size in corpus_sizes:
        engine, keys, _tmp = _build_engine(size, hidden_dim)
        for name, runner in (
            ("cold (encode + retrieve)", _run_cold),
            ("warm (shared-LLM KV reuse)", _run_warm),
            ("text-RAG baseline (embed + vector-DB)", _run_rag_baseline),
        ):
            samples = runner(engine, keys)
            scenarios.append(
                ScenarioResult(
                    name=name,
                    corpus_size=size,
                    latency=PercentileLatency.from_samples(samples),
                    mean_ms=statistics.fmean(samples) if samples else 0.0,
                    stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
                ),
            )
        # Keep tmpdir alive until all scenarios for this corpus
        # size finish; drop afterwards so disk doesn't fill up
        # across many sweeps.
        del _tmp
    return Report(
        corpus_sizes=list(corpus_sizes),
        hidden_dim=hidden_dim,
        simulated_cold_prefill_ms=SIMULATED_COLD_PREFILL_MS,
        simulated_rag_embed_ms=SIMULATED_RAG_EMBED_MS,
        simulated_rag_vectordb_ms=SIMULATED_RAG_VECTORDB_MS,
        scenarios=scenarios,
        notes=(
            "Pure-synthetic benchmark. Model prefill and embedding-API "
            "costs are simulated with busy-waits using published p50 "
            "values for Qwen3-0.6B (CPU prefill ≈ 50ms) and OpenAI "
            "text-embedding-3-small (~50ms). Re-measure with real "
            "models when integrating with a specific consumer."
        ),
    )


def _render_markdown(report: Report) -> str:
    lines: list[str] = [
        "",
        "### Shared-LLM-path retrieval latency (M1.4)",
        "",
        "Three retrieval paths compared at three corpus sizes. The",
        "warm path measures TardigradeDB's architectural payoff —",
        "when the LLM is already running, retrieval is a dot product",
        "with no extra model spend. The cold path measures the cost",
        "of encoding a query from scratch (current bench shape). The",
        "RAG baseline simulates a typical text-RAG pipeline (embedding",
        "API + vector-DB lookup) using published p50 latencies.",
        "",
        f"Notes: {report.notes}",
        "",
        "| Corpus | Path | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    for s in report.scenarios:
        lines.append(
            f"| {s.corpus_size} | {s.name} | "
            f"{s.latency.p50_ms:.2f} | {s.latency.p95_ms:.2f} | {s.latency.p99_ms:.2f} |",
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
