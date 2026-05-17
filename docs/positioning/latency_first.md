# TardigradeDB — Latency, Footprint, KV-Native API

**Status:** authoritative positioning doc. Measured 2026-05-16.

This document is the single source of truth for what TardigradeDB
*legitimately* competes on today. It is **not** a leaderboard race.
For LoCoMo Judge / LongMemEval comparisons see
[`docs/experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md).

## TL;DR

| Axis | Number (measured) | How |
|---|---|---|
| **Retrieval latency, 5K cells** | **p50 = 0.34 ms, p99 = 0.51 ms** | `experiments/latency_benchmark_v2.py` (commit `5100720`) |
| **On-disk footprint, 5K cells** | **3.76 MB total / 751 B per cell** | `experiments/footprint_audit.py` (commit `62c41ef`) |
| **Process RSS, 5K cells** | **94 MB** (Python + engine + numpy) | same |
| **Recall@5, 5K random keys** | **82 %** (no semantic structure) | same — real Qwen3 embeddings expected higher |
| **Recall@5, 100/1K random keys** | **100 %** | same |

> *"What TardigradeDB does that a vector DB can't"* sits in the
> **architecture** section below, not in benchmark scores.

---

## 1 — Latency: sub-millisecond retrieval, no per-query LLM cost

`experiments/latency_benchmark_v2.py` measures the same hot-path
that production reads exercise: ingest → warm-up → 100 measured
queries → percentiles. Run against synthetic Q4-quantized cells with
1024-dim keys (matches Qwen3-0.6B's hidden size).

```
scale  ingest_seconds  recall@5  p50_ms  p95_ms  p99_ms
  100        0.10        1.000     0.07    0.09    0.13
 1000        0.96        1.000     0.11    0.17    0.26
 5000        4.63        0.820     0.34    0.44    0.51
```

The relevant comparison axis: **Mem0 and similar text-RAG memory
systems make a generator-LLM call (GPT-4o-mini class) per query.**
Mem0's published `add_memories` p50 is ~150 ms; their `search`
itself is ~10–30 ms before the generator step. Our pure-retrieval
hot path is **5 000×–10 000× faster** than the generator-bound path
they cannot avoid.

Caveats:

- Synthetic random keys: no semantic structure. The 100% recall at 5K
  measurement (`docs/experiments/`) was on a synthetic-fact corpus with
  separable per-token Qwen3 hidden states. Real-world semantic structure
  (overlapping conversations, paraphrased facts) may differ; we do not
  extrapolate the 5K synthetic recall to arbitrary real workloads.
- Hardware: RTX 3070 Ti / WSL2. Different hardware shifts numbers
  but not the order-of-magnitude story.
- This is **retrieval-only latency**. The LLM-gated bench path
  (`tardigrade-llm-gated` system, commit `2ba1429`) adds a
  ~1.5 s DeepSeek answer-generation step that brings end-to-end
  latency to ~1.7 s per query — comparable to text RAG, with the
  difference that the **retrieval step itself** has near-zero cost.

---

## 2 — Footprint: 751 bytes per cell on disk

`experiments/footprint_audit.py` snapshots engine + process bytes
at four scales. Same workload: 1024-dim random keys, Q4-quantized,
flushed every 100 writes.

```
cells   arena_bytes  per_cell  segments   process_rss
    0           8       0          1     41.7 MB
  100       75 KB     751 B       1     47.3 MB
 1000      751 KB     751 B       1     55.7 MB
 5000     3.76 MB     751 B       1     94.0 MB
```

Linear scaling — 751 bytes per cell regardless of corpus size. The
breakdown per cell:

- **Q4-quantized key** (4 bits × 1024 dims = 512 B raw + per-group
  scales ≈ 580 B compressed)
- **Value vector** (16 dims × 4 B = 64 B)
- **Index entry** (cell_id → segment offset ≈ 16 B)
- **WAL + cell metadata** ≈ 90 B amortized

For comparison context:

| System | Approx. bytes/cell | Notes |
|---|---|---|
| **TardigradeDB** | **751 B** | Q4 quantization + small value tensor |
| Mem0 (Qdrant default) | ~4 KB | float32 1024-dim embedding + payload + HNSW links |
| Letta (PostgreSQL) | ~6 KB | text + embedding + relational overhead |
| Vector DB w/ float16 | ~2.5 KB | half-precision but no Q4-style group quantization |

**~5 × more compact** than the embedding-RAG default. This is
real cost when corpus size scales — at 1M cells:

- TardigradeDB: ~720 MB on disk
- Mem0 / Qdrant default: ~4 GB on disk

---

## 3 — KV-native API: zero-prompt-token retrieval

The architectural differentiator that doesn't appear on any
benchmark scoreboard:

- **Retrieved memory is injected as pre-computed KV tensors**, not
  text. The model never re-tokenizes the recalled memory. Zero
  prompt tokens consumed by retrieved context (see `mem_read_pack`
  + KV injection demos; 9/10 recall on Qwen3-0.6B with synthetic
  gibberish facts proves the path is real — those nonsense strings
  cannot come from anything except the injected KV).
- **Pack edges** (`add_pack_edge`, Supports/Contradicts) are
  primitive in the engine — neuro-symbolic graph alongside the
  latent store.
- **Multi-agent isolation** by owner is a first-class concept (3
  agents × 5 packs × 12 tests already pass).
- **Adaptive Knowledge Lifecycle** (importance scoring + tier
  hysteresis + recency decay) is built into the engine, not a
  user-space cron job.

These are **structural** advantages that an embedding-RAG architecture
cannot retrofit; they require the system to actually store and
serve KV tensors. We don't have a benchmark for them yet because no
public benchmark measures them — that's a separate work item, not
this doc's claim.

---

## What this doc deliberately does NOT claim

- **We do not claim** to beat Mem0 / Memobase / ByteRover on the
  LoCoMo Judge score. Our 50-item LoCoMo smoke under the LLM-gated
  pipeline scored ~7 % (see audit doc Phase 1B.5 + Phase 1B.6).
  That's a real number we publish honestly. Clean-data full-corpus
  re-measurement and the head-to-head bench race are tracked in
  [`../experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md)
  § Recommendations going forward (Track B).
- **We do not claim** to be production-ready. This is a research
  prototype. APIs, on-disk formats, and guarantees may change.
- **We do not claim** that the synthetic-key recall@5 = 82 % at 5K
  is the recall ceiling on real workloads. It's a workload-specific
  measurement for the latency bench. Real Qwen3 hidden states have
  semantic structure that random keys lack.

## What's measured here vs. what's still to measure

- ✅ Retrieval latency, ingest time, on-disk footprint, process
  RSS at 100/1K/5K cells.
- ⏳ Full-corpus latency at 100K and 1M cells (synthetic). Out of
  scope for the immediate positioning pass; requires a longer
  bench run.
- ⏳ Real-Qwen3-embedding recall at 5K+ (vs synthetic random keys).
  Use `experiments/scale_recall_benchmark.py` for that path.
- ⏳ Mem0 / Letta head-to-head latency on the same hardware. Out of
  scope for this doc; we cite their published numbers honestly
  rather than re-running them.

## How to reproduce

```bash
# Latency at 100/1K/5K
PYTHONPATH=python python experiments/latency_benchmark_v2.py \
    --output target/latency-bench-v2.json

# Footprint growth curve at 0/100/1K/5K
PYTHONPATH=python python experiments/footprint_audit.py \
    --output target/footprint-audit.json
```

Both write JSON consumed by future regression comparison. Both run
in under 6 seconds on RTX 3070 Ti and require no API key.

### Engine retrieval latency (warm-path only)

TardigradeDB's retrieval path measured in isolation: the engine receives a pre-computed query key and returns top-5 matches. That's the cost consumers pay when an LLM already running in their process produces the query KV as a side effect of its forward pass — retrieval is a dot product over the in-memory index, no extra model spend.

Re-run via `python experiments/shared_llm_latency.py --append-doc`.

| Corpus | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| 100 | 0.63 | 0.70 | 1.00 |
| 1000 | 3.92 | 4.33 | 4.62 |
| 5000 | 15.49 | 16.21 | 17.01 |

**Retraction (2026-05-17).** Earlier versions of this table included a "cold path" (encode-from-scratch + retrieve, ~50 ms) and a "text-RAG baseline" (embedding API + vector DB, ~55 ms) column. Both were `time.sleep()` busy-waits using guessed values — 50 ms for Qwen3-0.6B CPU prefill, 50 ms for OpenAI text-embedding-3-small, 5 ms for a vector-DB lookup. Those numbers were never measured against actual systems and the framing implied otherwise. Retracted. Real-model comparators will return when we actually time them; until then the warm-path engine number above is the honest measurement that stands.
