# Ingestion Performance Roadmap

Rough-draft designs for the six biggest unrealized speedups in the
benchmark ingestion path. Surfaced when the full LoCoMo run took ~3 hr
in agent mode on RTX 3070 Ti / Qwen3-0.6B; profiling pointed to
GPU forward passes (~60% of wall time) and per-chunk fsync (~30%) as
the dominant costs. Numbers below are projections, not measurements;
each item closes with the acceptance test that turns the projection
into a measurement.

Order is by bang-for-buck. Tier 1 lands order-of-magnitude wins; Tier 2
is the next 30-40%; Tier 3 is trade-off territory. None of these are in
scope for the current OOM fix — this doc is the follow-up plan.

## Prerequisite — retrieval correctness regression (2026-05-13)

The OOM fix (lazy `PerTokenRetriever`) introduced a **recall regression**
because it removed the in-RAM INT8-quantized token tier that was load-
bearing for scoring, leaving only Q4-quantized tokens on disk. Q4 group-
of-32 quantization collapses non-outlier values to zero in the presence
of activation-outlier channels — a known failure mode (Dettmers 2022,
"LLM.int8()"). LoCoMo recall dropped from 68% to 3.3%.

**The performance items in this roadmap are blocked on the recall fix.**
There is no point optimizing throughput on a path that produces wrong
answers. The retrieval-storage redesign (per-channel scaling + INT8
group quantization, SmoothQuant-for-storage pattern) takes precedence;
see `docs/refs/external-references.md` §A3f for the full reference body
that establishes the correct design.

---

## Tier 1 — order-of-magnitude wins

### 1. GPU batching for the ingest forward pass

**Problem.** `python/tdb_bench/adapters/tardigrade.py:305-316` loops one
chunk per `tokenizer(...) → model(...)` call. An RTX 3070 Ti can run
8-16 chunks of 128 tokens through Qwen3-0.6B in a single forward pass
with comfortable VRAM headroom, but the adapter never asks it to. The
GPU sits at 40-50% utilization through the bench instead of saturating.

**Proposed change.** Collect `BATCH_SIZE` chunks per item, tokenize and
right-pad to a common length, run one batched forward pass, slice the
output `hidden_states` back into per-chunk segments. The hook's
`on_generate` is per-chunk; either call it `BATCH_SIZE` times in a row
over the sliced outputs, or generalize the hook to accept a batched
hidden-states tensor and emit `BATCH_SIZE` `WriteDecision`s.

**Constants.**
- `TDB_BENCH_GPU_BATCH_SIZE` env var, default `8`. Named env var, not a
  literal in code.
- Attention mask handled honestly so padded positions don't pollute the
  per-token K projection.

**Expected gain.** GPU forward pass is ~60% of per-item wall time.
Batching to 8 yields a theoretical 5-8× speedup on that portion → net
**~3-4× faster ingest end-to-end**. From ~3 hr to ~45-60 min on the
full LoCoMo set.

**Risk.** Padding correctness. If the padding mask leaks into the
hidden-states slice handed to the hook, the encoded per-token key will
include padded positions and ranking will degrade. The acceptance test
below pins this.

**Acceptance test (AT).**
*Given* a single item with N chunks,
*when* the adapter ingests it with `GPU_BATCH_SIZE = 1` and again with
`GPU_BATCH_SIZE = 8`,
*then* the resulting top-K result IDs for any query match across the
two runs (modulo Q4 noise, ≥4-of-5 ID overlap), and the batched run's
wall time is ≤ 0.3× the serial run.

**Effort.** ~4 hours (refactor + AT + validate on smoke fixture).

---

### 2. Engine-side write batching via `mem_write_batch`

**Problem.** The adapter calls `engine.mem_write` once per chunk
(`tardigrade.py:321`). Each call performs its own fsync. The engine
already exposes `mem_write_batch`
(`crates/tdb-engine/src/engine.rs:587`) documented at *"~80μs/cell
amortized vs ~8ms/cell for individual writes"* — a 100× per-cell
amortization that the adapter simply doesn't use.

**Proposed change.** Per item, collect `WriteRequest`s for all chunks
into a `Vec`, call `engine.mem_write_batch(requests)` once. This is a
~10-line refactor in the adapter; no engine change.

**Constants.** None — the batch size is "all chunks of the current
item." If items are large the engine handles segmentation internally
via the same fsync at the end.

**Expected gain.** Engine writes are ~30% of per-item time. Batching
gives 100× amortization on the fsync portion → effectively eliminates
this share. **~25-30% faster** ingest end-to-end. Stacks with Tier-1
item 1.

**Risk.** None significant. The batch path is already covered by the
engine's existing tests (`mem_write_batch` has matching `mem_write`
parity tests in tdb-engine).

**AT.** Equivalence test: ingest the same fixture via `mem_write`
serially and via `mem_write_batch`, assert the resulting engine state
(cell count, pack count, sample query top-K) is identical. Effectively
covered by existing engine tests — this AT just pins the *adapter's*
correct use of the batch API.

**Effort.** ~2 hours.

---

## Tier 2 — useful but smaller

### 3. CPU/GPU pipeline overlap

**Problem.** Today's loop runs strictly:
`tokenize → forward → on_generate → encode → mem_write → next chunk`.
The GPU is idle while the CPU encodes and writes; the CPU is idle while
the GPU computes. Two distinct hardware resources, sequential
utilization.

**Proposed change.** Producer-consumer pattern with two threads (or a
PyTorch async stream + a write queue):
- **Producer thread** runs forward passes back-to-back, posts
  `(hidden_states_cpu_tensor, item_id, chunk_index)` onto a bounded
  queue.
- **Consumer thread** drains the queue, runs `on_generate`, encodes,
  calls `mem_write_batch` per item.

Bounded queue (e.g., `QUEUE_DEPTH = 4` items ahead) caps memory growth.

**Expected gain.** Best case: max(GPU time, CPU time) instead of GPU +
CPU. With Tier-1 changes applied, both portions are smaller, so the
overlap win shrinks. **~15-25%** on top of Tier-1.

**Risk.** Threading + PyTorch interactions. Need to be careful that
producer's torch tensors are moved to CPU before being handed off (no
GPU references crossing thread boundaries), and that GIL release inside
the engine (already implemented per CLAUDE.md *"Python Engine wrapped
in Arc<Mutex<>> with GIL release via py.detach()"*) actually permits
the consumer to run in parallel with the producer's GPU launch.

**AT.** End-to-end ingest time on the LoCoMo smoke fixture with overlap
on vs off — assert overlap-on is ≤ 0.85× overlap-off.

**Effort.** ~1 day. Threading + careful tensor handoff. Worth it only
after Tier-1 is in.

---

### 4. Async durability — decouple fsync from the hot loop

**Problem.** Even with `mem_write_batch`, each batch's fsync is on the
hot path. The Aeon Architecture's `durable_offset` semantic lets us
acknowledge a write at the WAL level before the segment's fsync
completes — production transaction engines (Postgres group commit,
RocksDB pipelined writes) do exactly this.

**Proposed change.** A background fsync worker thread that pulls
`(segment_id, byte_offset)` ranges off a queue and calls `fdatasync`.
Writes acknowledged via WAL append in the foreground; `durable_offset`
advances only when the worker confirms. This is a **durability
contract change** (per CLAUDE.md Reliability Rules) and must declare
`unconfirmed` semantics for reads against post-WAL/pre-fsync data.

**Expected gain.** With Tier-1 batch writes already in place, fsync is
once per item rather than once per chunk. Background fsync turns that
~10ms wait per item into ~0ms — **maybe 5-10%** in this benchmark, but
**much larger** for write-heavy production workloads.

**Risk.** This is the **largest scope of any item in this doc.** The
durability contract change requires:
- New WAL replay tests asserting confirmed-vs-unconfirmed semantics.
- Crash tests at the WAL/fsync boundary.
- API exposure of confirmed-read vs latest-read distinction.

Do not roll this into a bench-speed PR. It's a real engine feature.

**AT (durability-class).** Per CLAUDE.md mandatory tests for
durability changes: crash during append, crash between WAL commit and
snapshot, snapshot restore + WAL suffix replay correctness, confirmed
read visibility (`durable_offset >= tx_offset`).

**Effort.** 2-3 days minimum, plus careful review. Defer until the
write-throughput conversation comes up for a real reason.

---

## Tier 3 — trade-offs and marginal

### 5. Smaller capture model (quality/speed dial)

**Problem.** Qwen3-0.6B was chosen for representation quality.
Per-chunk forward pass dominates wall time. Smaller models exist that
produce coarser but faster representations.

**Proposed change.** New env var `TDB_BENCH_CAPTURE_MODEL` with named
tiers:
- `"tiny"` → DistilGPT2 / 82M params (~3× faster, much coarser)
- `"small"` → GPT-2 small / 124M (~4× faster, coarse)
- `"default"` → Qwen3-0.6B (current)
- `"large"` → Qwen3-1.7B (3× slower, +recall on cross-family
  baselines per existing experiments)

Defaults to `"default"`. Smoke + agent runs stay on default. CI smoke
gate could optionally use `"tiny"` for speed.

**Expected gain.** Speed scales roughly linearly with parameter count.
**~3-4× faster on `"tiny"`**, at unmeasured quality cost — published
TardigradeDB recall numbers are on Qwen3-0.6B, smaller models would
need their own baseline.

**Risk.** This is **not a free win** — it's a quality knob. Documenting
it as a knob is fine; presenting it as a perf optimization is not.

**AT.** Run smoke fixture on each model tier; record retrieval recall
+ wall time. Persist as `docs/bench/model-tier-comparison.md` so future
debates about model choice have data to cite.

**Effort.** ~1 hour for the knob, ~2 hours for the comparison study.

---

### 6. Fused Q4 quantize + write kernels

**Problem.** Q4 quantization and segment append are sequential — first
copy the f32 buffer into Q4-packed bytes, then write the bytes to the
segment. Each step iterates the same data once.

**Proposed change.** Single pass that quantizes directly into the
segment-file write buffer. Saves one full traversal of ~520 KB per
cell.

**Expected gain.** With Tier-1 batch writes, this is marginal.
**Single-digit %** in the engine portion → low single-digit overall.

**Risk.** Minimal. The on-disk Q4 format is unchanged.

**AT.** Engine throughput benchmark before/after, on the same fixture
sizes (`cargo bench -p tdb-engine`).

**Effort.** ~4 hours. **Defer until profiling identifies it as the
remaining bottleneck.** Premature otherwise.

---

## Recommended execution order

1. **Item 1 (GPU batching)** — biggest single win. Ship first.
2. **Item 2 (`mem_write_batch`)** — trivial change, stacks cleanly.
3. **Re-measure end-to-end** after #1 + #2. Most of the original
   "ingestion is slow" complaint should be gone.
4. **Item 3 (CPU/GPU overlap)** — only if #1+#2 still leave a
   noticeable gap.
5. **Item 5 (model tier knob)** — independent of #1-3; ship the knob,
   defer the comparison study until there's a question to answer.
6. **Item 4 (async durability)** — separate workstream; not a bench
   concern.
7. **Item 6 (fused kernels)** — only if profiling demands it.

## Out of scope (call out, don't do)

- **Distributed ingestion across machines.** Single-node is fine for
  the corpora we benchmark.
- **Custom CUDA kernels for the model.** PyTorch's are good enough; the
  win is in *how* we call them (batching), not *what* they do.
- **Replacing the model architecture** (e.g. swapping to BGE-style
  retrieval encoders). Different project; would invalidate every
  existing benchmark number on record.

## Verification plan

Each Tier-1/Tier-2 item ships with its named AT above. The roll-up
metric is **end-to-end full LoCoMo + LongMemEval ingest wall time** on
the operator's hardware (RTX 3070 Ti / 24 GB WSL2), measured before
and after each stacked change:

| State | Expected wall time |
|---|---|
| Baseline (this PR) | ~3 hr ingest |
| + #1 (GPU batching) | ~45-60 min |
| + #2 (batch writes) | ~35-45 min |
| + #3 (CPU/GPU overlap) | ~30 min |

Record actual numbers in this doc as each item lands. Projections that
don't materialize are bugs in the projection, not the measurement —
update accordingly.
