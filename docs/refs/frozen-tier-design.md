# Frozen tier — design archive

**Status:** designed, deferred. Not scheduled — captured here so the rationale lives in-repo from before the code does. When the work is picked up, the implementation plan lives at `~/.claude/plans/is-there-anything-dreamy-pike.md`.

**Originating research:** [`docs/research/2026-05-21-production-kv-cache-learnings.md`](../research/2026-05-21-production-kv-cache-learnings.md).

---

## The two-axis problem

TardigradeDB's `Draft / Validated / Core` tiers encode a single property: **confidence** — how sure are we this memory matters? Governance moves cells through that state machine based on importance score and access patterns with hysteresis.

That collapses two distinct properties of a memory into one variable. Real memory has a second, orthogonal property: **temperature** — how cold is this in retrieval terms? A high-confidence Core memory that hasn't been accessed in two years is still high-confidence; it's just cold. The current state machine has nowhere to put it that reflects that, and the engine pays full retrieval and storage cost for it as if it were live.

A fourth tier — `Frozen` — splits the two axes. Confidence stays a property of `Draft / Validated / Core`; temperature becomes a property of `Frozen`. Cells in `Frozen` were *previously* one of the warm tiers; the demotion is on the temperature axis only.

## Why this is load-bearing, not optional

Four independent drivers make a Frozen tier load-bearing rather than a nice-to-have:

1. **Honest "persistent memory" story.** A memory engine without a cold/archival tier cannot truthfully say it handles long-lived corpora. Today the project's positioning leans on "persistent memory engine"; the missing cold tier is the gap between that claim and the implementation.
2. **Footprint priority.** 5K cells fits comfortably at 751 B/cell. 500K cells of accumulated long-running-agent memory does not. Compressed cold storage is what unlocks the next 100× of scale without burning RAM/disk.
3. **Cognitive realism.** Real memory has "forgotten but recoverable" — the half-remembered detail that comes back when something jogs it. Auto-thaw on retrieval hit is the architectural primitive that models this. Maps directly onto Project Ares NPC memory: NPCs whose deep history compresses cold but resurfaces when the right cue arrives.
4. **Multi-tenant economics.** Owners that go silent for long periods can be force-frozen at the owner level. Operationally important for any consumer running TardigradeDB as shared infrastructure (SaaS, multi-agent platforms).

All four reinforce each other; the design doesn't have to pick.

## Retrieval semantics decision

Two independent axes:

| Axis | Choice | Why |
|---|---|---|
| Default index inclusion | **Included in Vamana scan by default** | Lights up drivers 1 and 3. The honest persistent-memory story and cognitive realism both require that cold memories are *findable*, not archived behind an opt-in flag. |
| On-hit behaviour | **Decode-on-hit + auto-thaw to Validated** | Decode cost is paid only for cells that make top-k; auto-thaw models memory consolidation literally (touched memory warms up). |

The tail-latency cost of always-scanning Frozen cells is the price paid for the honest story. Bench-gated before commit: any retrieval latency regression at the corpus sizes the project claims (5K → 100K) is documented in absolute numbers, not hidden.

Auto-thaw is a load-bearing read-path side effect. Documented explicitly so consumers aren't surprised that `mem_read_pack` can mutate tier state.

## Codec roadmap

The compression codec for Frozen storage is a Strategy, swappable per engine instance. Two codecs in the roadmap; each ships as its own plan.

### Part 1 — `ZstdQ4Codec` (MVP)

Plain zstd level 3 over the already-Q4-quantized cell bytes. Expected ~3–4× compression. No model coupling, no calibration, no per-model bookkeeping, no new failure modes. Ships first because it lets the four-tier state machine, freeze triggers, auto-thaw, and owner-cascade freeze all land *without* taking on the calibration complexity of KVTC. The state machine is the architecturally load-bearing change; the codec is a knob.

### Part 2 — `KvtcCodec` (gated follow-up)

NVIDIA KVTC ([arXiv 2511.01815](https://arxiv.org/abs/2511.01815), ICLR 2026, open-source reference at `OnlyTerp/kvtc`). Three stages:

1. **PCA decorrelation on a learned basis.** Computed once per model from ~10K representative KV vectors. After rotation, variance concentrates in a small number of principal components; the rest carry near-zero signal.
2. **DP bit allocation.** Dynamic programming finds the optimal per-component bit budget given a target average bitrate. Many bits to high-variance components, few or zero to low-variance.
3. **DEFLATE entropy coding.** Quantized bit-stream squeezed by zlib (CPU path) or nvCOMP (future GPU path).

Published numbers: 20× compression typical, up to 40× on some workloads, reasoning/long-context accuracy preserved.

**On-demand calibration is the key API design choice.** Pre-shipping bases per "supported model" was rejected for four reasons:

- Fine-tunes become first-class — calibrate on the consumer's actual KV distribution, not a generic corpus.
- No "supported models" list to maintain or redistribute.
- Calibration samples never leave the consumer's machine.
- Per-owner / per-corpus bases stay architecturally possible if a real consumer needs them.

The pattern already exists in TardigradeDB: `Engine::load_embedding_table` and `Engine::set_projection_matrix` are consumer-provided, model-specific data the engine consumes. `Engine::set_kvtc_basis` is the same shape.

Two consumer flavours:

- **Explicit batch.** Consumer feeds a fixed calibration corpus, calls `finalize()`, persists the basis alongside the engine.
- **Bootstrap-by-doing.** Engine accumulates samples from normal `mem_write_pack` calls; when threshold is reached, the basis is computed and persisted automatically. Until the basis exists, Frozen tier falls back to `ZstdQ4Codec` — exactly why zstd ships first.

Lifecycle invariants:

- Every KVTC-encoded pack carries a 16-byte basis fingerprint in its header. Decode refuses on mismatch — typed error, never silent corruption.
- Basis lives in the snapshot; restore loads it; no recalibration after restore.
- Multi-basis support: a consumer who captures from two different models tags each basis with an ID; packs carry their basis ID.
- Re-calibration produces a new fingerprint; old packs keep their old basis. Old basis stays loaded (4 MB cost is negligible). No forced migration.

## What was rejected and why

Documented here so future contributors don't relitigate:

- **Per-sequence page tables (PagedAttention).** Vamana already indexes our cells; we have owners, not sequences. No problem to solve.
- **Radix-tree primary retrieval (RadixAttention).** Destroys the entire pitch — memory feels like cognition, not prefix lookup. Radix-tree dedup *at ingest* (inside `FileIngestor`) is fine; radix-tree at retrieval is not.
- **Remote tier (Redis/S3) in the hot path.** Sub-ms p99 local is the positioning. A remote tier is a different product.
- **LRU-only governance.** Confidence + recency decay is a strictly better signal for a memory engine. LRU is a downgrade.
- **Stacking transfer compression (CacheGen) over Q4 on the hot path.** Hot-path latency is sacred per CLAUDE.md. CacheGen-style variable-bitrate quant on the cold path is a different proposal (and listed as a separate top-3 recommendation in the originating research memo).
- **Multi-node distributed cold storage (Mooncake-style).** Future-horizon. Adds coordination overhead the latency budget can't absorb today.
- **GPU-accelerated DEFLATE via nvCOMP in v1.** CPU `flate2` path is good enough for a cold tier. GPU is a Part 3 optimization if ever needed.
- **Per-owner KVTC bases.** Architecturally possible, premature. Revisit if a real consumer needs it.

## Reuse audit

Existing patterns to lean on:

- `Tier::retrieval_boost` in `crates/tdb-core/src/types.rs` — pattern for the Frozen branch.
- `RetrievalKeyStrategy` trait in `crates/tdb-retrieval` — Strategy pattern reference for `FrozenCodec`.
- `MaintenanceWorker` / `ConsolidationSweepThread` Active Object daemons — the home for the auto-freeze tick.
- `CheckpointRepository` (Repository pattern) — snapshot bundling code path; extend for `frozen.seg.*`.
- `Engine::status()` — extend; don't add a parallel reporting API.
- `compute_for_save` / `load_embedding_table` / `set_projection_matrix` — pattern reference for `set_kvtc_basis`.

## Where the design touches the project

`tdb-core::types` (tier enum), `tdb-storage` (codec trait + impl + segregated `frozen.seg.*` family), `tdb-governance` (auto-freeze sweep), `tdb-engine` (freeze API + auto-thaw on retrieval + snapshot/restore bundling), `tdb-retrieval` (Vamana scan inclusion + decode dispatch), `tdb-python` (bindings parity), `examples/` (consumer demos for both parts).

The Frozen tier doesn't touch the warm-tier state machine. Draft/Validated/Core hysteresis stays exactly as is.
