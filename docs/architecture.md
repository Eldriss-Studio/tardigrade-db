# Architecture

TardigradeDB is a four-layer memory kernel ("Aeon Architecture") that treats memory as a managed OS resource. The layers stack from the bytes on disk up to the lifecycle policy that decides what gets remembered.

## Aeon four-layer model

```
┌─────────────────────────────────────────────────────┐
│  Governance    Adaptive Knowledge Lifecycle (AKL)    │
│                importance scoring · maturity tiers    │
│                recency decay · self-curation          │
├─────────────────────────────────────────────────────┤
│  Organization  Vamana graph index (DiskANN-style)    │
│                Trace (causal episodic graph)          │
│                WAL · checkpointed on refresh          │
├─────────────────────────────────────────────────────┤
│  Retrieval     Per-token Top5Avg (latent attention)  │
│                SLB (INT8 scalar quantization)         │
│                BruteForce (exact fallback)            │
├─────────────────────────────────────────────────────┤
│  Storage       Q4 KV-cache block pool                │
│                append-only segments · TextStore       │
│                DeletionLog · SynapticStore            │
└─────────────────────────────────────────────────────┘
```

### Storage Layer — custom from scratch, not a wrapper

- **Q4 group-wise 4-bit quantization** — same scheme as llama.cpp's Q4_0. 4× compression vs FP32 with MSE < 0.01 on typical activations. Implemented as a Strategy pattern (`Quantizer` trait) so Q8 / FP16 variants slot in without changing the storage layer.
- **Append-only segment files** — 256 MB segments with binary length-prefixed records. Magic/version header validation. `sync_data()` on every write for crash durability. Partial record detection on recovery (truncated writes are silently discarded, not propagated).
- **Segment scanning recovery** — on open, segments are scanned to rebuild the `CellId → (segment, offset)` index. O(n) in cell count, not cell size — only the record header is read during recovery.
- **TextStore** — durable Rust-side text persistence for pack fact strings, append-only and fsynced.
- **SynapticStore** — parallel repository for LoRA adapters (FP16), same binary conventions as the cell block pool.
- **DeletionLog** — crash-safe pack-delete history; reconciled on every `refresh()`.

### Retrieval Layer — latent-space attention, not text search

- **NEON SIMD INT8 dot product** — ARM aarch64 intrinsics (`vmull_s8` → `vpadalq_s16` → `vaddvq_s32`) with scalar fallback for x86. <1 % relative error vs FP32 reference.
- **Semantic Lookaside Buffer (SLB)** — fixed-capacity LRU cache storing keys in symmetric INT8 quantization. Exploits conversational locality — recently accessed cells are served at SIMD speed.
- **Brute-force attention scoring** — exhaustive `score = (q · k) / √d_k` over all indexed keys. Validated by the MemArt paper: at per-agent scale (< 10 K blocks), brute-force SIMD matmul outperforms ANN indexes.
- **Three-stage retrieval chain** — SLB (hot, INT8) → Vamana (warm, graph ANN) → BruteForce (cold, exact). Chain of Responsibility pattern with deduplication by `CellId`.
- **Per-token Top5Avg** — score each cell by the mean of the top 5 dot products between query tokens and stored tokens. Captures latent semantic similarity without falling into the mean-pool "gravity well" where one memory dominates all queries.
- **Refinement strategies** — `MeanCentered` and `LatentPrf` (Strategy pattern) re-rank first-stage results for vague-query rescue; cross-encoder reranker available as Stage-2 over text-bearing candidates.

### Organization Layer — graph index + causal memory + crash recovery

- **Vamana graph index** — DiskANN-style single-layer graph with multi-seed greedy beam search. Robust pruning (angular diversity via α parameter) produces navigable graphs. Supports both batch build and incremental `insert_online`. Lazy activation: built only when cell count crosses a configurable threshold.
- **Trace causal graph** — directed edges (`CausedBy`, `Follows`, `Contradicts`, `Supports`) with BFS transitive ancestor traversal. When a cell is written with a parent, the causal edge is recorded.
- **Write-Ahead Log** — every Trace mutation is WAL-logged (with fsync) before being applied in memory. On engine open, WAL is replayed to rebuild the graph. Lenient crash recovery: partial records are discarded, not propagated.

### Governance Layer — self-curating memory

- **Importance scoring (ι)** — bounded `[0, 100]`. `+3` on read access, `+5` on write. Daily decay factor `0.995`.
- **Maturity tiers with hysteresis** — Draft → Validated (ι ≥ 65, demote < 35) → Core (ι ≥ 85, demote < 60). The 30-point hysteresis gaps prevent oscillation near boundaries. Skip-tier promotion/demotion for large importance jumps.
- **Recency decay** — `r = exp(-Δt / 30)` applied as a retrieval score multiplier. ~21-day half-life. Cells unused for a month have their relevance halved.
- **Active governance** — tier-based retrieval boost at query time (Core 1.25×, Validated 1.1×), `evict_draft_packs()` with owner scoping, `MaintenanceWorker` (Active Object) running decay + eviction + compaction in a background thread.

### Engine — Facade that ties it all together

- **Memento-pattern state rebuild** — on `Engine::open`, all in-memory state (retriever index, governance scores/tiers, ID counter, pack directory, text store, deletion log) is reconstructed from durable sources (segments + WAL). The engine can crash at any point and recover.
- **Governance computed before persistence** — on-disk importance and tier match the in-memory state, ensuring correct rebuild.
- **KV Pack API** — atomic multi-layer KV storage (`mem_write_pack` / `mem_read_pack`). Stores all layers of a KV cache in a single fsync. Retrieves complete packs grouped by memory. Pack-level governance.
- **Python bridge (PyO3)** — `tardigrade_db.Engine` class with the full cell + pack API, numpy interop, thread safety via `Arc<Mutex<Engine>>` with GIL release. See [`docs/guide/python-api.md`](guide/python-api.md).

## Crate map

| Crate | Layer | Responsibility |
|-------|-------|----------------|
| `tdb-core` | — | Shared types: `MemoryCell`, `KVPack`, `SynapticBank`, `Tier`, error types |
| `tdb-storage` | Storage | Quantized block pool, mmap arena, segment management, text store, deletion log |
| `tdb-retrieval` | Retrieval | Per-token retrieval (Top5Avg), SLB (INT8), SIMD distance, refinement strategies, retrieval-key strategies |
| `tdb-index` | Organization | Vamana graph index, causal trace, write-ahead log |
| `tdb-governance` | Governance | Importance scoring, maturity tiers, temporal decay |
| `tdb-engine` | Orchestrator | Engine facade, pack API, scheduler, KV-block reshape, fingerprint cache |

## Why Python exists in this project

The Rust kernel (storage, retrieval, governance, indexing) is a self-contained library. It does not need Python.

Python exists for one reason: **to bridge TardigradeDB to model inference frameworks**. HuggingFace Transformers is the only practical way to access a model's KV cache (`past_key_values`) on local hardware. The Python layer (`tardigrade_hooks`) captures those tensors and feeds them to the Rust engine via PyO3 bindings.

It also hosts the **vLLM KV Connector** (`tardigrade_vllm.connector`), which plugs into vLLM's official KV Connector v1 API (validated end-to-end on vLLM 0.19 with Qwen3-0.6B). See [`docs/guide/vllm-setup.md`](guide/vllm-setup.md). Long-term, both bridges may be reimplemented as direct Rust integrations with vLLM or SGLang; today they are Python adapters.

## Design principles

- **Tensor-native** — the primary stored unit is a KV cache tensor. Reads inject pre-computed K/V directly into the attention stack. No tokenization round-trip.
- **Zero external dependencies** — no Postgres, Neo4j, or vector DB. Custom storage engine with custom indices.
- **Latent-space retrieval** — relevance via attention in latent space, not cosine similarity over external embeddings.
- **Self-curating** — the AKL algorithm autonomously manages promotion, demotion, and decay. No application-level memory management.

## Reliability & consistency contracts

- **Durability boundary** — writes are accepted asynchronously and durability is tracked via a monotonic `durable_offset`. Client-visible guarantees are defined against this boundary.
- **Read visibility modes** — two explicit modes exist:
  - `unconfirmed`: lowest-latency path, can expose state before durability is confirmed.
  - `confirmed`: waits until `durable_offset >= tx_offset` before exposing results/updates (default for safety-critical paths).
- **Recovery pipeline** — startup follows: restore latest valid snapshot → replay WAL/commitlog suffix → rebuild derived in-memory state.
- **Derived state policy** — indexes, caches, and other derived structures must be rebuildable from durable history and snapshots.
- **Replay failure policy** — replay runs fail-fast on inconsistency. The engine must not serve reconstructed-but-untrusted state.
- **Observability requirements** — replay/snapshot timing, replay counts, and durability queue depth are first-class metrics.

## See also

- [`docs/technical/tdd.md`](technical/tdd.md) — full technical design document.
- [`docs/technical/spec.md`](technical/spec.md) — condensed four-layer specification.
- [`docs/positioning/latency_first.md`](positioning/latency_first.md) — measured numbers on the latency / footprint / KV-native dimensions.
- [`docs/positioning.md`](positioning.md) — qualitative comparison vs embedding RAG and traditional KV cache.
