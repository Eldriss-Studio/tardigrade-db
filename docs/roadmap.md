# Roadmap

What's shipped, what's next, and where the project is headed. The README's `## Project Status` is the one-line summary; this document is the full picture.

## Done

### Storage Layer

- [x] **Q4 group-wise 4-bit quantization** — same scheme as llama.cpp's Q4_0. 4× compression vs FP32 with MSE < 0.01.
- [x] **Append-only segments with crash-safe binary records** — 256 MB segments, `sync_data()` on every write, partial-record detection on recovery.
- [x] **NEON SIMD INT8 dot product (ARM) with scalar fallback** — `vmull_s8` → `vpadalq_s16` → `vaddvq_s32`, < 1 % relative error.
- [x] **Segment compaction** — Mark-Sweep GC rewrites segments below 50 % live ratio. `Engine::compact()` + Python binding.
- [x] **TextStore** — durable Rust-side text persistence; JSON sidecar removed.
- [x] **DeletionLog** — crash-safe pack-delete history.
- [x] **SynapticBank (LoRA adapter) persistence** — FP16, same binary conventions as the cell block pool.

### Retrieval Layer

- [x] **Semantic Lookaside Buffer (SLB)** — fixed-capacity LRU, INT8 quantized.
- [x] **Brute-force latent-space attention retrieval** — `(q · k) / √d_k`, validated by MemArt at < 10 K blocks.
- [x] **Per-token retrieval with Top5Avg scoring** — 100 % recall at 100 memories.
- [x] **Three-stage retrieval chain** — SLB → Vamana → BruteForce, Chain of Responsibility with `CellId` dedup.
- [x] **Refinement strategies (Strategy pattern)** — `MeanCentered`, `LatentPrf` re-rank first-stage results.
- [x] **Cross-encoder reranker** — Stage-2 over text-bearing candidates via `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 M params); ~30 % latency overhead.
- [x] **Retrieval-key strategies** — `LastToken`, `MeanPool`, `Projected` in `tdb-retrieval::retrieval_key`. Engine-owned embedding table.
- [x] **Batch read API** — `Engine.mem_read_pack_batch(queries, k, owner)` — single engine call for multi-agent fanout.
- [x] **Multi-layer RRF fusion** — `Engine.mem_read_multi_layer(query_keys, k, owner, rrf_k)`.

### Organization Layer

- [x] **Vamana graph index** — DiskANN-style, batch + incremental, lazy activation above a configurable threshold.
- [x] **Trace causal graph** — `CausedBy`, `Follows`, `Contradicts`, `Supports` edges with BFS ancestor traversal.
- [x] **Write-Ahead Log** — every Trace mutation fsynced before in-memory apply; lenient crash recovery.
- [x] **WAL checkpointing** — periodic compaction of the log against the durable snapshot.

### Governance Layer

- [x] **Adaptive Knowledge Lifecycle** — importance scoring, tier hysteresis, recency decay.
- [x] **Active governance** — tier-based retrieval boost at query time (Core 1.25×, Validated 1.1×), `evict_draft_packs()` with owner scoping.
- [x] **MaintenanceWorker** — Active Object running decay + eviction + compaction in a background `std::thread`.
- [x] **Synchronous sweep trigger** — `Engine::sweep_now(hours, threshold)` runs one tick inline for save-game boundaries.

### Engine

- [x] **Memento-pattern state rebuild** — full in-memory state reconstructed from durable sources on `Engine::open`.
- [x] **KV Pack API** — atomic multi-layer storage (`mem_write_pack` / `mem_read_pack`), single fsync, pack-level governance.
- [x] **Streaming write buffer** — `Engine::open_with_write_buffer(dir, BufferConfig { … })` coalesces N writes behind one fsync.
- [x] **Owner registry** — `list_owners`, `owner_exists`, `delete_owner` with cascade.
- [x] **Portable snapshot/restore** — tar archive with magic, format version, codec identifiers, SHA-256, bounded reads to refuse OOM on corrupt archives.
- [x] **Labeled checkpoints** — `CheckpointRepository` with label-scoped slots (`autosave/0001.tar`, `chapter-3-end/0001.tar`, …).
- [x] **Durable action scheduler** — `Engine::schedule(fires_at, action)` / `cancel_scheduled` / `fire_due_scheduled`. Persists across restart.
- [x] **Engine-owned fingerprint LRU cache** — vLLM connector path; 2× faster than Python OrderedDict at capacity=256.
- [x] **KV-block reshape in Rust** — `tardigrade_db.flat_to_paged` / `paged_to_flat`; 2.1× over numpy on the save path.
- [x] **Torch tensor bridge** — `tardigrade_db.flat_to_paged_torch(tensor, …)` via `data_ptr()`; no tch-rs, no libtorch link.

### Python bridge + consumer surfaces

- [x] **PyO3 Python bindings** — full cell + pack API, numpy interop, `Arc<Mutex<Engine>>` with GIL release for thread safety.
- [x] **HuggingFace KV hook** — per-token Q / K projections, GQA expansion.
- [x] **KnowledgePackStore** — end-to-end injection via Knowledge Packs approach.
- [x] **vLLM KV Connector v1** — `tardigrade_vllm.connector.TardigradeConnector` on Qwen3-0.6B + vLLM 0.19.
- [x] **`tardigrade chat` CLI** — persistent backend, sub-ms recall reported inline, multi-persona subjectivity via owner scoping.
- [x] **CLI bootstrap** — `tardigrade init | store | query | status | consolidate`, two starter templates (`python-basic`, `rust-basic`).
- [x] **HTTP / REST bridge** — `python/tardigrade_http/` exposes `POST /mem/store`, `POST /mem/query`, `GET /mem/owners`, `GET /mem/status`, `POST /mem/save`, `POST /mem/restore` as an Adapter over `Engine`.
- [x] **Node.js consumer + TS types + consumer guide** — `docs/guide/consumers.md`, `examples/nodejs_consumer/index.mjs`.
- [x] **`TardigradeClient` facade** — `store / query / ingest_file / ingest_text / consolidate / consolidate_all`.
- [x] **Multi-view consolidation v2** — `add_view_keys`, `ViewGenerator` (3 rule-based + LLM option), `MemoryConsolidator` (tier-gated, idempotent).
- [x] **ReflectiveLatentSearch (RLS)** — 5 strategies: keyword / multiphrasing / embedding / generative / agent reformulation.
- [x] **TextChunker + FileIngestor** — token-bounded chunking with sequential Supports edges.
- [x] **Chat-template adapter** — `LegacySystemAdapter` (Qwen3 byte-compat) + `UserMessageAdapter` (strict-template default).
- [x] **Per-model retrieval-key calibration** — `select_query_layer(model, tok, registry=reg)` + `CalibrationRegistry`.
- [x] **K-vector retrieval encoding** — `KVectorKeyStrategy(softmax_layer_idx)`; unblocks the hybrid-attention cohort.
- [x] **Hybrid-safe storage path** — recurrent-layer filtering for RecurrentGemma, Jamba, Zamba, Granite-4, MiniMax, etc.
- [x] **Hybrid end-to-end recall validated** — 20/20 retrieval, 18/20 generation-hit on RecurrentGemma-2B-it (Griffin 3:1 layout).
- [x] **Typed errors with miette** — every variant has a stable `tdb::<area>::<name>` code and `#[help]` annotation.
- [x] **Python↔Rust boundary migration (v0.7.0)** — 12 boundary fixes shipped, 4 produced meaningful perf wins, honest commit messages on the parity ones. See `CHANGELOG.md` for the full breakdown.

### Benchmarking + CI

- [x] **Criterion benchmarks across all subsystems**.
- [x] **Bench V1 harness** — comparable smoke + full matrix (Tardigrade + Mem0 + Letta), Markdown reports, baseline-vs-candidate compare.
- [x] **CI gates** — fmt, clippy `--pedantic`, typos, cargo-deny, nextest on Ubuntu + macOS, MSRV (1.95), rustdoc with `-D warnings`, Bench V1 Smoke Gate.
- [x] ~~**Full benchmark runs** — LongMemEval 90.9 %, LoCoMo 68.2 %~~ **[RETRACTED 2026-05-14]** — those measured the lexical fallback adapter on corrupted data. Honest native-engine number on clean LoCoMo: ~36 % R@1 at 50-item scale; full-corpus re-run pending. See [`experiments/2026-05-14-bench-audit.md`](experiments/2026-05-14-bench-audit.md).

---

## Next up

- [ ] **Release-mode benchmark numbers** — publish actual `cargo bench` results post-`v0.7.0`.
- [ ] **LLM agent vocabulary bridge** — `LLMAgentReformulationStrategy` validation on LoCoMo full run on the clean dataset, against the honest ~36 % R@1 native baseline.
- [ ] **Storage reduction** — 730 KB per memory is large. Investigate selective layer storage, INT8 KV, or FP16 for injection-critical layers.

---

## Future

### Making the model remember across sessions — four paths to production

The core problem: an LLM forgets everything between requests. TardigradeDB stores the model's own internal state (KV cache tensors) so it can be restored later — making the model behave as if it had already lived through prior conversations. The question is how to get that stored state back into the model at serving time.

We discovered that vLLM's KV Connector v1 API only supports **prefix-cache acceleration** (same prompt prefix → skip recomputation). It cannot inject KV from a different prompt to influence generation. So there are four paths to actually making "remember" work in production, ordered from easiest to hardest:

**Path 1 — HuggingFace direct injection** ✅ **Verified (April 27, 2026).**

10 fully synthetic gibberish facts via `KnowledgePackStore` injected stored KV via `model.generate(past_key_values=...)` on Qwen3-0.6B. **9 / 10, matching text RAG exactly (100 % recall ratio), with 236 prompt tokens saved.** See [`experiments/synthetic-kv-injection.md`](experiments/synthetic-kv-injection.md).

**Path 2 — vLLM "memory prefix" cache** ✅ **Complete (April 27, 2026).**

`VLLMMemoryClient` (in `python/tardigrade_vllm/prefix_client.py`) prepends governed memory prefixes to prompts before they reach vLLM. Same owner's prefix is token-identical across requests, so vLLM's stock prefix-cache serves the stored KV at zero prefill cost. Per-owner isolation, pluggable format strategies (`BulletListFormat`, `TierAnnotatedFormat`), draft exclusion all built in.

**Path 3 — SGLang connector** ❌ **Ruled out (April 28, 2026).**

SGLang's RadixAttention is strictly prefix-based — same limitation as vLLM v1. `match_prefix()` operates on `RadixKey(token_ids)` (token identity only). No mechanism for cross-prompt KV injection. See [`experiments/sglang-investigation.md`](experiments/sglang-investigation.md).

**Path 4 — Custom attention plugin for vLLM** (hardest, most flexible)

If neither Path 2 nor Path 3 is sufficient, the remaining option is to modify vLLM's attention layer directly. A custom backend could mix loaded "memory" KV with the current request's computed KV — effectively extending the attention window with stored activations without them being part of the prompt. Requires a vLLM fork, careful handling of position encoding (RoPE offsets for the injected blocks), and ongoing maintenance as vLLM evolves. Worth pursuing only after Paths 1-3 are evaluated.

### Engine and infrastructure

- [ ] **Vamana edge persistence** — the graph is currently rebuilt from scratch on `refresh()` (O(n²)). Serializing edges for O(n) load is planned.
- [ ] **Incremental compaction** — release engine lock between segments to reduce contention window during compaction.
- [ ] **Disk-aware Vamana** — PageANN-style page-node alignment for billion-scale cold storage.
- [ ] **Multi-model dimension support** — handle different models (different `d_k`) in one engine instance.
- [ ] **CUDA GPU DMA** — direct NVMe → GPU transfers via cuFile / GDS (requires CUDA SDK integration).
- [ ] **RelayCaching** — cross-agent KV cache reuse for multi-agent handoffs.
- [ ] **Custom vLLM attention plugin** — see Path 4 above.

---

## See also

- [`CHANGELOG.md`](../CHANGELOG.md) — per-version user-visible changes.
- [`docs/technical/tdd.md`](technical/tdd.md) — full technical design document.
- [`docs/technical/spec.md`](technical/spec.md) — condensed four-layer specification.
- [`docs/experiments/`](experiments/) — every dated experiment writeup.
