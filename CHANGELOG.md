# Changelog

All notable changes to TardigradeDB are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project follows semver-ish during the 0.x.y series: minor bumps mark new user-facing surface, patch bumps mark fixes and internal changes. The release procedure lives in [`RELEASE.md`](RELEASE.md).

---

## [Unreleased]

_no changes yet_

## [0.2.0] — 2026-05-17

Foundation-completion release. Adds the surface non-Python consumers need to plug in (CLI, HTTP/REST bridge, Node.js example), the persistence primitives consumers need to ship real workloads (owner registry, portable snapshot/restore, labeled checkpoints, streaming write buffer, synchronous sweep trigger), and ergonomic upgrades to the Python facade (builder pattern, encode_query convenience, multi-agent shared-engine support). Two runnable demos validate the whole surface.

### Added

- **CLI** (`tardigrade init|store|query|status|consolidate`): Strategy + Template Method per subcommand, `python-basic` and `rust-basic` starter templates (`0282db2`).
- **Owner registry** on `Engine`: `list_owners`, `owner_exists`, `delete_owner` with cascade (`36680e3`).
- **Portable snapshot/restore** (`Engine::snapshot`, `Engine::restore_from`): tar archive with magic `"tdb!"`, format_version, codec ids (`q4` / `top5avg`), SHA-256 payload digest, pack/owner stats. Bounded reads cap manifest at 1 MiB and payload entries at 16 GiB. Foot-gun guard refuses `out_path` inside `engine_dir` (`dbf77a0`).
- **Streaming write buffer** (`Engine::open_with_write_buffer`): coalesces `mem_write_pack` calls behind a single `append_batch` fsync. Pack ids return synchronously; visibility deferred to `flush_buffer()` or auto-flush at threshold (`99c1482`).
- **Synchronous governance trigger** (`Engine::sweep_now(hours, threshold) -> usize`): runs one tick of the maintenance worker inline (`46286ee`).
- **Labeled checkpoints** (`CheckpointRepository`, Repository pattern): label-scoped slots `<root>/<label>/<NNNN>.tar`, `save_from` / `list` / `latest` / `restore_latest` (`1198c9a`).
- **HTTP/REST bridge** (`python/tardigrade_http/`): FastAPI Adapter exposing `POST /mem/store`, `POST /mem/query`, `GET /mem/owners`, `GET /mem/status`, `POST /mem/save`, `POST /mem/restore`. RFC 7807 `application/problem+json` errors. Checked-in OpenAPI contract at `python/tardigrade_http/schema.yaml` with drift-guard test (`4fe00b5`).
- **Node.js consumer + generated TypeScript types + consumer guide** (`examples/nodejs_consumer/index.mjs`, `python/tardigrade_http/types.ts`, `docs/guide/consumers.md`): dependency-free Node 22+ script exercises the foundation surface end-to-end; TypeScript types regenerated from the OpenAPI schema; three documented integration patterns (`6658eed`).
- **Client builder** (`TardigradeClient.builder()`): fluent `with_engine_dir` / `with_engine` (mutually exclusive — the latter is the multi-agent shared-engine pattern) / `with_owner` / `with_tokenizer` / `with_capture_fn` / `with_vamana_threshold`. `BuilderIncomplete` raised on ambiguous or incomplete state (`6232fe5`, `c1fb890`).
- **`TardigradeClient.encode_query(text) -> ndarray`**: returns the raw retrieval key without a query pass (`c92d584`).
- **Multi-agent demo** (`examples/multi_agent_demo.py`): three agents share one engine, exercise every foundation primitive as an acceptance gate (`c1fb890`).
- **Interactive-fiction demo** (`examples/if_demo.py`): two NPCs with distinct personas observe the same scripted events, store divergent memories under owner scoping (`6f00354`).
- **Pre-commit lint gate** (`scripts/lint_lazy_imports.py`): catches CI failures caused by import-time loading of the native extension (`0e84920`).
- **Bench instrumentation**: per-(dataset, category) score breakdown, retrieval-only headline (Recall@k / NDCG@k), answer-text metric sidecar, shared-LLM latency benchmark scaffold (warm-path real; cold-path and text-RAG comparators simulated and tracked as a follow-up).

### Changed

- `TardigradeClient.__init__` now accepts `engine=` as an alternative to `db_path` (mutually exclusive). Enables one engine + N clients in multi-agent workloads (`c1fb890`).
- `TardigradeClient.query()` routes through `encode_query()`, unifying the encode-then-read code path.
- `tardigrade_hooks` package switched to PEP 562 `__getattr__` for `TardigradeClient` so importing `tardigrade_hooks.constants` no longer loads the native extension (`0e84920`).
- Plan-phase identifiers stripped from code, public docstrings, demo print lines, and `CLAUDE.md` (`61e7d09`).

### Fixed

- Node.js consumer no longer double-reads the response body (undici `"Body has already been read"` guard); single `unwrap(path, response)` helper handles both OK and error branches.
- TardigradeAdapter's evidence path returns chunk text rather than the parent item's full context (`64be483`).
- TardigradeAdapter dedups retrieved evidence by chunk text (`31e711e`).
- Per-instance lock around `CrossEncoder.predict` removes the parallel-score drift observed under multi-threaded eval (`bfd248a`).
- LongMemEval bench-prep emits per-turn gold (not per-session) and raw turn text (not `[date]`-wrapped) (`f0fb531`, `e443dd8`).

### Notes

- A bench audit retracted the historical "68.2% LoCoMo / 90.9% LongMemEval" baseline numbers (they measured the lexical fallback adapter on a corpus corrupted by a dataset-prep bug); honest native-engine number on the clean dataset is ~36% R@1 at 50-item scale. Synthetic-corpus measurements (100% recall at 5K memories, vague-query refinement, KV injection on gibberish facts) are unaffected.
- Positioning pivoted to **latency / footprint / KV-native API** (sub-millisecond p99 retrieval at 5K cells, ~751 B per cell on disk) — see `docs/positioning/latency_first.md`.

## [0.1.5] — 2026-05-02

Last patch in the 0.1.x line. The features below were under `[Unreleased]` at tag time; recorded here so the history is not lost.

### Added — DX & Repo Quality
- First-class crate-level `//!` documentation across all 6 library crates, following tokio/serde/crossbeam quality bar
- `Documentation Standards` section added to `CLAUDE.md` and `AGENTS.md`
- `just ci` now mirrors GitHub CI exactly: fmt → lint → typos → test → deny → doc
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` propagated to all lint/test recipes (Python 3.14 compatibility)
- `lefthook` replaces manual `.git/hooks/pre-push`; pre-commit runs fmt/clippy/typos in parallel
- `LICENSE` (MIT), `CHANGELOG.md`, `CONTRIBUTING.md`, GitHub issue templates, PR checklist template
- `crates/tdb-engine/examples/basic_usage.rs` — runnable Rust demo

### Added — Python Bindings
- `tdb-python` crate: PyO3 bindings exposing `Engine` and `ReadResult` as Python classes
- `mem_write` / `mem_read` accepting NumPy `float32` arrays via zero-copy slice access
- `cell_importance`, `cell_tier`, `cell_count`, `trace_ancestors`, `has_vamana`, `advance_days` Python methods
- `maturin` build integration (`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop`)

### Added — Causal Trace Graph + WAL
- `tdb-index`: `TraceGraph` — directed episodic graph with `CausedBy`, `Follows`, `Supports`, `Contradicts` edge types
- `Wal` — append-only binary write-ahead log for `TraceGraph` mutations; 26-byte fixed-size records
- WAL replay on `Engine::open`; crash recovery restores causal graph from durable log
- `Engine::trace_ancestors` — transitive ancestor queries over causal edges
- `Engine::mem_write` accepts optional `parent_cell_id` for causal edge recording

### Added — Vamana ANN Index
- `tdb-index`: `VamanaIndex` — DiskANN-style single-layer graph for cold-path ANN retrieval
- Medoid seeding for greedy beam-search queries; `max_degree` configurable out-degree
- Automatic Vamana activation at 10,000 cells per engine instance
- `Engine::has_vamana` — reflects whether the graph index is active
- Benchmarks: `vamana_build`, `vamana_query` in `tdb-index`

### Added — Governance — Recency Decay + Tier Promotion
- `tdb-governance`: `recency_decay(days)` — exponential decay r = exp(−Δt/τ), τ = 30 days
- Recency multiplier applied to retrieval scores during `mem_read`
- `Engine::advance_days` — simulate time passage for testing and reproducible benchmarks
- `Engine::cell_tier` — expose current maturity tier from Python and integration tests

### Added — Governance — Importance Scoring
- `tdb-governance`: `ImportanceScorer` — ι ∈ [0,100]; +3 on access, +5 on update, ×0.995 daily decay
- `TierStateMachine` — three tiers (Draft/Validated/Core) with hysteresis gaps (30 and 25 pts)
- On-access and on-update governance hooks wired into `Engine::mem_read` / `Engine::mem_write`
- `Engine::cell_importance` — query current importance score by cell ID

### Added — Semantic Lookaside Buffer (SLB)
- `tdb-retrieval`: `SemanticLookasideBuffer` — LRU hot cache with INT8 scalar quantization
- Sub-5μs attention-score hits for recently-accessed cells (NEON SDOT on Apple Silicon)
- SLB inserts on every `mem_write`; checked first in the `mem_read` pipeline

### Added — Brute-Force Retrieval + Attention Scoring
- `tdb-retrieval`: `BruteForceRetriever` — exact scaled dot-product attention (q · k / √d_k)
- Owner-filtered partitions for per-agent isolation
- `BruteForceRetriever::query` returns top-k by score, supporting `mem_read`

### Added — Block Pool + Q4 Quantization
- `tdb-storage`: `BlockPool` — append-only mmap'd segment files (64 MiB default) with `fsync` durability
- Group-wise Q4 (GGML Q4_0) quantization: 8 × f32 → 4 × u8 + f16 scale, ~5.3x compression
- `SynapticStore` — secondary per-owner lookup index for O(1) cell retrieval by owner ID
- Binary record format: `magic(4) | version(1) | dim(4) | cell_id(8) | ... | key_q4 | val_q4`

### Added — Core Types
- `tdb-core`: `MemoryCell` + `MemoryCellBuilder` — KV-cache tensor pair with metadata
- `Tier` enum (Draft/Validated/Core) and `CellMeta` (tier, importance, timestamps, token span)
- `TardigradeError` + `Result<T>` — shared error type across all crates
- `SynapticBankEntry` — lightweight index record for owner-keyed lookups

### Added — Workspace Scaffold
- Cargo workspace: `tdb-core`, `tdb-storage`, `tdb-retrieval`, `tdb-index`, `tdb-governance`, `tdb-engine`, `tdb-python`
- `justfile` with `fmt`, `lint`, `test`, `bench`, `coverage`, `doc`, `ci`, `setup` recipes
- `lefthook.yml` — pre-commit (fmt/clippy/typos parallel) + pre-push (`just ci`)
- `cargo-deny` configuration (license allow-list, advisory deny, wildcard ban)
- GitHub Actions CI: check+lint, test matrix (stable/nightly), coverage, MSRV, doc deploy, doc-check
- `nextest` CI profile with retry on failure and extended timeouts

### Added — Project Foundation
- Initial repository structure and `CLAUDE.md` with Aeon Architecture specification
- Research corpus: technical design document, spec, competitor analysis, industry analysis
- Spec corrections based on external research: custom mmap arena (not safetensors), Vamana graph (not HNSW), Rust+CUDA (not pure Rust)
