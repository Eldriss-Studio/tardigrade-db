# Changelog

All notable changes to TardigradeDB are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project does not yet follow semantic versioning — all changes are under `[Unreleased]` until a stable public API is defined.

---

## [Unreleased]

### Added — Phase 11: DX & Repo Quality
- First-class crate-level `//!` documentation across all 6 library crates, following tokio/serde/crossbeam quality bar
- `Documentation Standards` section added to `CLAUDE.md` and `AGENTS.md`
- `just ci` now mirrors GitHub CI exactly: fmt → lint → typos → test → deny → doc
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` propagated to all lint/test recipes (Python 3.14 compatibility)
- `lefthook` replaces manual `.git/hooks/pre-push`; pre-commit runs fmt/clippy/typos in parallel
- `LICENSE` (MIT), `CHANGELOG.md`, `CONTRIBUTING.md`, GitHub issue templates, PR checklist template
- `crates/tdb-engine/examples/basic_usage.rs` — runnable Rust demo

### Added — Phase 10: Python Bindings
- `tdb-python` crate: PyO3 bindings exposing `Engine` and `ReadResult` as Python classes
- `mem_write` / `mem_read` accepting NumPy `float32` arrays via zero-copy slice access
- `cell_importance`, `cell_tier`, `cell_count`, `trace_ancestors`, `has_vamana`, `advance_days` Python methods
- `maturin` build integration (`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop`)

### Added — Phase 9: Causal Trace Graph + WAL
- `tdb-index`: `TraceGraph` — directed episodic graph with `CausedBy`, `Follows`, `Supports`, `Contradicts` edge types
- `Wal` — append-only binary write-ahead log for `TraceGraph` mutations; 26-byte fixed-size records
- WAL replay on `Engine::open`; crash recovery restores causal graph from durable log
- `Engine::trace_ancestors` — transitive ancestor queries over causal edges
- `Engine::mem_write` accepts optional `parent_cell_id` for causal edge recording

### Added — Phase 8: Vamana ANN Index
- `tdb-index`: `VamanaIndex` — DiskANN-style single-layer graph for cold-path ANN retrieval
- Medoid seeding for greedy beam-search queries; `max_degree` configurable out-degree
- Automatic Vamana activation at 10,000 cells per engine instance
- `Engine::has_vamana` — reflects whether the graph index is active
- Benchmarks: `vamana_build`, `vamana_query` in `tdb-index`

### Added — Phase 7: Governance — Recency Decay + Tier Promotion
- `tdb-governance`: `recency_decay(days)` — exponential decay r = exp(−Δt/τ), τ = 30 days
- Recency multiplier applied to retrieval scores during `mem_read`
- `Engine::advance_days` — simulate time passage for testing and reproducible benchmarks
- `Engine::cell_tier` — expose current maturity tier from Python and integration tests

### Added — Phase 6: Governance — Importance Scoring
- `tdb-governance`: `ImportanceScorer` — ι ∈ [0,100]; +3 on access, +5 on update, ×0.995 daily decay
- `TierStateMachine` — three tiers (Draft/Validated/Core) with hysteresis gaps (30 and 25 pts)
- On-access and on-update governance hooks wired into `Engine::mem_read` / `Engine::mem_write`
- `Engine::cell_importance` — query current importance score by cell ID

### Added — Phase 5: Semantic Lookaside Buffer (SLB)
- `tdb-retrieval`: `SemanticLookasideBuffer` — LRU hot cache with INT8 scalar quantization
- Sub-5μs attention-score hits for recently-accessed cells (NEON SDOT on Apple Silicon)
- SLB inserts on every `mem_write`; checked first in the `mem_read` pipeline

### Added — Phase 4: Brute-Force Retrieval + Attention Scoring
- `tdb-retrieval`: `BruteForceRetriever` — exact scaled dot-product attention (q · k / √d_k)
- Owner-filtered partitions for per-agent isolation
- `BruteForceRetriever::query` returns top-k by score, supporting `mem_read`

### Added — Phase 3: Block Pool + Q4 Quantization
- `tdb-storage`: `BlockPool` — append-only mmap'd segment files (64 MiB default) with `fsync` durability
- Group-wise Q4 (GGML Q4_0) quantization: 8 × f32 → 4 × u8 + f16 scale, ~5.3x compression
- `SynapticStore` — secondary per-owner lookup index for O(1) cell retrieval by owner ID
- Binary record format: `magic(4) | version(1) | dim(4) | cell_id(8) | ... | key_q4 | val_q4`

### Added — Phase 2: Core Types
- `tdb-core`: `MemoryCell` + `MemoryCellBuilder` — KV-cache tensor pair with metadata
- `Tier` enum (Draft/Validated/Core) and `CellMeta` (tier, importance, timestamps, token span)
- `TardigradeError` + `Result<T>` — shared error type across all crates
- `SynapticBankEntry` — lightweight index record for owner-keyed lookups

### Added — Phase 1: Workspace Scaffold
- Cargo workspace: `tdb-core`, `tdb-storage`, `tdb-retrieval`, `tdb-index`, `tdb-governance`, `tdb-engine`, `tdb-python`
- `justfile` with `fmt`, `lint`, `test`, `bench`, `coverage`, `doc`, `ci`, `setup` recipes
- `lefthook.yml` — pre-commit (fmt/clippy/typos parallel) + pre-push (`just ci`)
- `cargo-deny` configuration (license allow-list, advisory deny, wildcard ban)
- GitHub Actions CI: check+lint, test matrix (stable/nightly), coverage, MSRV, doc deploy, doc-check
- `nextest` CI profile with retry on failure and extended timeouts

### Added — Phase 0: Project Foundation
- Initial repository structure and `CLAUDE.md` with Aeon Architecture specification
- Research corpus: technical design document, spec, competitor analysis, industry analysis
- Spec corrections based on external research: custom mmap arena (not safetensors), Vamana graph (not HNSW), Rust+CUDA (not pure Rust)
