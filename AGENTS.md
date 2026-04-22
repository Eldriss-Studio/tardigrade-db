# AGENTS.md

This file provides guidance to Codex and other AI coding agents when working with code in this repository.
It is kept in sync with `CLAUDE.md` — if the two files diverge, `CLAUDE.md` is canonical.

## Project Overview

TardigradeDB is a from-scratch, LLM-native database kernel designed as a persistent memory system for autonomous AI agents. It is **not** a traditional database with tables/indexes, nor a vector DB with embeddings. It operates directly on the model's Key-Value (KV) cache tensors in latent space — memory is stored, retrieved, and organized as quantized neural activations, not text.

**Status:** Phases 0–11 complete. 117 tests passing. All CI checks green.

## Build & Test

```bash
# Build
cargo build --workspace

# Test (use nextest — standard cargo test is not used)
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo nextest run --workspace

# Lint
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy --workspace --all-targets -- -D warnings

# Format check
cargo fmt --all -- --check

# Full local CI (mirrors GitHub Actions exactly)
just ci

# Single crate
just test-crate tdb-engine

# Python bindings (requires maturin)
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop
```

`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` is required for Python 3.14+ compatibility and is included in all `just` recipes.

## Crate Structure

Rust workspace with strict dependency ordering (no upward deps):

```
tdb-core  (no internal deps)
  └─► tdb-storage      (mmap arena, Q4 quantization)
  └─► tdb-governance   (AKL, importance scoring)
  └─► tdb-index        (Vamana, Trace graph, WAL)  → tdb-storage
  └─► tdb-retrieval    (SLB, brute-force)           → tdb-storage
        └─► tdb-engine (facade, orchestration)      → all above
              └─► tdb-python (PyO3 bindings)
```

- **tdb-core** — Shared types: `MemoryCell`, `Tier`, `TardigradeError`. No deps on other tdb crates.
- **tdb-storage** — Block pool, mmap arena, Q4/Q8 quantization. Depends on tdb-core.
- **tdb-retrieval** — SLB (INT8 hot cache), SIMD brute-force matmul, attention scoring. Depends on tdb-core, tdb-storage.
- **tdb-index** — Vamana graph (DiskANN-style), Trace causal graph, WAL. Depends on tdb-core, tdb-storage.
- **tdb-governance** — AKL: importance scoring, tier state machine, recency decay. Depends on tdb-core.
- **tdb-engine** — Top-level orchestrator, batch cache. Depends on all above.
- **tdb-python** — PyO3 bindings exposing `Engine` and `ReadResult` to Python.

## Architecture (Aeon Architecture)

Four-layer system treating memory as a managed OS resource:

1. **Storage Layer** — Persistent quantized KV-cache block pool. Q4 (GGML Q4_0) custom mmap arena. `safetensors` retained for import/export only. Decoupled position encoding for safe historical KV block reuse.

2. **Retrieval Layer** — Latent space attention, not text search. Scaled dot-product: `score(q,k) = q·k / √d_k`. Two-level: INT8 SLB hot path (sub-5μs) → brute-force cold path (exact, valid ≤10K cells) → Vamana ANN (10K+ cells).

3. **Organization Layer** — Neuro-symbolic dual topology. Vamana: SIMD graph ANN for cold-path retrieval. Trace: episodic causal graph with WAL-backed crash recovery.

4. **Governance Layer** — Adaptive Knowledge Lifecycle (AKL). Importance scoring (ι ∈ [0,100], +3 access / +5 update, ×0.995 daily decay). Three maturity tiers with hysteresis: Draft→Validated at ι≥65 (demote <35), Validated→Core at ι≥85 (demote <60). Recency decay: r = exp(−Δt/τ), τ=30 days (~21-day half-life).

## Key Technical Specs

- **Target perf:** P99 read latency 750ns under 16-thread contention; 3.09μs Vamana traversal at 100K nodes
- **Quantization:** Q4 group-wise (GGML Q4_0): 8×f32 → 4×u8 + f16 scale ≈ 5.3x compression
- **Python bridge:** Zero-copy NumPy `float32` slices via PyO3 `PyReadonlyArray1`

## Key Documents

- `docs/technical/tdd.md` — Full technical design document (Aeon Architecture)
- `docs/technical/spec.md` — Condensed specification of the four layers
- `docs/refs/summary.md` — Project summary and value proposition
- `docs/refs/AI-db-discussion.md` — Design evolution from traditional DB → KV-native approach
- `docs/refs/AI Agentic Memory System Efficiency.md` — Comprehensive industry analysis
- `docs/competitors/competitors-search-1.md` — Direct competitors by architectural pillar
- `docs/competitors/competitors-search-2.md` — KV-cache and weights-as-memory research landscape

Plans are stored in `.claude/plans/`, not `.Codex/plans/`.

## Competitive Positioning

No existing project unifies all three of: persistent KV slices as semantic memory, per-user/agent trainable adapter banks, and a unified memory OS API. Closest overlaps by pillar:

- **Persistent KV reuse:** LMCache, LRAgent, Kelle — infrastructure-level KV sharing, not cognitive/agentic
- **Latent retrieval:** MemArt — computes attention in latent space but not a full memory engine
- **Agent-native lifecycle:** ByteRover 2.0 — AKL-style governance but operates on text/files, not tensors
- **Causal organization:** Genesys-Memory — causal graph with pruning, but text-based, not KV-native
- **Weights-as-memory:** MemoryLLM (Apple), PRIME — research exploring FFNs as memory, not yet productized
- **DB-as-runtime:** SpaceTimeDB — WASM reducers inside the kernel, relevant execution model

TardigradeDB's differentiator is unifying persistent quantized KV storage + latent-space retrieval + neuro-symbolic organization + adaptive lifecycle into a single kernel.

## Spec Corrections (from research, April 2026)

The original TDD/spec documents contain three assumptions that were invalidated by external research. The implementation follows these corrections:

1. **Storage format: Custom mmap'd arena, NOT safetensors.** safetensors has a 100MB header cap, no append, no in-place update. No production KV cache system uses it. safetensors is kept only for import/export.
2. **Index: Vamana graph (DiskANN-style), NOT HNSW + B+ tree.** MemArt uses brute-force matmul. HNSW fails for attention retrieval due to Q/K distribution shift. Hot path = SIMD brute-force matmul. Cold path = Vamana graph.
3. **Language: Rust core + CUDA C++ kernels.** Rust for storage engine, crossbeam-epoch for concurrency, PyO3 for Python bindings. CUDA C++ for GPU compute (attention, quantization, DMA) linked via cudarc.

## Reliability & Consistency Rules (Canonical)

These rules are mandatory for all storage/retrieval/index/engine changes:

- **Durability contract required:** Every design/commit touching write paths must define the durability boundary and how `durable_offset` advances.
- **Consistency mode declaration required:** Any API or externally visible read/update behavior must explicitly declare `confirmed` vs `unconfirmed` semantics.
- **Recovery contract required:** Any change to WAL/snapshot/replay must document crash boundaries and recovery behavior.
- **Derived-state rebuildability required:** Indexes, caches, and materialized/derived state must be reconstructable from durable history + snapshots.
- **Fail-fast replay required:** Replay inconsistencies are hard errors; do not serve reconstructed-but-untrusted state.

Minimum acceptance tests for durability/recovery changes:

- Crash during append/write path.
- Crash between WAL commit and snapshot capture.
- Snapshot restore + WAL suffix replay correctness.
- Confirmed-read visibility (`durable_offset >= tx_offset`) when enabled.

Mandatory metrics updates for durability/recovery changes:

- Replay total time.
- Snapshot read/restore time.
- WAL replay count/time.
- Durability queue depth/backlog.

## Design Principles

- **Tensor-native:** The primary stored unit is a KV cache tensor, not text or embeddings. Reads inject pre-computed K/V directly into the attention stack. Writes capture internal activations — no tokenization/detokenization round-trip.
- **No external dependencies:** No Postgres, Neo4j, or vector DB. Custom storage engine with custom indices.
- **Latent-space retrieval:** Relevance is computed via attention in latent space, not cosine similarity over embeddings from a separate model.
- **Self-curating:** The AKL algorithm autonomously manages memory lifecycle — promotion, demotion, and decay are built into the engine, not left to application code.

## Development Guidelines

- **ATDD-first:** Every fix or feature starts with Acceptance Test-Driven Development. Write the failing acceptance test that defines "done" before writing any implementation code.
- **SOLID & Clean Code:** All code follows SOLID principles and Clean Code standards. Prefer small, focused files and functions with single responsibilities. Break complex modules into smaller, digestible parts.
- **Design patterns:** Solve problems using well-known engineering design patterns. Choose the pattern that fits the problem — don't force-fit, but don't ad-hoc either. Name and document the pattern in use.
- **Mandatory refactor pass:** No work is considered complete until a final refactoring pass has been done for code quality — naming, structure, duplication, and adherence to SOLID.
- **Mandatory gap review:** No work is considered complete until a review for gaps has been performed — missing edge cases, untested paths, incomplete error handling, and architectural blind spots.

## Documentation Standards

- **First-class open source quality required.** Follow the standard set by tokio, serde, crossbeam, rocksdb. Documentation that merely restates what the code does is not acceptable.
- **Crate-level `//!` docs are mandatory** in every `lib.rs`. They must include: a one-sentence summary, what problem this crate solves, an ASCII diagram where helpful, key public types linked with intra-doc links, and at least one working `# Usage` code example.
- **Module-level `//!` docs** must explain *why* the module exists — design decisions and trade-offs belong here.
- **Item-level `///` docs** must explain invariants, preconditions, panics, and non-obvious behavior. Do not repeat the function signature in prose.
- **Challenge assumptions externally:** Before documenting a design choice, verify it against published research or first-class OSS references.

## What Not To Do

- Do not add `#[allow(...)]` annotations — fix the root cause.
- Do not use `println!` / `eprintln!` / `dbg!` in library code.
- Do not add `unsafe` blocks without a `// SAFETY:` comment.
- Do not write backwards-compatibility shims for removed code. Delete it completely.
- Do not add speculative features beyond what the current task requires.
- Do not create branches or pull requests. Push directly to `main` after `just ci` passes.
