# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TardigradeDB is a from-scratch, LLM-native database kernel designed as a persistent memory system for autonomous AI agents. It is **not** a traditional database with tables/indexes, nor a vector DB with embeddings. It operates directly on the model's Key-Value (KV) cache tensors in latent space — memory is stored, retrieved, and organized as quantized neural activations, not text.

**Status:** All implementation phases complete + P1/P2 architectural unification. 450+ tests (249 Rust + 201+ Python). **Active governance:** tier-based retrieval boost (Core 1.25×, Validated 1.1×) and `evict_draft_packs()`. WAL checkpointing. Text storage unified in Rust `TextStore`. `Engine::status()` for monitoring. Configurable engine from Python. Pluggable retrieval key strategies (3). KV injection verified with fully synthetic gibberish facts (9/10 recall on Qwen3-0.6B). `KnowledgePackStore` is the canonical injection path. Docs rewritten to match implementation — unimplemented claims moved to "Future Work."

## Build & Test

### Prerequisites

```bash
# Rust (1.95+, edition 2024)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python (3.13+ recommended, 3.14 works with ABI3 compat)
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest

# Optional: for the GPT-2 end-to-end demo
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

### Rust tests (238 tests)

```bash
cargo build --workspace                              # build all crates
cargo test --workspace --exclude tdb-python           # run all Rust tests
cargo clippy --workspace --exclude tdb-python -- -D warnings  # lint (pedantic)
cargo fmt --all -- --check                            # format check
cargo test -p tdb-core                                # test a single crate
cargo test -p tdb-engine --test acceptance            # run engine acceptance tests only
cargo test test_rebuild_retriever                     # run a single test by name
```

Note: `tdb-python` is excluded from `cargo test/clippy` because PyO3 needs `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` on Python 3.14.

### Python tests (145 tests)

```bash
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml
pytest tests/python/ -v -m "not gpu"        # CPU-only (safe everywhere)
pytest tests/python/ -v -m gpu              # vLLM round-trip (Linux + GPU + vLLM ≥ 0.19)
```

### Benchmarks

```bash
cargo bench --workspace --exclude tdb-python          # run all criterion benchmarks
cargo bench -p tdb-retrieval                          # SLB + dot product benchmarks
cargo bench -p tdb-index                              # Vamana build + WAL benchmarks
cargo bench -p tdb-engine                             # end-to-end write/read benchmarks
```

### End-to-end demo (GPT-2)

```bash
source .venv/bin/activate
python examples/e2e_demo.py
```

Captures KV cache from GPT-2 inference on *"The capital of France is"*, then retrieves semantically related cells when querying with *"What is the main city of France"* — proving latent-space retrieval works with a real model.

### Test breakdown by phase

| Layer | Crate | Tests | What's covered |
|-------|-------|-------|----------------|
| Core | tdb-core | 6 | Builder, SynapticBank, KVPack types, tier defaults |
| Storage | tdb-storage | 33 | Q4 round-trip, segment rollover, persistence, SynapticStore, TextStore (single + batch), DeletionLog |
| Retrieval | tdb-retrieval | 51 | Per-token Top5Avg, SLB eviction, pipeline (+ clear_stages), SIMD dot product, owner filter, PerTokenConfig |
| Organization | tdb-index | 23 | Vamana recall + incremental, trace chains, WAL recovery, concurrency |
| Governance | tdb-governance | 26 | Importance scoring, tier hysteresis, recency decay, sweep |
| Engine | tdb-engine | 109 | Write/read, pack API, text storage (single + batch), delete, state rebuild, SLB chain, Vamana activation, list_packs, refresh pipeline rebuild, WAL checkpointing, active governance (tier boost + eviction) |
| Python | pytest | 194 | PyO3 bindings, hook ABC, HF KV hook, per-token encoding, KV pack, MCP tools, diagnostics, RAG baseline, vLLM connector format/load-path/integration (with `-m gpu`), synthetic-fact KV injection (7), prefix builder (11), vLLM prefix client (13), vLLM prefix e2e (4 GPU), retrieval key strategy (7), docstring contract (1) |

Per-crate counts include unit + acceptance + doctest tests. Sum: 6+33+51+23+26+109+194 = 442.

## Crate Structure

Rust workspace with strict dependency ordering:

- **tdb-core** — Shared types: `MemoryCell`, `KVPack`, `SynapticBankEntry`, `Tier`, error types. No dependencies on other tdb crates.
- **tdb-storage** — Block pool, mmap arena, Q4/Q8 quantization. Depends on tdb-core.
- **tdb-retrieval** — Per-token retrieval (Top5Avg), SLB, SIMD brute-force matmul, retriever pipeline. Depends on tdb-core, tdb-storage.
- **tdb-index** — Vamana graph (DiskANN-style), Trace causal graph, WAL. Depends on tdb-core, tdb-storage.
- **tdb-governance** — AKL: importance scoring, tier state machine, recency decay. Depends on tdb-core.
- **tdb-engine** — Top-level orchestrator, scheduler, batch cache. Depends on all above.
- **cuda/** — CUDA C++ kernels (attention, quantization), linked via cudarc FFI.

## Architecture (Aeon Architecture)

Four-layer system treating memory as a managed OS resource:

1. **Storage Layer** — Persistent quantized KV-cache block pool. 4-bit (Q4) quantized custom mmap arena (`safetensors` retained for import/export only). GPU DMA for direct NVMe→GPU transfers bypassing CPU. Decoupled position encoding for safe historical KV block reuse. Includes a sidecar blob arena (append-only, mmap-backed) with generational GC.

2. **Retrieval Layer** — Per-token latent-space scoring, not text search. PerTokenRetriever with Top5Avg scoring computes dot products between individual query and memory hidden-state tokens (100% recall at 100 memories). Semantic Lookaside Buffer (SLB) for sub-5μs retrieval using INT8 scalar quantization with NEON SDOT intrinsics. Three-stage pipeline: SLB → PerTokenRetriever → BruteForceRetriever.

3. **Organization Layer** — Neuro-symbolic dual topology. Atlas Index: SIMD-accelerated Page-Clustered Vector Index (small-world graph + B+ Tree disk locality). Trace: episodic graph tracking causal relationships between KV blocks. Decoupled WAL for crash recovery with <1% overhead. Epoch-based reclamation for lock-free concurrent reads.

4. **Governance Layer** — Adaptive Knowledge Lifecycle (AKL). Importance scoring (ι ∈ [0,100], +3 access / +5 update, 0.995 daily decay). Three maturity tiers with hysteresis: draft→validated at ι≥65 (demote <35), validated→core at ι≥85 (demote <60). Recency decay: r = exp(-Δt/τ), τ=30 days (~21-day half-life).

## Key Technical Specs

- **Target perf:** 3.09μs tree traversal at 100K nodes, P99 read latency 750ns under 16-thread contention
- **Concurrency:** BatchQuantizedKVCache for concurrent Q4 inference; interleaved prefill/decode scheduling hides 500ms warm-reload latency
- **Bridge:** Zero-copy C++/Python bridge between inference engine and memory kernel

## Key Documents

- `docs/technical/tdd.md` — Full technical design document (Aeon Architecture)
- `docs/technical/spec.md` — Condensed specification of the four layers
- `docs/refs/summary.md` — Project summary and value proposition
- `docs/refs/AI-db-discussion.md` — Design evolution from traditional DB → KV-native approach
- `docs/refs/AI Agentic Memory System Efficiency.md` — Comprehensive industry analysis (Mem0, Letta, Zep, ByteRover benchmarks, economics)
- `docs/competitors/competitors-search-1.md` — Direct competitors by architectural pillar (MemArt, RelayCaching, Aeon, ByteRover, Letta, Genesys, SpaceTimeDB)
- `docs/competitors/competitors-search-2.md` — KV-cache and weights-as-memory research landscape (LMCache, LRAgent, FwPKM, MemoryLLM, PRIME)

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

1. **Storage format: Custom mmap'd arena, NOT safetensors.** safetensors has a 100MB header cap (~833K tensors), no append, no in-place update, O(n) header parse. No production KV cache system uses it. safetensors is kept only for import/export.
2. **Index: Vamana graph (DiskANN-style), NOT HNSW + B+ tree.** MemArt (the paper the spec cites) uses brute-force matmul, not ANN. HNSW fails for attention retrieval due to Q/K distribution shift. Hot path = SIMD brute-force matmul. Cold path = page-node Vamana graph (PageANN-inspired).
3. **Language: Rust core + CUDA C++ kernels.** Rust for storage engine, concurrency (crossbeam-epoch), Python bindings (PyO3). CUDA C++ for GPU compute (attention, quantization, DMA) linked via cudarc. Same pattern as TiKV and Neon.

See `.claude/plans/whimsical-crafting-stonebraker.md` for the full research findings and implementation plan.

## Reliability & Consistency Rules (Canonical)

These rules are mandatory for all storage/retrieval/index/engine changes:

- **Durability contract required:** Every design/PR touching write paths must define the durability boundary and how `durable_offset` advances.
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

- **SOLID & Clean Code:** All code follows SOLID principles and Clean Code standards. Prefer small, focused files and functions with single responsibilities. Break complex modules into smaller, digestible parts.
- **ATDD-first:** Every fix or feature starts with Acceptance Test-Driven Development. Write the failing acceptance test that defines "done" before writing any implementation code.
- **Design patterns:** Solve problems using well-known engineering design patterns. Choose the pattern that fits the problem — don't force-fit, but don't ad-hoc either. Name and document the pattern in use.
- **Mandatory refactor pass:** No work is considered complete until a final refactoring pass has been done for code quality — naming, structure, duplication, and adherence to SOLID.
- **Mandatory gap review:** No work is considered complete until a review for gaps has been performed — missing edge cases, untested paths, incomplete error handling, and architectural blind spots.

## Documentation Standards

- **First-class open source quality required.** Follow the standard set by top Rust crates (tokio, serde, crossbeam, rocksdb). Documentation that merely restates what the code does is not acceptable.
- **Crate-level `//!` docs are mandatory** in every `lib.rs`. These are the landing pages in `cargo doc`. They must include:
  - A one-sentence summary (appears in workspace index).
  - What problem this crate solves within TardigradeDB's 4-layer architecture.
  - An ASCII text diagram where the crate's architecture benefits from visualization.
  - Key public types/functions linked with `[Type]` intra-doc links.
  - At least one working `# Usage` code example for the most important public API.
  - Cross-references to related crates.
- **Module-level `//!` docs** must explain *why* the module exists, not just what it contains. Design decisions and trade-offs belong here.
- **Item-level `///` docs** must explain invariants, preconditions, panics, and non-obvious behavior. Do not repeat the function signature in prose.
- **Challenge assumptions externally:** Before documenting a design choice, verify it against published research or first-class OSS references. Never document a claim that hasn't been validated.
