# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

TardigradeDB is a from-scratch, LLM-native database kernel designed as a persistent memory system for autonomous AI agents. It is **not** a traditional database with tables/indexes, nor a vector DB with embeddings. It operates directly on the model's Key-Value (KV) cache tensors in latent space — memory is stored, retrieved, and organized as quantized neural activations, not text.

**Status:** Phase 0 complete (scaffold). Phase 1 (storage) next.

## Build & Test

```bash
cargo build --workspace          # build all crates
cargo test --workspace           # run all tests
cargo clippy --workspace -- -D warnings  # lint
cargo fmt --all -- --check       # format check
cargo test -p tdb-core           # test a single crate
cargo test test_name             # run a single test by name
```

## Crate Structure

Rust workspace with strict dependency ordering:

- **tdb-core** — Shared types: `MemoryCell`, `SynapticBankEntry`, `Tier`, error types. No dependencies on other tdb crates.
- **tdb-storage** — Block pool, mmap arena, Q4/Q8 quantization. Depends on tdb-core.
- **tdb-retrieval** — SLB, SIMD brute-force matmul, attention scoring. Depends on tdb-core, tdb-storage.
- **tdb-index** — Vamana graph (DiskANN-style), Trace causal graph, WAL. Depends on tdb-core, tdb-storage.
- **tdb-governance** — AKL: importance scoring, tier state machine, recency decay. Depends on tdb-core.
- **tdb-engine** — Top-level orchestrator, scheduler, batch cache. Depends on all above.
- **cuda/** — CUDA C++ kernels (attention, quantization), linked via cudarc FFI.

## Architecture (Aeon Architecture)

Four-layer system treating memory as a managed OS resource:

1. **Storage Layer** — Persistent quantized KV-cache block pool. 4-bit (Q4) quantized custom mmap arena (`safetensors` retained for import/export only). GPU DMA for direct NVMe→GPU transfers bypassing CPU. Decoupled position encoding for safe historical KV block reuse. Includes a sidecar blob arena (append-only, mmap-backed) with generational GC.

2. **Retrieval Layer** — Latent space attention, not text search. Multi-Token Aggregation (MemArt) computes attention scores directly against compressed keys in latent space (91-135x prefill reduction). Semantic Lookaside Buffer (SLB) for sub-5μs retrieval using INT8 scalar quantization with NEON SDOT intrinsics. RelayCaching reuses decode-phase KV caches across agents to cut TTFT by up to 4.7x.

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

See `.Codex/plans/whimsical-crafting-stonebraker.md` for the full research findings and implementation plan.

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
