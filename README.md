# TardigradeDB

[![CI](https://github.com/Eldriss-Studio/tardigrade-db/actions/workflows/ci.yml/badge.svg)](https://github.com/Eldriss-Studio/tardigrade-db/actions/workflows/ci.yml)

**An LLM-native database kernel — persistent memory for autonomous AI agents.**

TardigradeDB is not a traditional database with tables and indexes, nor a vector DB with embeddings. It operates directly on the model's Key-Value (KV) cache tensors in latent space — memory is stored, retrieved, and organized as quantized neural activations, not text.

> *From "storage as a service" to "storage as cognition."*

## Why TardigradeDB?

Current agent memory systems (Mem0, Letta, Zep) rely on text retrieval — tokenize, embed, search, detokenize. This creates a lossy round-trip through representations the model never asked for. TardigradeDB eliminates that entirely by persisting the model's own internal state and restoring it directly into the attention stack.

**Key results:**
- **91–135x** prefill reduction via latent-space retrieval (MemArt)
- **Sub-5μs** cache-hit retrieval via the Semantic Lookaside Buffer (SLB)
- **Up to 4.7x** TTFT reduction through cross-agent KV cache reuse (RelayCaching)
- **4x** more agent contexts in fixed device memory via Q4 quantization
- **750ns** P99 read latency under 16-thread contention

## Architecture (Aeon)

Four-layer system treating memory as a managed OS resource:

```
┌─────────────────────────────────────────────────────┐
│  Governance    Adaptive Knowledge Lifecycle (AKL)    │
│                importance scoring · maturity tiers    │
│                recency decay · self-curation          │
├─────────────────────────────────────────────────────┤
│  Organization  Atlas Index (SIMD vector index)       │
│                Trace (causal episodic graph)          │
│                WAL · epoch-based reclamation          │
├─────────────────────────────────────────────────────┤
│  Retrieval     MemArt (latent-space attention)       │
│                SLB (INT8 scalar quantization)         │
│                RelayCaching (cross-agent KV reuse)    │
├─────────────────────────────────────────────────────┤
│  Storage       Q4 KV-cache block pool                │
│                custom mmap arena · GPU DMA            │
│                blob arena · decoupled position IDs    │
└─────────────────────────────────────────────────────┘
```

## Crate Map

| Crate | Layer | Responsibility |
|-------|-------|----------------|
| `tdb-core` | — | Shared types, error definitions, memory cell & synaptic bank primitives |
| `tdb-storage` | Storage | Quantized block pool, mmap arena, segment management |
| `tdb-retrieval` | Retrieval | Latent-space attention, INT8 quantization, SLB, SIMD distance |
| `tdb-index` | Organization | Vamana graph index, causal trace, write-ahead log |
| `tdb-governance` | Governance | Importance scoring, maturity tiers, temporal decay |
| `tdb-engine` | Orchestrator | Batch KV cache, prefill/decode scheduler, top-level engine |

## Requirements

- **Rust** ≥ 1.85 (edition 2024) — pinned via `rust-toolchain.toml`
- **[just](https://github.com/casey/just)** — task runner (`cargo install just`)
- **[lefthook](https://github.com/evilmartians/lefthook)** — git hooks (`brew install lefthook`)
- **CUDA toolkit** (optional, for GPU DMA paths)
- **Nightly toolchain** (optional, for fuzzing: `rustup toolchain install nightly`)

## Getting Started

```bash
# Clone and set up git hooks
git clone https://github.com/Eldriss-Studio/tardigrade-db.git
cd tardigrade-db
lefthook install

# Build and test
just build
just test
```

## Development

Run `just` to see all available recipes:

```
Quality:     fmt, fmt-fix, lint, deny, typos
Testing:     test, test-ci, test-crate <name>
Benchmarks:  bench, bench-crate <name>
Coverage:    coverage, coverage-lcov
Build:       build, release, doc
Fuzz:        fuzz <target>
CI-local:    ci
```

### Common workflows

```bash
just ci                  # Run full CI locally (fmt + lint + typos + test + deny)
just bench               # Run criterion benchmarks with native CPU opts
just coverage            # Generate HTML coverage report
just fuzz fuzz_q4_round_trip  # Fuzz Q4 quantization (requires nightly)
```

### Pre-commit hooks

Lefthook runs automatically on commit (fmt + clippy + typos) and push (tests). Install with `lefthook install`.

### API Documentation

Rustdoc is built on every push to `main` and deployed to GitHub Pages when available:

**[eldriss-studio.github.io/tardigrade-db](https://eldriss-studio.github.io/tardigrade-db)**

To build locally: `just doc` (output in `target/doc/`).

### CI

Five jobs run on every push and PR:

| Job | What it checks |
|-----|---------------|
| **Check & Lint** | `cargo fmt`, `clippy --pedantic`, `typos`, `cargo-deny` |
| **Test** | `cargo nextest` on Ubuntu + macOS |
| **Coverage** | `cargo-llvm-cov` with Codecov upload |
| **MSRV** | Verifies build on Rust 1.85 |
| **Documentation** | Rustdoc build with `-D warnings` (deploys to Pages on main, lint-only on PRs) |

## Project Status

**Active development.** The core architecture is implemented across all four layers. Current focus is hardening the storage layer for production use.

| Phase | Status |
|-------|--------|
| Phase 0 — Scaffold | Complete |
| Phase 1 — Storage (block pool, quantization, segments) | Complete |
| Phase 2 — Retrieval (SLB, SIMD dot-product, INT8 quantization) | Complete |
| Phase 3 — Organization (Vamana index, Trace graph, WAL) | Complete |
| Phase 4 — Storage hardening & integration | Next |

See `docs/technical/tdd.md` for the full technical design document and `docs/technical/spec.md` for the condensed specification.

## Reliability & Consistency Contracts

- **Durability boundary** — Writes are accepted asynchronously and durability is tracked via a monotonic `durable_offset`. Client-visible guarantees are defined against this boundary.
- **Read visibility modes** — Two explicit modes exist:
  - `unconfirmed`: lowest-latency path, can expose state before durability is confirmed.
  - `confirmed`: waits until `durable_offset >= tx_offset` before exposing results/updates (default for safety-critical paths).
- **Recovery pipeline** — Startup follows: restore latest valid snapshot -> replay WAL/commitlog suffix -> rebuild derived in-memory state.
- **Derived state policy** — Indexes, caches, and other derived structures must be rebuildable from durable history and snapshots.
- **Replay failure policy** — Replay runs fail-fast on inconsistency. The engine must not serve reconstructed-but-untrusted state.
- **Observability requirements** — Replay/snapshot timing, replay counts, and durability queue depth are first-class metrics.

## Design Principles

- **Tensor-native** — The primary stored unit is a KV cache tensor. Reads inject pre-computed K/V directly into the attention stack. No tokenization round-trip.
- **Zero external dependencies** — No Postgres, Neo4j, or vector DB. Custom storage engine with custom indices.
- **Latent-space retrieval** — Relevance via attention in latent space, not cosine similarity over external embeddings.
- **Self-curating** — The AKL algorithm autonomously manages promotion, demotion, and decay. No application-level memory management.

## License

MIT
