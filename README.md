# TardigradeDB

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

- **Rust** ≥ 1.95 (edition 2024)
- **CUDA toolkit** (optional, for GPU DMA paths)

## Building

```bash
cargo build
```

Run tests:

```bash
cargo test --workspace
```

## Project Status

**Early development.** The crate structure and core type definitions are in place. See `docs/technical/tdd.md` for the full technical design document and `docs/technical/spec.md` for the condensed specification.

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
