# TardigradeDB

[![CI](https://github.com/Eldriss-Studio/tardigrade-db/actions/workflows/ci.yml/badge.svg)](https://github.com/Eldriss-Studio/tardigrade-db/actions/workflows/ci.yml)

**An experimental LLM-native memory-kernel prototype for autonomous AI agents.**

TardigradeDB is not a traditional database with tables and indexes, nor a vector DB with embeddings. It operates directly on the model's Key-Value (KV) cache tensors in latent space — memory is stored, retrieved, and organized as quantized neural activations, not text.

> **Research status (April 23, 2026): experimental prototype**
>
> TardigradeDB is a research experiment, not a production-ready database.
> Current results are from controlled demos, experiments, and benchmarks.
> APIs, on-disk formats, and guarantees may change while the architecture is validated.

> *From "storage as a service" to "storage as cognition."*

## Quick Links

- Experiments log: [docs/experiments/README.md](docs/experiments/README.md)
- Benchmark narrative: [https://eldriss-studio.github.io/tardigrade-db/dev/bench-v1/index.html](https://eldriss-studio.github.io/tardigrade-db/dev/bench-v1/index.html)
- Observed benchmark results so far: [https://eldriss-studio.github.io/tardigrade-db/dev/bench-v1/results.html](https://eldriss-studio.github.io/tardigrade-db/dev/bench-v1/results.html)

## Naming Trivia

The name *TardigradeDB* is a metaphor, and these are the development pillars behind it:

- **Cryptobiosis -> dormant memory revival**: quantized KV state can be persisted, then "reanimated" by retrieval and reinjection later.
- **Resilience under stress -> recovery-first design**: WAL + rebuildable derived state + fail-fast replay boundaries.
- **Tiny footprint -> compressed survival**: Q4/Q8 compression keeps memory practical under constrained capacity.
- **Adaptive survival -> memory lifecycle control**: AKL promotion/demotion/decay keeps useful memory active and stale memory fading.

## What The Prototype Demonstrates Today

At runtime, TardigradeDB acts like a memory engine for agents:

1. It stores model KV activations durably (`mem_write`) in quantized form.
2. It retrieves relevant past activations (`mem_read`) using latent-space scoring.
3. It tracks causal links and importance so memory can evolve over time.
4. It exposes the engine through Rust APIs and Python bindings.
5. It ships a comparable benchmark harness (`tdb_bench`) for transparent system-to-system evaluation.

If you only need one mental model: **capture memory state, persist it, and retrieve it later with attention-compatible relevance.**

## Why TardigradeDB?

Current agent memory systems (Mem0, Letta, Zep) rely on text retrieval — tokenize, embed, search, detokenize. This creates a lossy round-trip through representations the model never asked for. TardigradeDB eliminates that entirely by persisting the model's own internal state and restoring it directly into the attention stack.

### Quick comparison

| Dimension | Embedding RAG / vector memory | Traditional KV cache | TardigradeDB (experimental prototype) |
|-----------|-------------------------------|----------------------|---------------------------------------|
| Primary stored unit | Text chunks + embedding vectors | Attention K/V tensors for active context window | Quantized K/V tensors as durable memory cells |
| Retrieval signal | ANN/cosine similarity in embedding space | Usually none (append + replay only) | Attention-native latent scoring (`q · k / sqrt(d_k)`) |
| Semantic recall | Strong for text-level similarity | Not a retrieval system by itself | Designed for latent semantic recall directly on K/V |
| Persistence scope | External DB persistence is common | Usually process/session-local and ephemeral | Built for cross-session persistence in the engine |
| Context usage pattern | Retrieve text, then re-tokenize into prompt | Replay prior cache pages, often all-or-nothing | Retrieve and inject selected memory slices |
| Lifecycle/governance | App-defined policies, often ad hoc | None by default | AKL promotion/demotion/decay (prototype) |
| Causal/episodic structure | Optional metadata graph | None by default | Trace graph + WAL recovery model |
| Representation round-trip | text -> embed -> search -> text | none, but no semantic retrieval layer | native tensor path (no mandatory text/embedding round-trip) |
| Status | Mature ecosystem/pattern | Core inference primitive | Research prototype under active validation |

### "Isn't this just a KV cache?"

Yes at the data level; no at the system level.

A raw KV cache is append-and-replay state for one running model session. Compared with raw KV cache and text/embedding memory stacks, this prototype demonstrates:

- **Attention-native semantic retrieval** (`q · k / sqrt(d_k)`), so recall is based on latent similarity rather than text keyword overlap.
- **Selective memory injection**, fetching only relevant KV slices instead of replaying full history.
- **Durable compressed persistence** (Q4/Q8) across sessions/runs, not process-local ephemeral cache pages.
- **Memory lifecycle control** (AKL promotion, demotion, decay), avoiding unmanaged growth.
- **Causal/episodic structure + recovery model** (Trace + WAL + rebuildable derived state).
- **Cross-agent memory boundary** via a shared engine API, instead of one cache per process.
- **Native tensor path**, with no mandatory text -> embedding -> ANN -> text round-trip.

In short: TardigradeDB stores KV tensors, but is aiming to behave like a managed long-term memory kernel.
These are architectural differentiators under active validation, not final production claims.

### Real KV Cache Retrieval Result (April 23, 2026)

`experiments/sonia_real_kv_cache.py` compared storing hidden-state proxies vs storing actual KV cache key projections (`past_key_values`) on `Qwen/Qwen3-0.6B` across 16 diverse memories and 16 semantic queries.

| What was stored | Mean-pool recall | Per-token recall | Unique top-1 memories |
|-----------------|------------------|------------------|-----------------------|
| Hidden states (proxy, wrong target) | 31.2% | 31.2% | not reported in this run |
| Real KV cache (actual key projections) | 62.5% | 75.0% | 7 / 16 |

Key outcomes from that run:

- Switching from hidden states to real KV tensors doubled mean-pool recall (31.2% -> 62.5%).
- Per-token recall more than doubled (31.2% -> 75.0%).
- The top-1 "gravity well" collapsed: 7 different memories reached top-1 across 16 queries.
- Remaining misses were limited to 4/16 queries in that setup (smoke-alarm cooking, doctor visit, first snow, Coco movie).

Primary takeaway: the retrieval architecture was not the core failure mode; storing the wrong representation was.

**Measured prototype performance** (Apple M-series, release mode, criterion):

| Operation | Latency | Notes |
|-----------|---------|-------|
| INT8 dot product (NEON) | **5.3ns** | dim=128, 8.5x faster than FP32 |
| Q4 quantize + dequantize | **292ns** | dim=128, full round-trip |
| Engine `mem_read` | **119μs** | 1K cells, dim=64, full pipeline |
| SLB query (256 entries) | **3.8μs** | INT8, dim=128 |
| Block pool random read | **20μs** | from 10K cells on disk |
| Engine `mem_write` | **8.2ms** | single cell with fsync |
| Batch write (amortized) | **~80μs/cell** | 100-cell batch, single fsync |

**Architectural targets** (from spec, not yet all measured):
- **91–135x** prefill reduction via latent-space retrieval (MemArt)
- **4x** more agent contexts in fixed device memory via Q4 quantization
- **Up to 4.7x** TTFT reduction through cross-agent KV cache reuse (RelayCaching)

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

- **Rust** ≥ 1.95 (edition 2024) — MSRV enforced in CI (`MSRV (1.95)`), local toolchain tracks `stable` via `rust-toolchain.toml`
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
Quality:        fmt, fmt-fix, lint, deny, typos
Testing:        test, test-ci, test-crate <name>
Benchmarks:     bench, bench-crate <name>, bench-v1-smoke, bench-v1-full
Coverage:       coverage, coverage-lcov
Build:          build, release, doc
Fuzz:           fuzz <target>
CI-local:       ci
```

### Common workflows

```bash
just ci                  # Run full CI locally (fmt + lint + typos + test + deny + doc)
just bench               # Run criterion benchmarks (workspace, excludes tdb-python)
just bench-v1-smoke      # Run benchmark harness smoke profile + reports
just bench-v1-full       # Run benchmark harness full profile + reports
just coverage            # Generate HTML coverage report
just fuzz fuzz_q4_round_trip  # Fuzz Q4 quantization (requires nightly)
```

### Pre-commit hooks

Lefthook runs automatically on commit (fmt + clippy + typos) and push (full CI: fmt + lint + typos + test + deny + doc). Install with `lefthook install` or `just setup`.

### API Documentation

Rustdoc is built and deployed to GitHub Pages on every push to `main`:

**[eldriss-studio.github.io/tardigrade-db](https://eldriss-studio.github.io/tardigrade-db)**

Benchmark result pages:
- Criterion dashboard: **[.../dev/bench/index.html](https://eldriss-studio.github.io/tardigrade-db/dev/bench/index.html)**
- Benchmark v1 narrative + latest links: **[.../dev/bench-v1/index.html](https://eldriss-studio.github.io/tardigrade-db/dev/bench-v1/index.html)**
- Observed completed runs (sample/smoke + caveats): **[.../dev/bench-v1/results.html](https://eldriss-studio.github.io/tardigrade-db/dev/bench-v1/results.html)**

To build locally: `just doc` (output in `target/doc/`).

### CI

Five jobs run on every push and PR:

| Job | What it checks |
|-----|---------------|
| **Check & Lint** | `cargo fmt`, `clippy --pedantic`, `typos`, `cargo-deny` |
| **Test** | `cargo nextest` on Ubuntu + macOS |
| **Coverage** | `cargo-llvm-cov` with Codecov upload |
| **MSRV** | Verifies build on Rust 1.95 |
| **Documentation** | Rustdoc build with `-D warnings`, deployed to GitHub Pages on main |

## Quick Start

```bash
git clone https://github.com/Eldriss-Studio/tardigrade-db.git
cd tardigrade-db

# Build Rust workspace
cargo build --workspace

# Run all Rust tests (101 unit/acceptance + 6 doctests)
cargo nextest run --workspace --exclude tdb-python
cargo test --doc --workspace --exclude tdb-python

# Set up Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml

# Run 10 Python tests
pytest tests/python/ -v

# Run the GPT-2 end-to-end demo
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
python examples/e2e_demo.py
```

## End-to-End Demo Results

The `examples/e2e_demo.py` script proves TardigradeDB works with a real LLM:

```
[1] Loading GPT-2 model...
    Model: 12 layers, d_model=768

[3] Capture: 'The capital of France is'
    Written 12 cells (12 total)

[4] Retrieve: 'What is the main city of France'
    Layer 0: 5 cells (best=942.7470)
    Layer 1: 5 cells (best=1518.0767)
    Layer 2: 5 cells (best=10148.2637)

[5] Governance:
    Cell 0: importance=100.0, tier=Core
    Cell 1: importance=100.0, tier=Core

[6] Persistence:
    Before=12, After=12

SUCCESS
```

Two semantically related prompts find each other through **latent-space attention scoring** — the same dot product the transformer uses internally. No embedding model, no text search, no cosine similarity.

## Testing

### Rust (107 tests)

```bash
cargo nextest run --workspace --exclude tdb-python    # all unit/acceptance tests
cargo test --doc --workspace --exclude tdb-python     # all doctests
cargo clippy --workspace --all-targets -- -D warnings # pedantic lint
cargo fmt --all -- --check                            # format check
just test-crate tdb-storage                           # single crate
```

### Python (10 tests)

```bash
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml
pytest tests/python/ -v
```

### Benchmarks (criterion)

```bash
cargo bench --workspace --exclude tdb-python          # all benchmarks
cargo bench -p tdb-retrieval                          # SLB + SIMD dot product
cargo bench -p tdb-index                              # Vamana build + WAL throughput
cargo bench -p tdb-engine                             # end-to-end write/read
```

### Benchmarks (v1 harness)

```bash
# Run comparable smoke benchmark matrix (Tardigrade + Mem0 + Letta)
PYTHONPATH=python python -m tdb_bench run \
  --mode smoke \
  --repeat 3 \
  --config python/tdb_bench/config/default.json \
  --output target/bench-v1/smoke-run.json

# Full LoCoMo + LongMemEval run requires pinned dataset paths
export LOCOMO_DATA_PATH=/abs/path/to/locomo_phase1.jsonl
export LONGMEMEVAL_DATA_PATH=/abs/path/to/longmemeval_phase1.jsonl
PYTHONPATH=python python -m tdb_bench run \
  --mode full \
  --repeat 3 \
  --config python/tdb_bench/config/default.json \
  --output target/bench-v1/full-run.json

# Generate reports
PYTHONPATH=python python -m tdb_bench report \
  --input target/bench-v1/smoke-run.json \
  --format md \
  --output target/bench-v1/smoke-report.md

# Compare two runs
PYTHONPATH=python python -m tdb_bench compare \
  --baseline target/bench-v1/smoke-run.json \
  --candidate target/bench-v1/full-run.json \
  --format md \
  --output target/bench-v1/compare.md
```

### Test coverage by layer

| Layer | Crate | Tests | Coverage |
|-------|-------|-------|----------|
| Core types | `tdb-core` | 6 | Builder, SynapticBank dimensions, tier defaults |
| Storage | `tdb-storage` | 11 | Q4 round-trip, segment rollover, persistence, SynapticStore |
| Retrieval | `tdb-retrieval` | 17 | NEON INT8 dot product, SLB eviction, brute-force, owner filter |
| Organization | `tdb-index` | 18 | Vamana recall + incremental, trace chains, WAL recovery, concurrency |
| Governance | `tdb-governance` | 25 | Importance scoring, tier hysteresis, recency decay, sweep |
| Engine | `tdb-engine` | 22 | End-to-end write/read, state rebuild, SLB chain, Vamana activation, WAL replay, throughput |
| Docs | doctests | 8 | Crate-level usage examples |
| Python | pytest | 10 | PyO3 bindings, hook ABC, HF adapter, WriteDecision |

## Research Milestones Implemented

**All 11 planned prototype phases are implemented.** Current evidence: 117 tests passing and end-to-end demos/experiments working.

### Storage Layer — Custom from scratch, not a wrapper

- **Q4 group-wise 4-bit quantization** — Same scheme as llama.cpp's Q4_0. 4x compression vs FP32 with MSE < 0.01 on typical activations. Implemented as a Strategy pattern (`Quantizer` trait) so Q8/FP16 variants slot in without changing the storage layer.
- **Append-only segment files** — 256MB segments with binary length-prefixed records. Magic/version header validation. `sync_data()` on every write for crash durability. Partial record detection on recovery (truncated writes are silently discarded, not propagated).
- **Segment scanning recovery** — On open, segments are scanned to rebuild the `CellId → (segment, offset)` index. O(n) in cell count, not cell size — only the record header is read during recovery.
- **SynapticStore** — Parallel repository for LoRA adapters (FP16), same binary conventions as the cell block pool.

### Retrieval Layer — Latent-space attention, not text search

- **NEON SIMD INT8 dot product** — ARM aarch64 intrinsics (`vmull_s8` → `vpadalq_s16` → `vaddvq_s32`) with scalar fallback for x86. <1% relative error vs FP32 reference.
- **Semantic Lookaside Buffer (SLB)** — Fixed-capacity LRU cache storing keys in symmetric INT8 quantization. Exploits conversational locality — recently accessed cells are served at SIMD speed.
- **Brute-force attention scoring** — Exhaustive `score = (q · k) / √d_k` over all indexed keys. Validated by the MemArt paper: at per-agent scale (<10K blocks), brute-force SIMD matmul outperforms ANN indexes.
- **Three-stage retrieval chain** — SLB (hot, INT8) → Vamana (warm, graph ANN) → BruteForce (cold, exact). Chain of Responsibility pattern with deduplication by CellId.

### Organization Layer — Graph index + causal memory + crash recovery

- **Vamana graph index** — DiskANN-style single-layer graph with multi-seed greedy beam search. Robust pruning (angular diversity via α parameter) produces navigable graphs. Supports both batch build and incremental `insert_online`. Lazy activation: built only when cell count crosses a configurable threshold.
- **Trace causal graph** — Directed edges (CausedBy, Follows, Contradicts, Supports) with BFS transitive ancestor traversal. When a cell is written with a parent, the causal edge is recorded.
- **Write-Ahead Log** — Every Trace mutation is WAL-logged (with fsync) before being applied in memory. On engine open, WAL is replayed to rebuild the graph. Lenient crash recovery: partial records are discarded, not propagated.

### Governance Layer — Self-curating memory

- **Importance scoring (ι)** — Bounded [0, 100]. +3 on read access, +5 on write. Daily decay factor 0.995.
- **Maturity tiers with hysteresis** — Draft → Validated (ι ≥ 65, demote < 35) → Core (ι ≥ 85, demote < 60). The 30-point hysteresis gaps prevent oscillation near boundaries. Skip-tier promotion/demotion for large importance jumps.
- **Recency decay** — `r = exp(-Δt/30)` applied as a retrieval score multiplier. ~21-day half-life. Cells unused for a month have their relevance halved.

### Engine — Facade that ties it all together

- **Memento-pattern state rebuild** — On `Engine::open`, all in-memory state (retriever index, governance scores/tiers, ID counter) is reconstructed from durable sources (segments + WAL). The engine can crash at any point and recover.
- **Governance computed before persistence** — On-disk importance and tier match the in-memory state, ensuring correct rebuild.
- **Python bridge (PyO3)** — `tardigrade_db.Engine` class with `mem_write`, `mem_read`, `trace_ancestors`, `store_synapsis`, `load_synapsis`. Numpy arrays in, lists out.
- **Inference hook framework** — `TardigradeHook` ABC (Template Method pattern) with `HuggingFaceHook` reference implementation. Norm-based salience heuristic, automatic KV capture/retrieval.

### Proven in practice (proxy and real-KV paths)

The GPT-2 end-to-end demo validates the full persistence/retrieval loop with hidden-state proxy tensors for fast local verification.

The April 23, 2026 Qwen3 experiment (`experiments/sonia_real_kv_cache.py`) validates the same retrieval idea using actual KV cache key projections (`past_key_values`), with substantially higher recall (up to 75.0% per-token in that run).

---

## Roadmap

### Done

- [x] Custom binary storage with Q4 quantization and crash-safe segments
- [x] NEON SIMD INT8 dot product (ARM) with scalar fallback
- [x] Semantic Lookaside Buffer (LRU, INT8 quantized)
- [x] Brute-force latent-space attention retrieval
- [x] Vamana graph index with DiskANN robust pruning (batch + incremental)
- [x] Trace causal graph with WAL durability
- [x] Adaptive Knowledge Lifecycle (importance, tiers, decay)
- [x] Engine facade with Memento-pattern crash recovery
- [x] PyO3 Python bindings with numpy interop
- [x] HuggingFace inference hook (ABC + reference implementation)
- [x] SynapticBank (LoRA adapter) persistence
- [x] Criterion benchmarks across all subsystems
- [x] GPT-2 end-to-end demo
- [x] 117 tests (107 Rust + 10 Python)

### Next up

- [ ] **Real KV cache injection** — `torch.nn.Module` wrapper that modifies attention KV cache in-place with retrieved cells. Currently we capture and retrieve, but don't inject back into the model.
- [ ] **Batch write / group commit** — Buffer N writes, single fsync. Current per-write fsync (~5ms) limits throughput to ~200 writes/sec.
- [ ] **PyPI packaging** — `pip install tardigrade-db` via maturin publish.
- [ ] **Release-mode benchmark numbers** — Run `cargo bench`, publish actual performance data vs. spec targets.
- [ ] **Background governance sweep** — Tokio timer task for autonomous AKL decay/eviction instead of manual `advance_days()`.

### Future

- [ ] **vLLM / SGLang integration** — Adapter that hooks into their paged KV cache manager for production serving.
- [ ] **CUDA GPU DMA kernels** — Direct NVMe→GPU transfers via cuFile/GDS for GPU-resident inference.
- [ ] **Disk-aware Vamana** — PageANN-style page-node alignment for billion-scale cold storage on NVMe.
- [ ] **Multi-model dimension support** — Handle different models (different d_k) in one engine instance.
- [ ] **RelayCaching** — Cross-agent KV cache reuse for multi-agent handoffs.
- [ ] **Decoupled position encoding** — Safe injection of historical KV blocks with position ID remapping.

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
