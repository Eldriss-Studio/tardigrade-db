# TardigradeDB

[![CI](https://github.com/Eldriss-Studio/tardigrade-db/actions/workflows/ci.yml/badge.svg)](https://github.com/Eldriss-Studio/tardigrade-db/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tardigrade-db.svg)](https://pypi.org/project/tardigrade-db/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-eldriss--studio.github.io-blue)](https://eldriss-studio.github.io/tardigrade-db)
[![Rust 1.95+](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](rust-toolchain.toml)

> **TardigradeDB v0.7.0 is a research-grade preview.** Public APIs are stable; benchmark methodology is under active validation.

A persistent KV-cache memory engine for LLMs. Store the model's own internal attention state, retrieve it later with attention-native scoring, and inject it back into the model without spending prompt tokens. Built from scratch in Rust with PyO3 Python bindings.

## Hero example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tardigrade_db import Engine
from tardigrade_hooks import CalibrationRegistry, KnowledgePackStore

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16
).to("cuda")

engine = Engine("./memory")
kps = KnowledgePackStore(engine, model, tok, owner=1,
                        calibration_registry=CalibrationRegistry())

# Store a fact — captures the model's KV cache through Q4 quantization
kps.store("User prefers morning meetings")

# Retrieve it later — the model "remembers" without the text in the prompt
text, prompt_tokens, had_memory = kps.generate("When should we meet?")
# prompt_tokens is ~46% lower than the equivalent text-RAG path,
# output is byte-identical.
```

## Why TardigradeDB?

Embedding RAG asks an LLM to read text retrieved by a *different* model. TardigradeDB skips the text round-trip: it stores the LLM's own hidden-state tensors and reinjects them directly into attention. The model searches its own memories using its internal activations — no translator model, no prompt tokens consumed on injection.

**Use TardigradeDB when** you need persistent memory across LLM sessions and care about latency / context-window cost. **Reach for embedding RAG when** your problem is "find the right document chunk and paste it into a prompt" — embedding RAG is more mature and stronger at vague-query text retrieval today.

See [`docs/positioning.md`](docs/positioning.md) for the full comparison vs embedding RAG and traditional KV cache.

## Features

| Capability | Status | Where to learn more |
|------------|--------|----------------------|
| KV pack write / read (atomic multi-layer, Q4 quantized) | Stable | [`docs/architecture.md`](docs/architecture.md) |
| HuggingFace direct injection (`KnowledgePackStore`) | Stable | [`docs/guide/knowledge-pack-store.md`](docs/guide/knowledge-pack-store.md) |
| vLLM KV Connector v1 (prefix-cache acceleration) | Stable | [`docs/guide/vllm-setup.md`](docs/guide/vllm-setup.md) |
| Per-model retrieval-key calibration | Stable | [`docs/guide/calibration.md`](docs/guide/calibration.md) |
| Hybrid-attention support (RecurrentGemma, Jamba, Granite-4, …) | Stable | [`docs/guide/calibration.md`](docs/guide/calibration.md) |
| Multi-agent / multi-owner isolation | Stable | [`docs/guide/consumers.md`](docs/guide/consumers.md) |
| Adaptive Knowledge Lifecycle (importance, tiers, decay) | Stable | [`docs/architecture.md#governance-layer`](docs/architecture.md) |
| Portable snapshot + labeled checkpoints | Stable | [`docs/architecture.md`](docs/architecture.md) |
| `TardigradeClient` facade (chunking + ingestion + consolidation) | Stable | [`docs/guide/python-api.md`](docs/guide/python-api.md) |
| HTTP / REST bridge | Stable | [`python/tardigrade_http/`](python/tardigrade_http/) |
| LoCoMo / LongMemEval benchmark methodology | Under validation | [`docs/research-log.md`](docs/research-log.md) |

## Performance snapshot

Measured on a 5K-cell synthetic corpus with 1024-dim keys (matches Qwen3-0.6B hidden size) unless noted.

| Metric | Number | Source |
|--------|--------|--------|
| Retrieval latency, 5K cells | **p50 = 0.34 ms, p99 = 0.51 ms** | [`experiments/latency_benchmark_v2.py`](experiments/latency_benchmark_v2.py) |
| Per-cell on-disk footprint, 5K cells | **751 B** | [`experiments/footprint_audit.py`](experiments/footprint_audit.py) |
| Recall @ 100 memories, real Qwen3 keys | **100 %** (Top5Avg, Q4 pipeline) | [`docs/research-log.md`](docs/research-log.md) |
| `Engine.compute_retrieval_key('last_token')` | **6.5 µs** at prompt_len 1024 (14× over numpy) | [`experiments/retrieval_key_microbench.py`](experiments/retrieval_key_microbench.py) |
| `tardigrade_db.paged_to_flat` | **68 µs** at Qwen3-0.6B dims (2.1× over numpy) | [`experiments/kv_reshape_microbench.py`](experiments/kv_reshape_microbench.py) |

For the full positioning narrative, see [`docs/positioning/latency_first.md`](docs/positioning/latency_first.md).

## Architecture

Four-layer system treating memory as a managed OS resource.

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

Storage is a custom mmap arena with Q4 quantization, not safetensors. Retrieval is brute-force SIMD matmul at < 10 K blocks (per the MemArt paper), Vamana graph at larger scales. Organization is a DiskANN-style index + a Trace causal graph + a WAL for crash recovery. Governance is the AKL state machine: cells get promoted to higher tiers as they're accessed, decay over time when they aren't.

Full design: [`docs/architecture.md`](docs/architecture.md).

## Quick Start

### Python users — install from PyPI

```bash
pip install tardigrade-db
# For HuggingFace injection examples:
pip install transformers torch
```

Then drop the hero example into a Python file and run it.

### Rust contributors — build from source

```bash
git clone https://github.com/Eldriss-Studio/tardigrade-db.git
cd tardigrade-db
lefthook install
just ci        # fmt + lint + typos + test + deny + doc
```

Python bindings:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop
pytest tests/python/ -v -m "not gpu"
```

End-to-end GPT-2 demo (validates the full persistence / retrieval loop):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
python examples/e2e_demo.py
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor workflow (CI gates, benchmarks, MSRV, reliability contracts).

## Project Status

**Current version:** v0.7.0. Public APIs (Python + Rust) are stable for the 0.x series; breaking changes ride minor bumps per the semver pre-1.0 convention.

**Stable surfaces:**
- The PyO3 Python API on `tardigrade_db.Engine` and the `tardigrade_hooks` consumer modules.
- The Rust `tdb-engine` crate and supporting workspace crates.
- Portable snapshot / restore (tar archive with magic, codec identifiers, SHA-256).

**Under active validation:**
- LoCoMo / LongMemEval benchmark methodology — earlier headline numbers (68.2 % LoCoMo / 90.9 % LongMemEval) were **retracted on 2026-05-14** after an audit found the runs measured the lexical fallback adapter on a corpus corrupted by a dataset-prep bug. Honest native-engine number on clean LoCoMo: ~36 % R@1 at 50-item scale; full-corpus re-measurement pending. Synthetic-corpus results (100 % recall at 5K, vague-query refinement, KV injection on gibberish facts, cross-model retrieval) are unaffected. Full record: [`docs/experiments/2026-05-14-bench-audit.md`](docs/experiments/2026-05-14-bench-audit.md).
- GPU integration paths to production (HuggingFace direct injection works; vLLM custom-attention plugin is future work). See [`docs/roadmap.md`](docs/roadmap.md).

## Documentation map

### Guides
- [Python API (`TardigradeClient`)](docs/guide/python-api.md) — the high-level facade
- [HuggingFace direct injection](docs/guide/knowledge-pack-store.md) — `KnowledgePackStore`
- [Calibration](docs/guide/calibration.md) — picking the right retrieval-key layer for your model
- [vLLM setup](docs/guide/vllm-setup.md) — KV Connector v1 path
- [Consumers](docs/guide/consumers.md) — integration patterns for agents, NPCs, document ingest
- [MCP setup](docs/guide/mcp-setup.md) — wire TardigradeDB into an MCP server
- [Concepts](docs/guide/concepts.md) — core vocabulary

### Reference
- [Architecture](docs/architecture.md) — the four-layer Aeon model
- [Positioning](docs/positioning.md) — why TardigradeDB, vs embedding RAG, vs traditional KV cache
- [Performance positioning](docs/positioning/latency_first.md) — measured numbers
- [Research log](docs/research-log.md) — the experiments that shaped the retrieval pipeline
- [Roadmap](docs/roadmap.md) — shipped, next, future paths to production
- [Technical design document](docs/technical/tdd.md)

### Community
- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Citing this project](CITATION.cff)
- [Changelog](CHANGELOG.md)

## FAQ

### Why "Tardigrade"?

Four design pillars borrowed from the animal:

- **Cryptobiosis → dormant memory revival.** Quantized KV state can be persisted, then "reanimated" by retrieval and reinjection later.
- **Resilience under stress → recovery-first design.** WAL + rebuildable derived state + fail-fast replay boundaries.
- **Tiny footprint → compressed survival.** Q4 / Q8 compression keeps memory practical under constrained capacity.
- **Adaptive survival → memory lifecycle control.** AKL promotion / demotion / decay keeps useful memory active and stale memory fading.

### Isn't this just a KV cache?

Yes at the data level; no at the system level. A raw KV cache is append-and-replay state for one running model session. TardigradeDB stores KV tensors, but behaves like a managed long-term memory kernel: attention-native semantic retrieval (not text keyword overlap), selective injection (not full-history replay), durable Q4 persistence across sessions (not process-local ephemerality), lifecycle governance (not unmanaged growth), a causal Trace + WAL recovery model, and a cross-agent boundary via a shared engine API.

### How does it compare to embedding RAG and a traditional KV cache?

| Dimension | Embedding RAG | Traditional KV cache | TardigradeDB |
|-----------|---------------|----------------------|--------------|
| Primary stored unit | Text + embedding vectors | K/V tensors for active context | Quantized K/V as durable memory cells |
| Retrieval signal | ANN / cosine similarity | None (append + replay only) | Attention-native (`q · k / √d_k`) |
| Persistence | External DB | Process / session-local | Cross-session, in-engine |
| Context usage | Retrieve text → re-tokenize | Replay prior cache pages | Inject selected slices |
| Lifecycle | App-defined | None | AKL: importance + tiers + decay |
| Round-trip | text → embed → search → text | none, but no retrieval | native tensor path |

For the longer treatment with "where RAG remains stronger today (honest)", see [`docs/positioning.md`](docs/positioning.md).

### Design principles

- **Tensor-native.** The primary stored unit is a KV cache tensor. Reads inject pre-computed K/V directly into the attention stack — no tokenization round-trip.
- **Zero external dependencies.** No Postgres, Neo4j, or vector DB. Custom storage engine with custom indices.
- **Latent-space retrieval.** Relevance via attention in latent space, not cosine similarity over external embeddings.
- **Self-curating.** The AKL algorithm autonomously manages promotion, demotion, and decay. No application-level memory management.

## Acknowledgments

The retrieval architecture is informed by the [MemArt paper](https://arxiv.org/abs/2409.17264) (brute-force SIMD over ANN at agent scale), [DiskANN](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/) (Vamana graph), and [Knowledge Packs](https://arxiv.org/abs/2604.03270) (atomic multi-layer KV injection). Hybrid-attention retrieval insights come from Michalak & Abreu 2025. Direct competitor analyses live in [`docs/competitors/`](docs/competitors/).

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you use TardigradeDB in research, see [`CITATION.cff`](CITATION.cff) or use GitHub's "Cite this repository" button.
