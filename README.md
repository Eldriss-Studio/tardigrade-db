# TardigradeDB

[![CI](https://github.com/Eldriss-Studio/tardigrade-db/actions/workflows/ci.yml/badge.svg)](https://github.com/Eldriss-Studio/tardigrade-db/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tardigrade-db.svg)](https://pypi.org/project/tardigrade-db/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-eldriss--studio.github.io-blue)](https://eldriss-studio.github.io/tardigrade-db)
[![Rust 1.95+](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](rust-toolchain.toml)

> **TardigradeDB v0.7.5 is a research-grade preview.** Public APIs are stable; benchmark methodology is under active validation.

Most LLM memory systems run a separate embedding model on every store and every query, then re-tokenize the retrieved text into the prompt — three round trips per fact recalled, and a context window that fills as the agent learns. TardigradeDB skips both detours. It stores the model's own attention state (the KV cache), retrieves it via the same dot-product attention the model uses to think, and reinjects it without spending a single prompt token. Built from scratch in Rust with PyO3 Python bindings.

## Hero example

This example needs a CUDA-capable GPU with ~2 GB free. To run on CPU, see [`examples/e2e_demo.py`](examples/e2e_demo.py), which uses GPT-2.

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
# and the output is byte-identical to having the fact in the prompt.
```

## Why TardigradeDB?

Embedding RAG asks one LLM to read text retrieved by a *different* model that was trained to embed text. There's a translator in the middle. Every fact your agent stores costs an embedding call; every fact it recalls costs another embedding call and then a re-tokenization back into the prompt. TardigradeDB cuts the translator out. It stores the LLM's own hidden-state tensors and reinjects them directly into attention. The model searches its own memories using its internal activations — no translator model, no prompt tokens consumed on injection.

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
| Warm-tier compression (zstd-over-Q4 on Validated/Core writes, 2.66× shrink) | Stable | [`docs/experiments/2026-05-21-warm-tier-codec.md`](docs/experiments/2026-05-21-warm-tier-codec.md) |
| Portable snapshot + labeled checkpoints | Stable | [`docs/architecture.md`](docs/architecture.md) |
| `TardigradeClient` facade (chunking + ingestion + consolidation) | Stable | [`docs/guide/python-api.md`](docs/guide/python-api.md) |
| HTTP / REST bridge | Stable | [`python/tardigrade_http/`](python/tardigrade_http/) |
| LoCoMo / LongMemEval benchmark methodology | Under validation | [`docs/research-log.md`](docs/research-log.md) |

## Performance snapshot

Measured on a 5K-cell synthetic corpus with 1024-dim keys (matches Qwen3-0.6B hidden size) unless noted.

| Metric | Number | Source |
|--------|--------|--------|
| Retrieval latency, 5K cells | **p50 = 0.34 ms, p99 = 0.51 ms** | [`experiments/latency_benchmark_v2.py`](experiments/latency_benchmark_v2.py) |
| Per-cell on-disk footprint, 5K cells | **751 B** (Draft, worst-case) / **~516 B** (Validated/Core via ZstdQ4) | [`experiments/footprint_audit.py`](experiments/footprint_audit.py), [`docs/experiments/2026-05-21-warm-tier-codec.md`](docs/experiments/2026-05-21-warm-tier-codec.md) |
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

### Is this a Rust library or a Python library?

Both, but not equally. The engine — storage, retrieval, governance, indexing — is a Rust workspace. Consumers reach it through PyO3 bindings that ship as a Python wheel. We put the engine in Rust for the latency and footprint reasons the [Architecture](#architecture) section covers, and we put the consumer surface in Python because that's where the LLM ecosystem lives: HuggingFace `transformers`, vLLM, the agent frameworks, the calibration tools, the notebooks.

The practical consequence is that **`pip install tardigrade-db` is the install command for everyone using the library**, regardless of whether you write Rust elsewhere. There is no separate Rust crate to depend on from your own Rust application today — the workspace crates are unpublished (`publish = false`) and not designed as a third-party Rust API. If you want TardigradeDB inside a Rust binary, the options are to embed a Python interpreter, to fork and `path = "..."` the workspace, or to wait for the crates to land on crates.io ([roadmap](docs/roadmap.md)).

Two paths follow.

### Using TardigradeDB from Python

This is the path for almost everyone — anyone storing and retrieving KV memory from a HuggingFace model, a vLLM serving deployment, an agent framework, or a notebook.

```bash
pip install tardigrade-db
# Only if you'll run the HuggingFace injection examples:
pip install transformers torch
```

Then drop the [hero example](#hero-example) into a Python file and run it. For more usage patterns, see [`docs/guide/python-api.md`](docs/guide/python-api.md) (the `TardigradeClient` facade) and [`docs/guide/knowledge-pack-store.md`](docs/guide/knowledge-pack-store.md) (direct HuggingFace injection).

### Building TardigradeDB from source

This is the path if you're contributing to the engine, hacking on the Python bindings, or want to run the end-to-end demo on a CPU-only box.

```bash
git clone https://github.com/Eldriss-Studio/tardigrade-db.git
cd tardigrade-db
lefthook install
just ci        # fmt + lint + typos + test + deny + doc
```

Rebuild the Python bindings against your local changes:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop
pytest tests/python/ -v -m "not gpu"
```

Run the end-to-end GPT-2 demo (CPU-friendly; validates the full persistence and retrieval loop without needing a GPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
python examples/e2e_demo.py
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor workflow — CI gates, benchmarks, MSRV, reliability contracts, working with Claude.

## Project Status

**Current version:** v0.7.5. Public APIs (Python + Rust) are stable for the 0.x series; breaking changes ride minor bumps per the semver pre-1.0 convention.

**Stable surfaces:**
- The PyO3 Python API on `tardigrade_db.Engine` and the `tardigrade_hooks` consumer modules.
- The Rust `tdb-engine` crate and supporting workspace crates.
- Portable snapshot / restore (tar archive with magic, codec identifiers, SHA-256).

**Experimental:**
- Free-threaded Python 3.13t (PEP 703). The publish workflow ships a `cp313-cp313t` wheel alongside the abi3 wheels — installing under `python3.13t` resolves to it automatically. Lifts the GIL ceiling on concurrent pack reads: GIL Python plateaus at ~42k qps for single-call workloads at 8+ threads; the cp313t wheel keeps scaling. Proof-of-concept measurements: [`docs/experiments/2026-05-22-freethreaded-python-proof.md`](docs/experiments/2026-05-22-freethreaded-python-proof.md).

**Under active validation:**
- LoCoMo / LongMemEval benchmark methodology — earlier headline numbers (68.2 % LoCoMo / 90.9 % LongMemEval) were **retracted on 2026-05-14** after an audit found the runs measured the lexical fallback adapter on a corpus corrupted by a dataset-prep bug. Honest native-engine number on clean LoCoMo: ~36 % R@1 at 50-item scale; full-corpus re-measurement pending. Synthetic-corpus results (100 % recall at 5K, vague-query refinement, KV injection on gibberish facts, cross-model retrieval) are unaffected. Full record: [`docs/experiments/2026-05-14-bench-audit.md`](docs/experiments/2026-05-14-bench-audit.md).
- Production serving. HuggingFace direct injection works today via `KnowledgePackStore`. vLLM is partial: the official KV Connector v1 supports prefix-cache acceleration (a real win on repeated prompts), but cross-prompt KV injection — the thing that lets the model behave as if it had lived through prior conversations — would need a custom attention plugin, which is future work. See [`docs/roadmap.md`](docs/roadmap.md).

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

Tardigrades survive what kills most things. They dehydrate to near-zero metabolism — cryptobiosis — and rehydrate years later. That's the same trick the engine plays with quantized KV state: a memory cell can be persisted to disk and reanimated by retrieval and reinjection later, even after the process that wrote it is gone. Tardigrades also survive radiation, vacuum, and crushing pressure, which is the recovery-first instinct behind the WAL, the rebuildable derived state, and the fail-fast replay model. They're tiny — about half a millimetre — which matches the engine's per-cell footprint of around 751 bytes on disk. And they adapt: organisms that aren't useful in a given environment don't last. The Adaptive Knowledge Lifecycle does the same thing for memory cells, promoting the ones that get used and decaying the ones that don't.

### Isn't this just a KV cache?

Yes at the data level; no at the system level. A raw KV cache is append-and-replay state for one running model session — it lives in GPU memory while the model is generating, and it disappears when generation ends.

TardigradeDB stores the same kind of tensors but treats them as a managed long-term memory kernel. Retrieval is attention-native and semantic, not keyword overlap on text. Injection is selective — only the relevant slices, not a full-history replay. Persistence is durable across sessions through Q4 compression, not process-local ephemerality. Lifecycle is governed by importance, tier, and decay rather than unmanaged growth. And because memory cells are owner-scoped, the same engine can serve multiple agents without one's memories leaking into another's.

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
