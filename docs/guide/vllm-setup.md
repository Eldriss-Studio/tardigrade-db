# vLLM Integration Guide

TardigradeDB integrates with vLLM via the KV Connector v1 API. This enables persistent memory for production LLM serving — KV cache is captured during generation and stored in TardigradeDB for future injection.

## Prerequisites

- vLLM >= 0.9.0 (validated against vLLM 0.19.1)
- GPU with CUDA support (vLLM requires GPU)
- TardigradeDB installed (`pip install tardigrade-db` or built from source)

## Quick Start

```bash
# Install vLLM and TardigradeDB
pip install vllm tardigrade-db

# Programmatic setup (vLLM 0.19+)
python - <<'PY'
from vllm import LLM
from vllm.config import KVTransferConfig

kv_config = KVTransferConfig(
    kv_connector="TardigradeConnector",
    kv_connector_module_path="tardigrade_vllm.connector",
    kv_role="kv_both",
    kv_connector_extra_config={"db_path": "/data/agent-memory", "owner": 1},
)

llm = LLM(model="Qwen/Qwen3-0.6B", kv_transfer_config=kv_config)
print(llm.generate(["The capital of France is"]))
PY
```

## Configuration

Pass config as JSON via `--kv-connector-config`:

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `./tardigrade-memory` | Engine storage directory |
| `owner` | `1` | Owner ID for memory isolation |

Or set via environment variables:

```bash
export TARDIGRADE_DB_PATH=/data/agent-memory
export TARDIGRADE_OWNER=1
```

## How It Works

### Save Path (during generation)

1. vLLM generates tokens using paged attention
2. After each layer's attention computation, `save_kv_layer()` is called
3. The connector accumulates K/V blocks per layer
4. After all layers complete, `wait_for_save()` writes a complete KV pack to TardigradeDB
5. The pack is Q4 quantized and persisted to disk

### Load Path (on new requests) — Partial Implementation

Per the status table below, the load-path primitives are implemented and unit-tested (block format conversion, semantic-match retrieval-key, GPU tensor copy via `start_load_kv` with a mock context, trace-linked retrieval via `mem_read_pack_with_trace_boost`). What's still in development is integration into the full vLLM scheduler→worker path under real concurrent load. The flow is:

1. When a new request arrives, `get_num_new_matched_tokens()` will query TardigradeDB
2. If a matching memory is found, its stored KV will be loaded into vLLM's paged buffer
3. vLLM will skip prefill for the externally-provided tokens

### Block Format Conversion

vLLM uses paged attention blocks: `(num_blocks, block_size, num_heads, head_dim)`
TardigradeDB stores flat arrays: `[K_flat | V_flat]` per layer

The `tardigrade_vllm.format` module handles conversion between these formats, including padding for partial blocks.

## Current Status

| Feature | Status |
|---------|--------|
| Save path (capture KV during generation) | Implemented + validated end-to-end with Qwen3-0.6B |
| Block format conversion (paged ↔ flat) | Implemented + tested (4 unit tests) |
| Semantic matching on request arrival | Implemented (embedding-table retrieval key) |
| Trace-linked retrieval in connector | Implemented (uses `mem_read_pack_with_trace_boost`) |
| Load path GPU tensor copy (`start_load_kv`) | Implemented + unit-tested with mock context |
| Per-request slot mapping in save | **Not yet** — currently saves block 0 as proof of round-trip |
| Multi-request batching | **Single-request assumption** — connector uses one shared `current` buffer |

## Architecture

```
vLLM Engine
  ├── Scheduler → get_num_new_matched_tokens() → TardigradeDB query
  ├── Worker → start_load_kv() → load stored KV blocks
  ├── Attention → save_kv_layer() → accumulate per-layer KV
  └── Post-forward → wait_for_save() → write KV pack to engine
                                              │
                                    TardigradeDB Engine (Rust)
                                    ├── BlockPool (Q4 storage)
                                    ├── PerTokenRetriever (Top5Avg)
                                    ├── TraceGraph (memory links)
                                    └── Governance (AKL lifecycle)
```

## Limitations

- **GPU required.** vLLM runs on CUDA GPUs only. The TardigradeDB engine (Rust) runs on CPU.
- **Model must match.** Stored KV from one model architecture cannot be injected into a different model.
- **Save granularity is one block per request.** Until per-request `slot_mapping` is threaded through `save_kv_layer`, only block 0 of each layer is captured. Round-trip works but stored KV is not yet semantically useful for full-prompt reuse.
- **One pack per forward pass.** The save path fires on every step (prefill + each decode), so a 20-token completion produces 20 packs. Real deployment needs to coalesce to one pack at request completion.
- **Single-request assumption.** The connector accumulates layers in a shared `current` buffer; concurrent requests would clobber each other. Need request-level keying.
- **Cross-process engine state.** TardigradeDB caches engine state at `Engine::open()`. The connector lives in vLLM's `EngineCore` subprocess and writes to disk; observers in other processes must reopen the engine to see fresh writes.
- **vLLM 0.19+ deprecation.** The connector `__init__` should accept a `kv_cache_config` second argument; without it, vLLM logs a deprecation warning but still works.

## Test Coverage

| Suite | Count | What it validates |
|-------|-------|-------------------|
| `test_vllm_format.py` | 4 | flat ↔ paged block round-trip |
| `test_vllm_connector.py` | 4 | engine retrieval surface used by the connector |
| `test_vllm_load_path.py` | 4 | `start_load_kv` tensor copy logic with mock GPU caches |
| `test_vllm_integration.py` | 5 | full vLLM round-trip with Qwen3-0.6B (`-m gpu`) |

Run CPU-only tests anywhere; run GPU tests in WSL2/Linux with `pytest -m gpu`.
