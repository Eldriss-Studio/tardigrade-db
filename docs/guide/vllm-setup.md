# vLLM Integration Guide

TardigradeDB integrates with vLLM via the KV Connector v1 API. This enables persistent memory for production LLM serving — KV cache is captured during generation and stored in TardigradeDB for future injection.

## Prerequisites

- vLLM >= 0.9.0 (KV Connector v1 API)
- GPU with CUDA support (vLLM requires GPU)
- TardigradeDB installed (`pip install tardigrade-db` or built from source)

## Quick Start

```bash
# Install vLLM and TardigradeDB
pip install vllm tardigrade-db

# Start vLLM with TardigradeDB connector
vllm serve Qwen/Qwen3-0.6B \
  --kv-connector tardigrade_vllm.connector.TardigradeConnector \
  --kv-connector-config '{"db_path": "/data/agent-memory"}'
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

### Load Path (on new requests) — Planned

The load path (injecting stored KV into new requests) requires semantic matching between the incoming query and stored memories. This is under development:

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
| Save path (capture KV during generation) | Implemented |
| Block format conversion (paged ↔ flat) | Implemented + tested |
| Load path (inject stored KV) | Skeleton — needs semantic matching |
| Semantic matching on request arrival | Planned |
| Trace-linked retrieval in connector | Planned |

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
- **Load path incomplete.** The semantic matching for automatic KV injection is not yet implemented. Currently only the save path works.
- **Single-request assumption.** The current save accumulation assumes batch size 1. Multi-request batching needs additional request-level tracking.
