# vLLM Integration Guide

You have a vLLM deployment serving an LLM in production, and you want to give it persistent memory across requests. This guide walks the wiring, but before you write any config it's worth being honest about what the integration currently delivers and what it doesn't — vLLM's KV Connector v1 API was designed for prefix-cache acceleration, not cross-prompt KV injection, and that distinction shapes what TardigradeDB can and can't do for you on the vLLM path today.

## What this integration does and doesn't do today

**It does** capture the KV cache produced during generation and store it in TardigradeDB for future reuse. If the same prompt arrives again (token-identical prefix), vLLM's stock prefix-cache layer can serve the stored KV directly and skip prefill computation entirely. That's a real and useful win: faster response, lower GPU time, and durable across vLLM restarts.

**It doesn't** inject stored KV across *different* prompts. The thing you might be looking for — *"the model behaves as if it had already lived through prior conversations"* — would need vLLM to mix loaded "memory" KV with the current request's computed KV inside the attention layer itself, and the v1 connector API doesn't expose that. The same constraint applies to SGLang's RadixAttention: both serving frameworks treat the connector as a prefix cache. Full cross-prompt injection would require a custom vLLM attention backend (a fork, with RoPE-offset handling and ongoing maintenance), and that's [tracked as future work](../roadmap.md) rather than something you can wire up today.

If you want zero-token cross-prompt KV injection on a HuggingFace model that you control directly, take [`KnowledgePackStore`](knowledge-pack-store.md) — it does what vLLM's v1 connector can't, just at a different scale of deployment.

## Prerequisites

- vLLM ≥ 0.9.0 (validated against vLLM 0.19.1).
- A CUDA-capable GPU. vLLM itself only runs on CUDA; the TardigradeDB engine runs on CPU alongside it.
- TardigradeDB installed — either `pip install tardigrade-db` or built from source per the [Quick Start](../../README.md#quick-start).

## Quick Start

Install vLLM and TardigradeDB into the same environment, then wire the connector through `KVTransferConfig`. `KVTransferConfig` is vLLM's mechanism for plugging in third-party KV connectors; `kv_role="kv_both"` tells it to use TardigradeDB for both saving (capturing during generation) and loading (matching on incoming requests):

```bash
pip install vllm tardigrade-db
```

```python
from vllm import LLM
from vllm.config import KVTransferConfig

kv_config = KVTransferConfig(
    kv_connector="TardigradeConnector",
    kv_connector_module_path="tardigrade_vllm.connector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "db_path": "/data/agent-memory",  # where memory cells persist
        "owner": 1,                       # owner id for memory isolation
    },
)

llm = LLM(model="Qwen/Qwen3-0.6B", kv_transfer_config=kv_config)
print(llm.generate(["The capital of France is"]))
```

The first generation runs normally and the resulting KV gets stored. A second identical prompt will hit the prefix cache and skip prefill.

## Configuration options

You can pass config either through `kv_connector_extra_config` (as in the snippet above) or through environment variables that the connector reads at construction time. Environment variables are useful for CLI-style vLLM invocations where you can't easily pass Python objects in.

| Key | Env variable | Default | What it does |
|-----|--------------|---------|--------------|
| `db_path` | `TARDIGRADE_DB_PATH` | `./tardigrade-memory` | Engine storage directory. Pick somewhere stable, not `/tmp`. |
| `owner` | `TARDIGRADE_OWNER` | `1` | Owner id under which memories are stored. Use distinct ids if you're serving multiple isolated tenants from one engine. |

---

## How it works under the hood

This section explains what happens inside the connector when vLLM generates a token, and what the engine sees on the other side. If you just want to wire the integration, you've already seen everything you need above — keep reading only if you're debugging, contributing, or wanting to understand the constraints behind the limitations section.

### Save path — capturing KV during generation

vLLM generates tokens using paged attention, which means the KV cache lives in fixed-size blocks of shape `(num_blocks, block_size, num_heads, head_dim)` so requests can be batched efficiently. The connector hooks into vLLM at two points: `save_kv_layer()` fires after each attention layer's forward pass and accumulates the layer's K/V blocks into a per-request buffer, and `wait_for_save()` fires after the final layer to flush the buffer to TardigradeDB as one Q4-quantized pack. The save is atomic across all layers in a single fsync — either the whole pack lands or none of it does.

### Load path — matching on incoming requests

The primitives are in place but the integration into vLLM's scheduler under real concurrent load is still partial (see the status table below). When a new request arrives, `get_num_new_matched_tokens()` queries TardigradeDB for a semantically matching pack using the request's prompt as a retrieval key. If a match is found, `start_load_kv()` copies the stored KV blocks into vLLM's paged buffer for the matched token range, and vLLM skips prefill for those tokens. The trace-boosted retrieval path (`mem_read_pack_with_trace_boost`) is wired into the connector for multi-hop queries.

### Block format conversion

vLLM's paged attention blocks and TardigradeDB's storage format are different shapes — vLLM wants `(num_blocks, block_size, num_heads, head_dim)` and the engine stores flat `[K_flat | V_flat]` arrays per layer. The `tardigrade_vllm.format` module handles the conversion in both directions, including zero-padding when the request's token count doesn't divide the block size.

## Status: what works end-to-end, what's still partial

The connector's parts have been built up over time, and not everything is at the same level of polish. The honest breakdown:

**Working end-to-end and validated on Qwen3-0.6B:** the save path captures KV during generation; the block format conversion round-trips faithfully; semantic matching on request arrival uses the embedding-table retrieval key correctly; trace-linked retrieval is wired in; the GPU tensor copy in `start_load_kv()` has been validated against a mock context.

**Implemented but with known gaps:** per-request `slot_mapping` is not yet threaded through `save_kv_layer()`, so the save path currently captures only block 0 of each layer rather than the full prompt range — round-trip works as a proof of concept but the stored KV isn't yet semantically complete for full-prompt reuse. The connector also assumes one active request at a time (the layer-accumulation buffer is shared), so concurrent requests would clobber each other; multi-request batching needs request-level keying. And the save path fires on every forward step including each decode, so a 20-token completion produces 20 packs rather than one coalesced pack at completion.

**Not currently supported, by design or by serving-framework constraint:** cross-prompt KV injection (the framework's v1 API doesn't expose the attention-layer hook that would make it possible — see the top of this guide). Cross-model injection (stored KV from one model architecture isn't reinjectable into a different architecture). Cross-process visibility without `Engine::refresh()` (vLLM's `EngineCore` subprocess writes to disk; observers in other processes need to reopen the engine to see new writes).

If you hit a vLLM 0.19+ deprecation warning about the connector `__init__` not accepting a `kv_cache_config` second argument, that's expected and harmless on current vLLM versions — the connector keeps working.

## Test coverage

The integration has four test suites at increasing levels of integration. CPU-only suites run anywhere; the GPU suite needs CUDA available.

| Suite | Tests | What it validates |
|-------|-------|-------------------|
| `test_vllm_format.py` | 4 | flat ↔ paged block round-trip |
| `test_vllm_connector.py` | 4 | engine retrieval surface the connector calls |
| `test_vllm_load_path.py` | 4 | `start_load_kv()` tensor copy logic against a mock GPU context |
| `test_vllm_integration.py` | 5 | full vLLM round-trip with Qwen3-0.6B (`pytest -m gpu`) |

Run the first three anywhere; run the fourth in a Linux/WSL2 environment with `pytest -m gpu`.

## See also

- [`knowledge-pack-store.md`](knowledge-pack-store.md) — the HuggingFace direct-injection path. Use it if you control the model and want zero-token cross-prompt injection (the thing vLLM's v1 API can't currently support).
- [`../roadmap.md`](../roadmap.md) — the custom-vLLM-attention-plugin path that would enable full cross-prompt injection in production serving.
- [`../experiments/sglang-investigation.md`](../experiments/sglang-investigation.md) — why SGLang has the same prefix-cache-only constraint.
