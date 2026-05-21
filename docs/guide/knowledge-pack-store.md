# KnowledgePackStore — HuggingFace direct-injection

`KnowledgePackStore` is the consumer that wraps the engine's `mem_write_pack` / `mem_read_pack` for HuggingFace `transformers` models. It captures the full KV cache from `past_key_values`, persists it through Q4 quantization, retrieves it later, and injects it back into `model.generate()`.

This is the canonical path for using TardigradeDB with HuggingFace `AutoModelForCausalLM`. For the higher-level `TardigradeClient` facade (chunking, ingestion, multi-view consolidation), see [`python-api.md`](python-api.md).

## End-to-end usage

```python
from tardigrade_db import Engine
from tardigrade_hooks import CalibrationRegistry, KnowledgePackStore

engine = Engine("/data/agent-memory")

# Calibration registry — first-run sweep picks the right retrieval-key
# strategy + layer per model; subsequent runs reuse the cached choice.
# Required for hybrid-attention models (Qwen3-Next, RecurrentGemma, …);
# optional but recommended for uniform-softmax models.
reg = CalibrationRegistry()
kps = KnowledgePackStore(
    engine, model, tokenizer,
    owner=agent_id,
    calibration_registry=reg,   # consults reg if cached, else uses default heuristic
)

# Single-memory store (8/10 on injection, zero prompt tokens)
pack_id = kps.store("User prefers morning meetings")
text, tokens, had_memory = kps.generate("When should we meet?")

# Link related facts (the agent decides what's related)
existing = kps.store("Went to bookstore in Pilsen")
kps.store_and_link("Bookstore is called Casa Azul", existing)

# Multi-memory retrieval (follows trace links)
text, tokens, had = kps.generate_with_trace("Tell me about the bookstore")

# Batch-link related facts
kps.store_linked(["Fact A about Tomoko", "Fact B about Tomoko"])
```

## Overriding the retrieval-key strategy

`calibration_registry=` is the default-ergonomic path. When you need direct control:

```python
from tardigrade_hooks import HiddenStateKeyStrategy, KVectorKeyStrategy

# Force a specific hidden-state layer (uniform-softmax models):
kps = KnowledgePackStore(
    engine, model, tokenizer, owner=agent_id,
    retrieval_key_strategy=HiddenStateKeyStrategy(query_layer=17),
)

# Force K-vector encoding at a specific softmax attention layer
# (hybrid models — consult `model.config.layers_block_type` for valid indices):
kps = KnowledgePackStore(
    engine, model, tokenizer, owner=agent_id,
    retrieval_key_strategy=KVectorKeyStrategy(softmax_layer_idx=11),
)
```

See [`calibration.md`](calibration.md) for how the registry sweep picks the right combination automatically, and for the supported-architectures table.

## How it works under the hood

1. **Store** — wrap the fact text in the model's chat template, run a forward pass to materialise `past_key_values`, hand the multi-layer KV tensor to `engine.mem_write_pack()` (single fsync, atomic across all layers).
2. **Retrieve** — compute the query's hidden states, call `engine.mem_read_pack()`, reconstruct a `DynamicCache` from the returned pack.
3. **Inject** — clone the cache and pass it as `past_key_values=` to `model.generate()`. Output is byte-identical to having the text in the prompt, with zero prompt-token cost.

The Q4 round-trip preserves the signal: cosine similarity ≈ 0.999 stage-by-stage. See [`docs/experiments/synthetic-kv-injection.md`](../experiments/synthetic-kv-injection.md) for the validation experiment using fully synthetic gibberish facts (any correct recall is unambiguous proof — those strings can only come from the injected tensors).

## Hybrid-attention support

`KnowledgePackStore.store()` filters recurrent layers via `_softmax_layer_payloads`, and `n_softmax_layers` derives the architecture-aware count from `cfg.layers_block_type` / `cfg.layer_types`, so the retrieve-side pack-integrity guard accepts legitimately-shorter hybrid packs. End-to-end on RecurrentGemma-2B-it (26 layers / 8 attention, Griffin 3:1 spacing): 5/5 stores succeed, `retrieve_and_inject` reconstructs the cache with exactly 8 populated layers.

## See also

- [`docs/guide/calibration.md`](calibration.md) — picking the right retrieval-key strategy.
- [`docs/guide/python-api.md`](python-api.md) — `TardigradeClient` facade for the chunking / ingestion / multi-view path.
- [`docs/experiments/synthetic-kv-injection.md`](../experiments/synthetic-kv-injection.md) — the validation experiment.
- [`docs/architecture.md`](../architecture.md) — the four-layer engine that lives behind this consumer.
