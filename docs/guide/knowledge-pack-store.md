# KnowledgePackStore — HuggingFace direct-injection

You're using HuggingFace `transformers` and you want to give your model long-term memory it can recall *without* the recalled facts costing prompt tokens. `KnowledgePackStore` is the consumer that wires that up: it captures the model's KV cache after a forward pass on the fact you want to remember, compresses it to 4 bits per value (Q4 quantization — the signal survives intact at ~0.999 cosine similarity; see [`docs/experiments/synthetic-kv-injection.md`](../experiments/synthetic-kv-injection.md) for the validation), persists it to disk, and reinjects it into `model.generate()` later by passing the reconstructed cache as `past_key_values`. The generated output is byte-identical to having the original fact in the prompt — but you spent zero new prompt tokens recalling it, because the cache was precomputed and stored rather than re-tokenized.

This is the canonical TardigradeDB path for HuggingFace `AutoModelForCausalLM`. If you want the higher-level facade with automatic chunking, ingestion, and multi-view consolidation, use [`TardigradeClient`](python-api.md) instead — it sits one level up.

## End-to-end usage

```python
from tardigrade_db import Engine
from tardigrade_hooks import CalibrationRegistry, KnowledgePackStore

engine = Engine("/data/agent-memory")

# Calibration registry — picks the right way to compute retrieval keys
# (mean-pooled hidden states vs raw K projection, and which layer to read)
# for your specific model. The first call sweeps the options; subsequent
# calls reuse the cached choice. Required for hybrid-attention models
# (Qwen3-Next, RecurrentGemma, etc.); optional but recommended for
# uniform-softmax models like Qwen3, Llama-3, Mistral.
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

Storing a fact takes three steps inside `KnowledgePackStore.store()`. The text gets wrapped in the model's chat template, a forward pass materialises `past_key_values` (HuggingFace's name for the KV cache tuple), and the multi-layer KV tensor is handed to `engine.mem_write_pack()` for a single atomic fsync. Retrieving works in reverse: the query's hidden states get scored against stored packs, the winning pack comes back, and the engine reconstructs a `DynamicCache` from it. Injection is the last step — `model.generate(past_key_values=cache.clone(), ...)` accepts the reconstructed cache, and from the model's perspective the generation continues as if the original fact had been in the context all along.

The Q4 round-trip preserves the signal: cosine similarity between the original and reconstructed KV tensors stays at roughly 0.999 stage by stage. The strongest evidence that injection actually works lives in [`docs/experiments/synthetic-kv-injection.md`](../experiments/synthetic-kv-injection.md), where the same loop was tested with fully synthetic gibberish facts (nonsense proper nouns, fake units, invented entities). Any correct recall there is unambiguous proof — those strings can only come from the injected tensors, because they don't exist in any training corpus.

## Hybrid-attention support

If you're using a model with mixed attention types — RecurrentGemma, Qwen3-Next, Jamba, Granite-4, and the rest of the hybrid cohort — `KnowledgePackStore` handles the architecture correctly, but the mechanism is worth knowing about. These models interleave traditional softmax-attention layers with recurrent or linear-attention layers, and only the softmax layers contribute a meaningful KV cache. The store path filters the recurrent layers out automatically; the retrieve path's pack-integrity guard knows to accept the shorter pack instead of failing on "wrong layer count."

Validated end-to-end on RecurrentGemma-2B-it (26 total layers, 8 of which are softmax attention in Griffin's 3:1 layout): all stores succeed and retrieval reconstructs a cache with exactly the 8 softmax layers populated, which is what the model wants.

## See also

- [`docs/guide/calibration.md`](calibration.md) — picking the right retrieval-key strategy.
- [`docs/guide/python-api.md`](python-api.md) — `TardigradeClient` facade for the chunking / ingestion / multi-view path.
- [`docs/experiments/synthetic-kv-injection.md`](../experiments/synthetic-kv-injection.md) — the validation experiment.
- [`docs/architecture.md`](../architecture.md) — the four-layer engine that lives behind this consumer.
