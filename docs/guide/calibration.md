# Calibration Guide

TardigradeDB's retrieval key — the vector used to match a query against stored cells — is computed from the model's hidden states. *Which* layer to read, and *which encoding* to use (mean-pooled hidden state vs raw K projection), depends on the model architecture. Picking the wrong combination silently produces near-random retrieval.

Calibration is a one-time, ~30 s sweep that picks the right `(strategy, layer)` empirically and caches the result.

## Supported model architectures

TardigradeDB works with two architectural families.

| Family | Examples | Default strategy | What to know |
|---|---|---|---|
| **Uniform softmax attention** | Qwen3, Llama-3, Mistral, Gemma-2, Phi-3.5, TinyLlama, GPT-2 | `HiddenStateKeyStrategy(query_layer)` | Works out of the box. The static `int(num_hidden_layers × 0.67)` heuristic picks a reasonable layer. Optional: run calibration to pick empirically. |
| **Hybrid attention** (linear / SSM / recurrent layers mixed with softmax) | Qwen3-Next, RecurrentGemma, Jamba, Zamba, Falcon-Mamba, Granite-4, MiniMax, Hunyuan-T1, IBM Bamba, Nemotron-H | `KVectorKeyStrategy(softmax_layer_idx)` — picked automatically by calibration | Mean-pooled hidden states flatline at every layer on these (per Michalak & Abreu 2025, retrieval lives in attention heads). K-vector encoding at a calibrated softmax layer hits 95–100 % top-1. **Calibration is required** — there is no good default layer. |

## Running calibration

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tardigrade_hooks import select_query_layer, CalibrationRegistry

tok = AutoTokenizer.from_pretrained("google/recurrentgemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/recurrentgemma-2b-it",
    torch_dtype=torch.bfloat16,
).to("cuda")

# One-time sweep — picks (strategy, layer) empirically, caches at
# ~/.tardigrade/calibration.json. Override the cache path with
# the TARDIGRADE_CALIBRATION_PATH environment variable.
reg = CalibrationRegistry()
result = select_query_layer(model, tok, registry=reg)
print(f"Best: {result.best_strategy}@{result.best_layer}")
# Future loads pick up the cached result automatically.
```

Enable per-layer progress logging on the sweep:

```python
import logging
logging.getLogger("tardigrade_hooks.calibrate").setLevel(logging.INFO)
```

## What the sweep does

`LinearSweepStrategy` enumerates:

- `HiddenStateKeyStrategy` × all `hidden_states` indices
- `KVectorKeyStrategy` × all softmax cache-layer indices

For each `(strategy, layer)` pair, it runs the bundled small paraphrased corpus through `(store, retrieve)` and measures top-1 / top-5 recall. Tiebreak order: top-1 → top-5 → deepest layer (lower layers are noisier, so a tie at depth wins).

The winner is cached in `CalibrationRegistry` (atomic JSON write). Subsequent loads on the same model skip the sweep.

## Overriding the cached choice

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

## Empirical baseline — RecurrentGemma-2B-it

| Strategy | Top-1 |
|----------|-------|
| `HiddenStateKeyStrategy` (any layer) | 3 / 20 |
| `KVectorKeyStrategy` @ layer 5 (first softmax attention layer) | 19 / 20 |
| `KVectorKeyStrategy` @ layer 11 | 20 / 20 |

Calibration on RecurrentGemma picks `k_vector @ layer 5` (the first attention layer in Griffin's 3:1 layout) at 19/20 top-1 / 20/20 top-5. With this cached choice, the full pipeline (calibrate → store → retrieve → inject → generate) reaches **20/20 retrieval (100 %)** and **18/20 generation-hit (90 %)** on the bundled 20-fact paraphrased corpus.

## See also

- [`docs/guide/python-api.md`](python-api.md) — TardigradeClient API reference.
- [`docs/guide/knowledge-pack-store.md`](knowledge-pack-store.md) — HuggingFace direct-injection consumer.
- [`docs/guide/vllm-setup.md`](vllm-setup.md) — vLLM connector path.
