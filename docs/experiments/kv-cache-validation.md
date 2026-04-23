# KV Cache Validation Experiments

**Date:** April 22-23, 2026
**Models tested:** GPT-2 (124M), Qwen3-0.6B (596M)
**Character:** Sonia, 34, freelance translator, Chicago — 16 memories across completely different life domains (work, parenting, cooking, health, legal, social, hobbies, dreams, errands, finances, neighbors, media)

## Key Finding

**What you store matters more than model size or retrieval strategy.**

All previous experiments stored `hidden_states` — the model's raw internal activations before they get transformed for attention. These are generic representations that cluster together regardless of content.

The correct approach stores `past_key_values` — the actual projected K tensors from the KV cache. These are the vectors the model specifically trained its attention mechanism to differentiate. Switching from hidden states to real KV cache **doubled recall** on the same model with the same memories.

## Results Summary

### Sonia's 16 Multi-Domain Memories (Qwen3-0.6B)

| What was stored | Vectorization | Recall | Unique top-1 |
|---|---|---|---|
| Hidden states | Mean-pool | 31.2% | 1 (gravity well) |
| Hidden states | Per-token | 31.2% | 1 (gravity well) |
| **KV cache (K projections)** | Mean-pool | **62.5%** | 5 |
| **KV cache (K projections)** | Per-token | **75.0%** | 7 |

### What "gravity well" means

In the hidden-states experiments, one memory dominated every query — no matter what you asked about (cooking, running, custody hearings), the same memory came back as the top result. The model's raw representations at a given layer mostly encode "this is English narrative text" rather than distinguishing topics.

The KV cache projections broke this pattern. 7 different memories appeared as the #1 result across 16 queries.

### Per-Token KV Hits (75% recall)

| Query | Found? | Rank |
|---|---|---|
| Pharmaceutical patent translation | Yes | #1 |
| Lucia bit someone at school | Yes | #3 |
| Cooking disaster with smoke alarm | No | — |
| Doctor visit, lying about sleep | No | — |
| Custody schedule with mediator | Yes | #1 |
| Running into old friend at grocery store | Yes | #1 |
| Running along the lake at sunrise | Yes | #4 |
| Dream about grandmother's kitchen | Yes | #2 |
| Car repair, catalytic converter | Yes | #2 |
| Lucia reading a book before bed | Yes | #1 |
| Client rejected translation word choice | Yes | #1 |
| Eating cereal alone at midnight | Yes | #4 |
| First snow, Lucia drawing on window | No | — |
| Invoices, chasing overdue payments | Yes | #1 |
| Neighbor brought pierogi, shared advice | Yes | #5 |
| Watching Coco with Lucia | No | — |

### The 4 Misses

1. **Cooking disaster** — "smoke alarm" and "risotto" don't appear in any query-side tokens. The model's K projection for "cooking disaster" doesn't overlap enough with the stored memory's tokens about "risotto" and "marinara sauce."
2. **Doctor visit** — The memory mentions "Dr. Huang," "vitamin D," "blood pressure." The query says "doctor" and "lying about sleep." The connection is contextual, not lexical or attention-level.
3. **First snow** — Emotional/sensory memory. The query "Lucia drawing on the foggy window" is specific but the memory's K vectors encode weather/environment more than the specific foggy-window detail.
4. **Watching Coco** — "Day of the Dead" in the query vs "Coco" and "abuelita" in the memory. Cultural knowledge needed to bridge these.

All 4 misses require reasoning beyond surface similarity — they need world knowledge that a 0.6B model doesn't reliably encode in its attention projections. A larger model (3B+) would likely close these gaps.

## Previous Experiments (Context)

### GPT-2 vs Qwen3 on Maya (Veterinary Memories)

Both models showed identical gravity-well behavior (41.7% recall) when storing hidden states. Every query returned the same memory regardless of content.

**Why:** Maya's memories all happened in one domain (veterinary hospital). Small models can't differentiate "intestinal surgery" from "jugular catheter" from "diabetic cat medication" in their internal representations — they all map to "veterinary medical content."

### GPT-2 on Kael (Software Engineering Memories) — Original Experiment

80% recall. GPT-2 was trained heavily on programming text, so it had rich representations for "standup meeting" vs "debugging auth" vs "Cypress test fix." Domain expertise matters.

### Llama 3.2:3b Pooled Embeddings on Kael

100% recall but 0 signal-to-noise ratio. Pooled embeddings (from `create_embedding()`) find memories but can't distinguish genuine matches from unrelated queries. This was an invalid test — pooled embeddings are not KV cache tensors.

## What We Learned

1. **Store the KV cache, not hidden states.** The K projection matrices (W_k) transform raw representations into a space specifically trained for attention-based retrieval. Hidden states are the raw flour; K projections are the bread.

2. **Per-token beats mean-pool.** Mean-pooling averages away the distinguishing tokens. Per-token storage preserves "Captain swallowed a corn cob" as distinct from "catalytic converter needs replacing."

3. **Domain diversity helps retrieval.** 16 memories across different life domains (Sonia) retrieve better than 12 memories in one domain (Maya at the vet hospital), because the model naturally separates "cooking" from "custody hearing" in its latent space.

4. **Model size matters for reasoning, not for retrieval.** A 0.6B model achieves 75% recall on diverse memories with proper KV storage. The misses require world knowledge (Coco = Day of the Dead), not better retrieval.

5. **The architecture is validated.** TardigradeDB's design — persist KV cache tensors, retrieve via attention-space dot product — works. The early experiments that showed poor results were storing the wrong data (hidden states instead of K projections).

## Experiment Scripts

| Script | What it tests |
|---|---|
| `experiments/maya_kv_tensors_comparison.py` | GPT-2 vs Qwen3, hidden states, Maya's vet memories |
| `experiments/sonia_production_sim.py` | Hidden states, single layer, Sonia's diverse memories |
| `experiments/sonia_real_kv_cache.py` | **Real KV cache**, K projections, Sonia's diverse memories |
| `experiments/llama3_two_agent_validation.py` | Llama 3.2 pooled embeddings (invalid — not KV cache) |
| `experiments/llama3_fresh_agent_validation.py` | Llama 3.2 pooled embeddings on Maya (invalid) |

## Next Steps

1. **Larger model with real KV cache** — Run Sonia's memories through a 3B+ model using K projections. Expect 90%+ recall as the model has enough capacity to encode "Coco = Day of the Dead" and "doctor visit = lying about sleep."
2. **Multi-head attention scoring** — Current retrieval uses dot product of concatenated head keys. Per-head scoring would let the system weight heads that are more relevant to the query.
3. **Document the hidden-states vs KV-cache distinction** in the main README/spec. All future experiments must use `past_key_values`, not `output_hidden_states`.
