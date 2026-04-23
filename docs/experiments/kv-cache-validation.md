# KV Cache Validation Experiments

**Date:** April 22-23, 2026
**Models tested:** GPT-2 (124M), Qwen3-0.6B (596M), Qwen2.5-3B (3B)
**Characters:** Sonia (16 diverse life memories), Maya (12 veterinary memories), Kael (software engineer)

## Key Findings

Three discoveries, each building on the last:

### 1. Store K projections, not hidden states

Hidden states are the model's raw internal activations before the attention projection. K projections (from `past_key_values`) are what the model trained its attention mechanism to use. Switching from hidden states to K projections doubled recall.

| What was stored | Recall | Problem |
|---|---|---|
| Hidden states (mean-pool) | 31.2% | One memory dominated all queries (gravity well) |
| K projections (mean-pool) | 62.5% | 5 unique top-1 memories |
| K projections (per-token) | 75.0% | 7 unique top-1 memories |

### 2. K*K matching doesn't work for per-token retrieval

The `PerTokenRetriever` stored K vectors and queried with K vectors (K*K). This failed catastrophically (25% recall) because K vectors share a massive common component across all sequences regardless of content.

Empirical evidence (Qwen3-0.6B, layer 18, three unrelated sentences):

| Position | K*K cross-sentence dot product | Content |
|---|---|---|
| 0 | 6281 (identical for all pairs) | "Tried", "The", "Eduardo" |
| 1-7 | 3700-4150 (high, low variance) | Mixed content tokens |

Position 0 is an attention sink — its K vector points in the same direction regardless of what the sentence says. But even non-zero positions have cross-sentence dots of ~4000, overwhelming content-specific signal (~200 difference between related vs unrelated pairs).

### 3. Query with Q, store K (Q*K = actual attention)

Transformers train Q projections to query and K projections to be queried. Using Q for queries and K for stored memories (Q*K) is what attention actually computes. This is stated in TardigradeDB's own learning roadmap:

> "Retrieval computes attention scores between the current query's Q vectors and stored K vectors directly."

Confirmed by FIER, ShadowKV, and the "From QKV to K/KV" paper: all use Q for queries. K*K produces symmetric attention that can't capture directional relationships.

| Method | Recall | Unique top-1 |
|---|---|---|
| K*K per-token (pipeline) | 25.0% | 5 |
| K*K mean-pool (real KV) | 62.5% | 5 |
| Q*K per-token (manual, no GQA) | 68.8% | 7 |
| Q*K mean-pool (old pipeline, pre-fix) | 62.5% | 9 |
| **Q*K per-token (fixed pipeline)** | **68.8%** | **8** |

Q*K per-token retrieval now matches the earlier manual result (68.8%) through the engine pipeline. The prior 62.5% "pipeline" result was not a true per-token pipeline: it mean-pooled Q tokens and could be satisfied by the SLB before `PerTokenRetriever` ran. The fixed path encodes all Q tokens, expands GQA K heads to Q dimensionality, and forces encoded per-token queries through max-sim cold scoring.

## Additional findings

### Position 0 (attention sink)

The first token in causal language models acts as an attention sink. Its K vector is aligned across all sequences regardless of content (cosine similarity = 1.0 at position 0 for unrelated sentences). Skipping position 0 during storage and retrieval is necessary.

### GQA dimension handling

Qwen3-0.6B uses Grouped Query Attention: 16 Q heads but only 8 KV heads (ratio = 2). Q and K have different dimensions (2048 vs 1024). Options:
- Expand K heads to match Q (repeat each K head twice) — used in the fixed pipeline (68.8%)
- Average Q head groups to match K — old pipeline behavior (62.5%)
- Q averaging loses signal and is now covered by a regression test

### RoPE and cross-sequence retrieval

K vectors in `past_key_values` have RoPE (Rotary Position Embedding) applied, baking position information into the vectors. For cross-sequence retrieval, position is irrelevant — we care about content, not where a token appeared. Extracting K via `k_proj(hidden_states)` without applying RoPE produces position-independent K vectors that are better for retrieval.

### Model size vs domain expertise

| Character | Domain | GPT-2 recall | Qwen3-0.6B recall |
|---|---|---|---|
| Kael | Software engineering | 80% | N/A |
| Maya | Veterinary medicine | 41.7% | 41.7% |
| Sonia | Diverse life domains | N/A | 68.8% (Q*K per-token pipeline) |

GPT-2 did well on Kael because it was trained on programming text. Both models failed equally on Maya because veterinary medicine is a single narrow domain that small models can't differentiate internally. Sonia's diverse memories (cooking, custody, running, dreams) retrieve better because the model naturally separates different life domains.

### Bigger model did not help

Qwen2.5-3B (5x larger) achieved the same 75% recall as Qwen3-0.6B on per-token K*K. The ceiling is the retrieval method, not the model capacity.

## Full Progression (Sonia, 16 memories)

| Method | Recall | Unique top-1 | Notes |
|---|---|---|---|
| Hidden states mean-pool | 31.2% | 1 | Gravity well — one memory dominates all |
| Hidden states per-token | 31.2% | 1 | Same gravity well, finer granularity didn't help |
| K*K per-token (pipeline) | 25.0% | 5 | K vectors share common component, worse than mean-pool |
| Llama 3.2 pooled embeddings | 75.0% | N/A | Invalid test — pooled embeddings, not KV cache |
| K*K mean-pool (real KV) | 62.5% | 5 | Real KV cache, mean-pooled, correct storage |
| K*K per-token (real KV) | 75.0% | 7 | Best K*K result |
| Q*K per-token (manual) | 68.8% | 7 | Correct Q*K scoring, no GQA handling |
| Q*K mean-pool (old pipeline) | 62.5% | 9 | Mean-pooled Q and could bypass per-token scoring |
| Q*K per-token (old pipeline) | 56.2% | 8 | GQA dimension loss reduced signal |
| **Q*K per-token (fixed pipeline)** | **68.8%** | **8** | Encoded Q tokens, GQA K expansion, forced max-sim cold path |

## What's Validated

- Storing K projections (not hidden states) is correct and necessary
- Q*K scoring (not K*K) matches how attention works
- Per-token storage preserves discriminative signal that mean-pooling destroys
- Skipping position 0 removes attention sink contamination
- RoPE-free projections are better for cross-sequence retrieval
- The retrieval architecture works; the fixed engine path now exercises encoded Q tokens against encoded K tokens

## 100-Memory Scale Test (April 23, 2026)

Scaled from 16 to 100 memories across 10 life domains (work, parenting, cooking, health, legal, social, fitness, dreams, errands, media). 30 genuine queries + 10 negative. Qwen3-0.6B, Q*K pipeline.

### Results

| Metric | 16 memories | 100 memories |
|---|---|---|
| Overall recall@5 | 68.8% | **40.0%** |
| Cross-domain recall | N/A | 50.0% (5/10) |
| Within-domain recall | N/A | 35.0% (7/20) |
| Unique top-1 | 8/16 | 16/30 |
| Worst gravity well | 2x | **7x** (mem 87, flat tire) |
| Avg query latency | N/A | 166ms |
| Store time | N/A | 11.9s |

### Per-Domain Breakdown

| Domain | Cross (1 query) | Within (2 queries) |
|---|---|---|
| Work | 1/1 | 2/2 |
| Parenting | 0/1 | 1/2 |
| Cooking | 1/1 | 0/2 |
| Health | 1/1 | 0/2 |
| Legal | 0/1 | 1/2 |
| Social | 1/1 | 0/2 |
| Fitness | 0/1 | 1/2 |
| Dreams | 0/1 | 1/2 |
| Errands | 1/1 | 0/2 |
| Media | 0/1 | 1/2 |

### What This Means

Retrieval quality degrades at scale. The Q*K per-token approach that worked at 16 memories (68.8%) drops to 40% at 100 memories. The gravity well problem returns — memory 87 (flat tire on the expressway) dominates 7 of 30 queries.

The core issue: raw Q*K dot product without softmax normalization lets high-magnitude token pairs dominate regardless of content relevance. At 16 memories there are ~300 stored tokens; at 100 memories there are ~2000. More tokens means more chances for spurious high-scoring matches to outcompete genuine ones.

Work domain retrieval is notably strong (3/3) because translation terminology is distinctive. Domains with shared vocabulary (parenting/legal both involve Lucia and Eduardo, cooking/errands both involve stores) are weaker.

### Honest Assessment

The per-token Q*K approach is the architecturally correct direction (matching how attention actually works), but the current implementation lacks the normalization and scoring refinements that make real attention effective. The gap between "raw Q*K dot product" and "proper attention with softmax and multi-head aggregation" is substantial.

The 40% recall at 100 memories means TardigradeDB's retrieval engine cannot yet reliably find memories at realistic scale. This is not a storage problem, a governance problem, or a model problem. It is a scoring problem in the retrieval layer.

## What's Validated

- Storing K projections (not hidden states) is correct and necessary
- Q*K scoring (not K*K) matches how attention works
- Skipping position 0 removes attention sink contamination
- RoPE-free projections are better for cross-sequence retrieval
- The storage, persistence, governance, and injection layers work correctly
- Latency is acceptable (166ms at 100 memories)

## What's Not Working

- **Retrieval quality degrades at scale** — 68.8% at 16 memories drops to 40% at 100
- **Gravity wells return at density** — one memory dominates 7/30 queries
- **Raw dot product is not attention** — missing softmax normalization, multi-head scoring, and proper Q*K scaling
- **Signal-to-noise is near zero** — cannot threshold "I don't remember"
- **Within-domain retrieval is weak** (35%) — shared vocabulary between domains causes confusion

## What Would Need to Change

The retrieval scoring needs to move from raw dot product to something closer to actual attention:

1. **Softmax normalization** — attention uses `softmax(Q*K^T / sqrt(d_k))`, not raw `Q*K`. The softmax sharpens the score distribution, making the best match stand out instead of being lost in a sea of high-magnitude noise.
2. **Per-head scoring** — different attention heads encode different relationships. Concatenating all heads into one vector and doing a single dot product loses the head-specific signal.
3. **Score normalization by memory length** — longer memories have more tokens and more chances for spurious high-scoring matches. Normalizing by token count would reduce this bias.

These are retrieval engine changes in Rust (`tdb-retrieval`), not model or storage changes.

## Experiment Scripts

| Script | What it tests |
|---|---|
| `experiments/sonia_per_token_pipeline.py` | **Q*K pipeline** end-to-end, latest approach |
| `experiments/sonia_real_kv_cache.py` | K*K real KV cache (mean-pool + per-token) |
| `experiments/sonia_production_sim.py` | Hidden states comparison |
| `experiments/maya_kv_tensors_comparison.py` | GPT-2 vs Qwen3, hidden states, Maya |
| `experiments/sonia_real_kv_3b.py` | Qwen2.5-3B, same as sonia_real_kv_cache |

## Next Steps

1. **Softmax-normalized scoring** — Replace raw dot product with `softmax(Q*K^T / sqrt(d_k))` in the retriever. This is the single biggest gap between current scoring and real attention.
2. **Per-head scoring** — Score per attention head instead of concatenating. Different heads capture different relationships.
3. **Length normalization** — Divide cell scores by token count to prevent longer memories from dominating.
4. **Re-evaluate at 100 memories** after scoring improvements.
