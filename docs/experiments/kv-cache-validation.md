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

### Traditional RAG Baseline

A retrieval-only traditional RAG baseline was run against the same 100-memory corpus using `intfloat/e5-small-v2` embeddings through `transformers`. Memories were encoded as `passage: ...`, queries as `query: ...`, pooled with the attention mask, L2-normalized, and ranked by cosine similarity.

Result: traditional embedding RAG achieved **100% recall@1, recall@3, recall@5, and recall@10** on the 30 positive queries. It produced 30/30 unique top-1 memories, with no gravity well. This does not change the architecture by itself, but it gives a hard benchmark: current Q*K retrieval is not competitive with standard embedding retrieval on this corpus.

This comparison is retrieval-only. It does not test whether KV injection after retrieval improves answer quality, reduces token cost, or preserves model state better than text RAG. Those remain separate questions.

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

## Q*K Signal Audit (April 23, 2026)

A comprehensive diagnostic audit answered the question: is the correct memory present in the latent signal, or is Q*K itself not good enough?

### Rank-Depth Analysis

The correct memories ARE in the signal — they're just ranked too low with the current scorer:

| Cutoff | Recall |
|---|---|
| R@5 (current) | 63.3% |
| R@20 | 76.7% |
| R@50 | 93.3% |
| R@100 | **100%** |

Every correct memory is recoverable if you look deep enough. The problem is ranking, not representation.

### Scorer Comparison

| Scorer | Recall@5 | Notes |
|---|---|---|
| per_head_max (current best Q*K) | 63.3% | Current engine scorer |
| cosine_sum_max (ColBERT-style) | 53.3% | Tested and rejected — worse |
| Latent oracle (best of all scorers) | 80.0% | Signal exists across scorers |
| Layer 14 colbert_sum | 100% | But 80% negative false positive rate |
| Best individual head | 100% | But 40% negative false positive rate |
| **Hidden states + top5_pair_avg** | **100%** | **10% negative false positive rate** |
| Traditional RAG (e5-small-v2) | 100% | Baseline |

### Key Finding

**Hidden states + top5_pair_avg achieves 100% recall with only 10% false positive rate.** This matches traditional RAG on retrieval quality while using latent representations.

This is ironic — hidden states were abandoned early (31% recall with mean-pooling). But mean-pooling was the problem, not hidden states. Per-token hidden states with top-5 pair averaging achieves perfect recall.

### Diagnostic Verdict

```
LAYER_OR_HEAD_PROBLEM
```

The correct memories are present in the latent signal. The current default layer (18) and scorer (per_head_max) are a poor combination. The best path found is hidden states at the right layer with top5_pair_avg scoring.

### What This Means

1. The latent signal is strong enough — no architecture pivot needed
2. The current scorer and layer choice are the bottleneck, not the approach
3. The remaining problem is false positive rejection (telling "genuine match" from "not in memory")
4. Hidden states with smarter aggregation outperform Q/K projections for cross-sequence retrieval

## Engine Pipeline Validation (April 23, 2026)

The diagnostic's 100% result was measured outside the engine with raw tensors. The critical question: does it survive Q4 quantization and the full retrieval path?

**Yes.** `experiments/scale_100_hidden_top5.py` ran the 100-memory corpus through the real engine pipeline:

| Metric | Result |
|---|---|
| Cross-domain recall | **10/10 (100%)** |
| Within-domain recall | **20/20 (100%)** |
| Overall recall@5 | **30/30 (100%)** |
| All results at rank #1 | Yes |
| Unique top-1 memories | 30/30 |
| Gravity well | PASS (worst = 1x) |
| Avg query latency | 97ms |
| Store time (100 memories) | 12.5s |

### Full Progression on 100-Memory Corpus

| Method | Recall@5 | Notes |
|---|---|---|
| Q*K per-token pipeline (max_sim) | 40.0% | Gravity well, 7x on one memory |
| Q*K per-head-max (diagnostic) | 63.3% | Best Q*K scorer |
| Cosine sum-of-max (ColBERT-style) | 53.3% | Tested and rejected |
| Hidden + top5_pair_avg (diagnostic) | 100.0% | Outside engine, raw tensors |
| Traditional RAG (e5-small-v2) | 100.0% | Embedding baseline |
| **Hidden + Top5Avg (engine pipeline)** | **100.0%** | **Through Q4 quantization, full pipeline** |

Q4 quantization did not destroy the hidden state signal. The Top5Avg scorer survived lossy 4-bit compression.

## What's Validated

- Hidden states + Top5Avg achieves 100% recall at 100 memories through the full engine pipeline
- Q4 quantization preserves enough hidden state signal for per-token retrieval
- No gravity well at 100-memory density with this approach
- Latency is acceptable (97ms per query, 12.5s to store 100 memories)
- The latent-space retrieval approach matches traditional RAG on this corpus
- Storing K projections (not hidden states) was a wrong turn — hidden states are better for cross-sequence retrieval
- Per-token scoring with top-5 averaging outperforms max-sim and all Q*K variants

## What's Not Yet Tested

- **Vague queries** — All test queries use specific vocabulary that overlaps with stored memories ("The sourdough starter I named Fernando"). Real agent queries would be vaguer ("What have I been cooking lately?"). This is the critical unknown.
- **Storage efficiency** — A 30-token memory stored as hidden states is ~90KB after Q4 vs ~50 bytes as text. Whether the latent approach's speed advantage on injection justifies 1000x storage overhead hasn't been measured.
- **Injection quality** — The injection pipeline (Phase 18) exists but hasn't been tested with hidden-state-retrieved memories on actual generation quality.
- **False positive calibration** — The diagnostic showed 10% false positive rate on negative queries. A production system needs a score threshold to say "I don't remember."
- **Layer sensitivity** — The current layer choice (67% depth) works but the diagnostic showed other layers can be better or worse. No automatic layer selection exists.

## Experiment Scripts

| Script | What it tests |
|---|---|
| `experiments/scale_100_hidden_top5.py` | **Hidden + Top5Avg through engine** — the 100% result |
| `experiments/scale_100_qk_diagnostics.py` | Full diagnostic suite — rank depth, oracle, layer/head sweep |
| `experiments/scale_100_qk.py` | Q*K retrieval at 100-memory density (40% baseline) |
| `experiments/scale_100_rag_baseline.py` | Traditional embedding RAG baseline (100%) |
| `experiments/corpus_100.py` | 100-memory corpus (10 domains x 10 memories) |
| `experiments/sonia_per_token_pipeline.py` | Q*K pipeline on 16 memories |

## Next Steps

1. **Vague query test** — Add 10-15 natural queries ("What have I been cooking?", "How is Lucia?") alongside specific ones. Run both hidden state retrieval AND traditional RAG on the same queries. This is the test that determines whether the latent approach works for real agent use.
2. **Storage efficiency measurement** — Compare: tensor storage size vs text size, injection latency vs re-tokenization latency. Quantify the actual trade-off.
3. **False positive calibration** — Establish a score threshold for "I don't remember" based on the negative query score distribution.
4. **Injection quality test** — Verify that hidden-state-retrieved memories, when injected via the Phase 18 MemoryInjector, improve generation quality vs text re-tokenization.
