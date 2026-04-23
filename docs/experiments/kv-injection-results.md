# KV Injection Validation — Results

**Date:** April 22, 2026  
**Model:** GPT-2 (117M, 12 layers, 12 heads, d_model=768)  
**Script:** `examples/kv_injection_validation.py`

## Summary

Three hypotheses were tested. **None of them was correct.**

| Hypothesis | Prediction | Result |
|-----------|-----------|--------|
| A: Breno is right (all injection fails) | Full KV and mean-pool both ≈ baseline | **Wrong** — full KV works dramatically |
| B: Mean-pooled is different | Mean-pool helps, full KV fails | **Wrong** — exact opposite |
| C: Injection works broadly | Both methods help | **Wrong** — only full KV works |

**The actual result:** Full per-token KV injection works. Mean-pooled injection is catastrophically broken.

## Raw Results

Six conditions tested across 5 experiential memory/query pairs. All memories are fictional — GPT-2 cannot infer the target tokens from training data.

| Pair | Target | 1. Baseline | 2. Text RAG | 3. Mean-pool (rel) | 4. Mean-pool (irr) | 5. Full KV (rel) | 6. Full KV (irr) |
|------|--------|------------|------------|-------------------|-------------------|-----------------|-----------------|
| Standup | "Marcus" | 0.0038% | 0.032% | **0.0000%** | 0.0000% | **0.101%** | 0.010% |
| Lunch | "ban" | 0.0096% | 0.538% | **0.0000%** | 0.0000% | **0.372%** | 0.001% |
| Slack | "sync" | 0.0021% | 1.697% | **0.0000%** | 0.0000% | **1.671%** | 0.002% |
| Bug | "Type" | 0.031% | 5.493% | **0.0000%** | 0.0000% | **3.074%** | 0.023% |
| Daniel | "Daniel" | 0.027% | 20.57% | **0.0001%** | 0.0001% | **22.40%** | 0.015% |

### Key Observations

**1. Full KV injection works — Breno's critique is empirically wrong for GPT-2.**

Across all 5 pairs, full per-token KV injection (Condition 5) dramatically increases the probability of the correct target token:

- Marcus: 26x improvement over baseline
- Banh mi: 39x improvement
- Sync: 795x improvement (matches Text RAG)
- TypeError: 98x improvement
- Daniel: 829x improvement (**exceeds** Text RAG)

For "Daniel" and "sync," full KV injection performs **as well as or better than** prepending the memory as text. The model successfully attends to the cross-context KV cache entries and extracts the relevant information.

**2. Irrelevant KV injection doesn't help.**

Condition 6 (irrelevant full KV) produces scores near baseline across all pairs. This confirms the improvement from relevant KV is genuine semantic transfer, not noise from having extra cache entries.

**3. Mean-pooled injection is catastrophically broken.**

Condition 3 produces P(target) = 0.000000 for every pair. Worse, the model's output distribution collapses to:
```
"?"     → 78-90% probability
"Immunity" → 3-5% probability
```

The mean-pooled pseudo-token doesn't just fail to help — it actively destroys the model's ability to generate coherent text. The model interprets the synthetic token as some kind of prompt terminator or malformed input.

**4. Text RAG is reliable but not always the best.**

Text RAG (Condition 2) is consistently strong, but for "Daniel" (P=20.6%) it's actually beaten by full KV injection (P=22.4%). This suggests that for some memories, the model's internal KV representation carries information more efficiently than the raw text.

## What This Means for TardigradeDB

### The Architecture Problem

TardigradeDB's current pipeline is:

```
Hidden states → mean-pool to (d_model,) → Q4 quantize → store → retrieve → reshape to (1, nH, 1, hD) → inject
```

This pipeline is broken at step 2. Mean-pooling destroys the per-token structure that makes KV injection work. The reshape at the end creates a synthetic token that the model cannot meaningfully attend to.

### What Works

```
Per-token KV cache → store full (seq_len, nH, hD) → retrieve → inject as past_key_values
```

This is what Condition 5 tested, and it works dramatically well. The question is whether TardigradeDB can practically store and retrieve full per-token KV cache.

### The Storage Trade-off

| What's stored | Size per memory (GPT-2) | Size per memory (Llama 8B) |
|--------------|------------------------|---------------------------|
| Mean-pooled (current) | 768 floats × 12 layers = 36 KB | 4096 × 32 layers = 512 KB |
| Full per-token KV (20 tokens) | 20 × 768 × 12 layers × 2 (K+V) = 1.4 MB | 20 × 4096 × 32 × 2 = 20 MB |
| Full per-token KV (100 tokens) | 7.2 MB | 100 MB |

Full KV is 40-200x larger. But TardigradeDB already has Q4 quantization (4x compression) and the SLB for fast retrieval. The architecture can handle this — it just needs to store the right thing.

## Revised Understanding

| Component | Before this test | After this test |
|-----------|-----------------|----------------|
| **Mean-pooled retrieval** (semantic search) | Proven (80-92% recall) | Still works — use as search index |
| **Mean-pooled injection** (kv_injector.py) | Assumed viable | **Broken — remove or redesign** |
| **Full KV injection** | Assumed broken (Breno's critique) | **Works — 26x to 829x improvement** |
| **Full KV storage** | Not implemented | **Should be the primary storage format** |

### The Emerging Architecture

```
                    ┌─────────────────────────────┐
                    │    Mean-pooled hidden state  │
                    │    (d_model,) per layer      │──── SEARCH INDEX
                    │    Used for retrieval only   │     (semantic similarity)
                    └─────────────────────────────┘
                                 │
                                 │ points to
                                 ▼
                    ┌─────────────────────────────┐
                    │    Full per-token KV cache   │
                    │    (seq, nH, hD) per layer   │──── STORED VALUE
                    │    Used for injection        │     (actual KV for attention)
                    └─────────────────────────────┘
```

Mean-pooled vectors are excellent for **finding** the right memory (proven by our retrieval experiments). Full per-token KV is necessary for **injecting** it back into the model (proven by this test). TardigradeDB should use mean-pooled as the search key and full KV as the stored value.

This is analogous to how a database index works: the index (B-tree, hash) is a compressed representation used for lookup. The actual row data is stored separately. Mean-pooled hidden states are the index; full KV cache is the row data.

---

## Root Cause Analysis: Why Mean-Pooling Fails

### The Space Mismatch

The mean-pooling failure is not a tuning issue — it's a mathematical category error. Hidden states and KV cache entries live in **completely different vector spaces**:

```
Hidden states:  d_model space (768 dimensions)
                Norm range: 50–3000 per token
                
KV cache:       head_dim space (64 dimensions × 12 heads)
                Norm range: 7–11 per token per head
                Computed as: K = W_K @ hidden_state, V = W_V @ hidden_state
```

When TardigradeDB's `kv_injector.py` reshapes a 768-dim mean-pooled hidden state to `(1, 12, 1, 64)`, it's not creating a K/V projection — it's reinterpreting raw residual stream values as if they were attention-projected vectors. The attention mechanism computes `Q · K^T`, and if K is in the wrong space, the attention scores are meaningless.

This explains the "?" collapse: the synthetic K vector creates anomalous attention scores that push the model into an out-of-distribution state, causing it to emit the most generic possible token.

### What About Projecting First?

Even projecting mean-pooled hidden states through the actual `W_K`/`W_V` matrices doesn't help:

```
Mean-pool → W_K/W_V projection → inject:  P(Marcus) = 0.000001
Baseline (no injection):                    P(Marcus) = 0.000038
```

The projection fixes the space mismatch (output is now coherent text, not "?"), but mean-pooling destroys too much information. "Marcus" is one of 11 tokens — averaging them all produces a generic "work standup" representation, not enough to recover specific names.

### Quantization Works

Q4-quantized full KV cache retains nearly all the injection benefit:

```
Full KV (fp32):  P(Daniel) = 0.2240  (829x baseline)
Full KV (Q8):    P(Daniel) = 0.2157  (787x baseline)  — 96% of fp32
Full KV (Q4):    P(Daniel) = 0.1990  (726x baseline)  — 89% of fp32
```

This is significant: TardigradeDB's existing Q4 quantization infrastructure can compress full KV cache with only 11% quality loss. The storage is viable.

### Selective Token Injection Doesn't Work

Injecting only the top-5 most "salient" tokens (by hidden state norm) fails:

```
Full KV (all 24 tokens):  P(Daniel) = 0.2240
Selective (top 5 tokens):  P(Daniel) = 0.0001  (worse than baseline)
```

The selected tokens were "I, tech, lead, name, backend" — notably missing "Daniel" itself. The salience heuristic (hidden state norm) doesn't identify the tokens that matter for recall. And even if it did, attention needs the full causal context — individual tokens without their neighbors are meaningless.

---

## Answering Breno's Next Critique

### "You're just restoring a previous context. This is just KV caching."

This is the natural follow-up, and it's partially right. The injection mechanism IS KV cache replay. But TardigradeDB does something that standard KV caching (vLLM, SGLang, TGI) cannot:

**Standard KV caching:** "I've seen this exact prefix before, skip recomputation."
- Requires exact token match
- Only works for shared prefixes
- No semantic understanding
- If the prompt is even slightly different, cache miss

**TardigradeDB:** "I've never seen this exact prompt, but I remember something semantically similar."
- Finds memories by **latent-space similarity**, not exact match
- Works across completely different phrasings
- "What happened at the morning meeting" retrieves KV from "During standup Marcus flagged..."
- The retrieval is the novel part; the injection is just the delivery mechanism

The analogy: Google Search and web browsers both "show you web pages." But Google's value isn't in displaying HTML — it's in finding the right page. TardigradeDB's value isn't in KV injection (that's just the delivery) — it's in **semantic retrieval in latent space** plus **persistent governance** (importance scoring, decay, tiering).

### What Makes This Different From a Vector DB + RAG?

| Feature | Vector DB + RAG | TardigradeDB |
|---------|----------------|--------------|
| Storage | Text chunks + embeddings | Full per-token KV cache (Q4 quantized) |
| Retrieval | Embedding similarity → return text | Hidden state similarity → return KV tensors |
| Injection | Prepend text to prompt (re-tokenize, re-encode) | Inject directly into attention (skip re-encoding) |
| Re-encoding cost | Full: model must re-process the retrieved text | Zero: KV is pre-computed, inject directly |
| Memory management | None (application layer) | Built-in: importance scoring, tier promotion, decay |
| Multi-agent | None (application layer) | Built-in: per-agent ownership, cross-agent sharing |

The performance benefit is real: Text RAG requires the model to re-tokenize and re-encode the retrieved text (additional forward pass over the context). KV injection skips that entirely — the KV is pre-computed. For long memories, this could save significant TTFT (time to first token).

### The Honest Assessment

TardigradeDB's value stack, from strongest to most speculative:

1. **Semantic retrieval in latent space** — Proven. Works. Novel approach vs text embeddings.
2. **Persistent KV storage with Q4 quantization** — Proven viable. 89% quality at 4x compression.
3. **Governance (AKL)** — Implemented but not stress-tested in production. Unique feature.
4. **KV injection for zero-cost context augmentation** — Proven to work with GPT-2. Needs validation on modern architectures (RoPE, GQA, sliding window attention).
5. **Cross-model KV portability** — Completely untested. May not work.
