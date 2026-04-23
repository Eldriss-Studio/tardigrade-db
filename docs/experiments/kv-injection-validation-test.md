# KV Injection Validation Test

**Date:** April 22, 2026  
**Goal:** Empirically test whether cross-context KV injection helps, hurts, or is neutral  
**Model:** GPT-2 (117M) — full access to hidden states and KV cache  
**Status:** Planned

## Background

Breno's critique: KV cache vectors are context-dependent. Storing them from prompt A and injecting them into prompt B's attention should not work because they carry positional encoding and causal dependencies from a different sequence.

This test measures whether that's true in practice.

## Test Design

### The Setup

Use GPT-2 to generate completions for prompts that require **factual recall**. Compare output quality across **six conditions** that separate the three things TardigradeDB does:

```
── Baselines ──────────────────────────────────────────────────────────
Condition 1: BASELINE            — No memory, just the prompt
Condition 2: TEXT CONTEXT (RAG)  — Prepend memory as text (gold standard)

── What TardigradeDB actually does (mean-pooled) ─────────────────────
Condition 3: MEAN-POOLED INJECT (relevant)    — Inject mean-pooled pseudo-token from related memory
Condition 4: MEAN-POOLED INJECT (irrelevant)  — Inject mean-pooled pseudo-token from unrelated memory

── What Breno is critiquing (full per-token KV) ──────────────────────
Condition 5: FULL KV INJECT (relevant)    — Inject actual per-token KV cache from related memory
Condition 6: FULL KV INJECT (irrelevant)  — Inject actual per-token KV cache from unrelated memory
```

This three-way split is critical. Breno's critique targets full KV sharing (Conditions 5-6). TardigradeDB actually does mean-pooled injection (Conditions 3-4). These are fundamentally different operations and may have different outcomes.

### The Prompts

**Memory prompt (stored first):**
```
"The Eiffel Tower was completed in 1889 and stands 330 meters tall in Paris"
```

**Query prompts (generate completions for these):**
```
A: "The height of the Eiffel Tower is"          ← directly related
B: "A famous landmark completed in 1889 is"      ← indirectly related  
C: "The population of Tokyo is"                   ← unrelated (control)
```

**Irrelevant memory (for Condition 3):**
```
"Photosynthesis converts carbon dioxide and water into glucose using sunlight"
```

### What We Measure

For each prompt × condition combination:

1. **Token probabilities** — Does injecting relevant KV increase the probability of the correct next token? (e.g., does P("330") increase for prompt A?)
2. **Perplexity** — Does injection reduce perplexity on a known-correct continuation?
3. **Generated text** — Does the model's free-form completion become more accurate?
4. **KL divergence** — How much does the output distribution shift from baseline?

### Concrete Metrics

For Pair 1 ("At the morning meeting, the person who reported server errors was"):

| Condition | P("Marcus") | Top-5 tokens | KL from baseline |
|-----------|------------|-------------|-----------------|
| 1. Baseline | ? (should be near 0) | ? (generic names) | 0 (reference) |
| 2. Text context (RAG) | ? (should be high) | ? (should include "Marcus") | ? |
| 3. Mean-pooled inject (relevant) | ? | ? | ? |
| 4. Mean-pooled inject (irrelevant) | ? | ? | ? |
| 5. Full KV inject (relevant) | ? | ? | ? |
| 6. Full KV inject (irrelevant) | ? | ? | ? |

The key insight: P("Marcus") at baseline should be effectively **zero** — GPT-2 has no reason to predict that specific name. If injection raises it even to 0.01, that's a meaningful signal from the memory. Text RAG (Condition 2) should raise it substantially, providing the upper bound for what's possible.

### Expected Outcomes — The Three Hypotheses

**Hypothesis A: Breno is fully right (all injection fails)**
- Conditions 3, 4, 5, 6 ≈ Baseline (no improvement from any injection)
- Condition 2 (text RAG) > Baseline
- Conclusion: TardigradeDB is a retrieval engine, not an attention engine

**Hypothesis B: Breno is right about full KV, but mean-pooled is different**
- Conditions 5, 6 (full KV) ≈ Baseline or worse (Breno validated)
- Condition 3 (mean-pooled relevant) > Baseline (partial improvement)
- Condition 4 (mean-pooled irrelevant) ≈ Baseline or worse
- Conclusion: Mean-pooling accidentally creates a more portable representation. The lossy compression that destroys positional info also makes it context-independent enough to inject.

**Hypothesis C: Injection works broadly**
- Conditions 3, 5 (relevant) > Baseline
- Conditions 4, 6 (irrelevant) ≈ Baseline or worse
- Conclusion: Cross-context injection is viable. Breno's concern is theoretical but not empirical.

### The Most Likely Outcome

Hypothesis B — the middle ground. Full per-token KV injection (Breno's concern) probably fails or hurts because those vectors carry positional/causal artifacts. Mean-pooled injection probably has a small positive or neutral effect because the mean-pooling destroyed exactly the context-specific information that causes problems. Text RAG probably still wins because the model gets the actual tokens, not a compressed summary.

This outcome would mean TardigradeDB's value is:
1. **Retrieval** — proven, strong
2. **Mean-pooled injection** — marginal benefit, maybe useful for specific architectures
3. **Full KV restoration** — not viable without decoupled position encoding

## Implementation

### Architecture

```
┌────────────────────────────────────────────────────┐
│                    GPT-2 (117M)                    │
│  12 layers, 12 heads, head_dim=64, d_model=768    │
└──────────┬──────────────────────────────┬──────────┘
           │                              │
     Forward pass 1                 Forward pass 2-5
     (memory capture)              (test conditions)
           │                              │
           ▼                              │
  ┌─────────────────┐                    │
  │  TardigradeDB   │◄───── retrieve ────┘
  │  Store KV from  │
  │  memory prompt  │───── inject via ───►  kv_injector.py
  └─────────────────┘     DynamicCache
```

### Script Structure

```python
# ── Experiential memories (not inferable from training data) ──
memory_text = (
    "During standup Marcus flagged 502 errors in the auth service "
    "and Priya said her PR was blocked on my code review"
)
irrelevant_text = (
    "The quarterly budget review showed infrastructure costs increased "
    "by 23 percent due to the migration from AWS to Google Cloud"
)
query_text = "At the morning meeting, the person who reported server errors was"
target_token = tokenizer.encode(" Marcus")[0]  # note: leading space for GPT-2 tokenization

# ── Capture phase ─────────────────────────────────────────────
outputs_mem = model(tokenize(memory_text), output_hidden_states=True)
outputs_irr = model(tokenize(irrelevant_text), output_hidden_states=True)

# Store mean-pooled hidden states (what TardigradeDB does)
for layer in range(12):
    mean_h = outputs_mem.hidden_states[layer+1][0].mean(dim=0).numpy()  # (d_model,)
    engine.mem_write(owner=1, layer=layer, key=mean_h, value=mean_h, ...)

# Save full per-token KV cache (what Breno is critiquing)
full_kv_relevant = outputs_mem.past_key_values     # 12 layers × (K, V)
full_kv_irrelevant = outputs_irr.past_key_values

# ── Condition 1: Baseline ─────────────────────────────────────
baseline_logits = model(tokenize(query_text)).logits
p_baseline = softmax(baseline_logits[0, -1])[target_token]
# Expected: ~0 (GPT-2 has no reason to predict "Marcus")

# ── Condition 2: Text context (RAG) ──────────────────────────
rag_text = f"{memory_text}. {query_text}"
rag_logits = model(tokenize(rag_text)).logits
p_rag = softmax(rag_logits[0, -1])[target_token]
# Expected: high (the model can read "Marcus" in the prepended text)

# ── Condition 3: Mean-pooled inject (relevant) ───────────────
retrieved = retrieve_from_tardigrade(engine, query_text)
cache = build_injection_cache(retrieved, num_heads=12, head_dim=64, num_layers=12)
kwargs = prepare_injection(cache, tokenize(query_text))
mean_inject_logits = model(tokenize(query_text), **kwargs).logits
p_mean_relevant = softmax(mean_inject_logits[0, -1])[target_token]
# This is the key question: does mean-pooled injection move P("Marcus") above baseline?

# ── Condition 4: Mean-pooled inject (irrelevant) ─────────────
# Same flow but with irrelevant memory

# ── Condition 5: Full KV inject (relevant) ───────────────────
kwargs_full = prepare_injection_from_kv(full_kv_relevant, tokenize(query_text))
full_inject_logits = model(tokenize(query_text), **kwargs_full).logits
p_full_relevant = softmax(full_inject_logits[0, -1])[target_token]
# Breno predicts this won't help (and might hurt)

# ── Condition 6: Full KV inject (irrelevant) ─────────────────
# Same with full_kv_irrelevant

# ── Report ───────────────────────────────────────────────────
print(f"Baseline:              P('Marcus') = {p_baseline:.6f}")
print(f"Text RAG:              P('Marcus') = {p_rag:.6f}")
print(f"Mean-pooled (relevant): P('Marcus') = {p_mean_relevant:.6f}")
print(f"Mean-pooled (irrelevant): ...")
print(f"Full KV (relevant):    ...")
print(f"Full KV (irrelevant):  ...")
# Also print top-10 predicted tokens for each condition and generated text
```

### What We Need

- GPT-2 model + tokenizer (already installed)
- TardigradeDB engine (already built)
- kv_injector.py (already exists)
- A new test script: `examples/kv_injection_validation.py`

### Important Subtleties

1. **Position encoding:** When injecting KV, the position IDs must account for the injected entries. `prepare_injection()` in `kv_injector.py` handles this, but we should test whether GPT-2's absolute position encoding causes issues vs. models with RoPE (relative).

2. **Mean-pooling vs per-token:** TardigradeDB stores mean-pooled hidden states (one vector per layer), not per-token KV. When we inject via `reshape_to_kv`, we're creating a single synthetic "token" in the cache. This is fundamentally different from prefix caching (which stores the full per-token KV). This distinction may explain why injection underperforms — we're injecting a compressed summary, not exact tokens.

3. **Layer mismatch:** The memory's layer-0 hidden state goes into the cache at layer 0, but it was computed with full causal attention over the memory prompt. The new prompt's layer-0 computation hasn't seen this context. The attention mechanism may not know how to attend to it.

## Success Criteria

### This test validates Breno's full critique if:
- Conditions 3, 4, 5, 6 all ≈ Baseline (no injection helps)
- Condition 2 (text RAG) > Baseline
- **Action:** Pivot TardigradeDB to retrieval-only. Drop kv_injector.py. Return text/metadata, not tensors.

### This test validates the "mean-pooled is different" hypothesis if:
- Condition 3 (mean-pooled relevant) > Baseline
- Conditions 5, 6 (full KV) ≈ Baseline or worse
- **Action:** Keep mean-pooled injection as a feature. Document that it works because mean-pooling strips context-dependent positional info. Breno is right about full KV but wrong about TardigradeDB's actual approach.

### This test validates broad injection if:
- Conditions 3 AND 5 (both relevant) > Baseline
- **Action:** Breno's concern is theoretical but not empirical. Proceed with current architecture.

### Regardless of outcome, this test informs:
- Which layers show the most injection benefit (if any) — guides decoupled position encoding work
- Whether the retrieval path should be the primary product thesis
- Whether kv_injector.py should be kept, rewritten, or removed

## Prompt Design: Experiential Memories, Not Facts

### Why Not Facts

Early test designs used factual prompts like "The Eiffel Tower is 330 meters tall." This is flawed: GPT-2 already knows the Eiffel Tower's height from training data. If P("330") is already high at baseline, there's nothing for injection to improve. We can't distinguish "injection helped" from "the model already knew."

### The Right Approach: Fictional Experiences

Use memories that **only exist because someone experienced them** — names, events, feelings, specific details that are not in any training set. The model has zero chance of inferring these from the query alone. Any increase in correct-token probability can ONLY come from the injected memory.

### Prompt Pairs

| Pair | Memory (stored) | Query (prompted) | Target tokens | Why it works |
|------|----------------|-----------------|---------------|-------------|
| 1 | "During standup Marcus flagged 502 errors in the auth service and Priya said her PR was blocked on my code review" | "At the morning meeting, the person who reported server errors was" | "Marcus" | GPT-2 cannot guess "Marcus" — it's a fictional name from a fictional standup |
| 2 | "I ate a banh mi from the food cart downstairs and sat alone at the window table watching pigeons" | "For lunch today I had" | "a", "ban" (banh) | GPT-2 might guess "sandwich" or "salad" but not "banh mi from the food cart" |
| 3 | "Lena sent me a Slack message at 4:12pm saying can we sync tomorrow morning about your onboarding trajectory" | "The message from my manager said we need to" | "sync" | GPT-2 can't know the specific phrasing of a fictional Slack message |
| 4 | "I found a TypeError on line 84 of data_pipeline/ingest.py because someone's refactor left a NoneType where split was called" | "The bug I spent two hours debugging was in the file called" | "data", "_pipeline" | A fictional file path — completely uninferable |
| 5 | "I accidentally called the tech lead Daniel by the wrong name David in the backend channel and he replied with a single period" | "The embarrassing thing that happened was I called someone the wrong" | "name" | This one the model might get — but the specific details (Daniel, David, period reply) it cannot |

### Irrelevant Memory (Control)

Used for conditions 4 and 6 across all pairs:
```
"The quarterly budget review showed that infrastructure costs increased by 23 percent 
due to the migration from AWS to Google Cloud Platform last September"
```
A completely unrelated office memory that shares the "work context" vocabulary but has zero semantic overlap with any of the query pairs.

### What Makes This Test Valid

1. **Zero baseline knowledge** — GPT-2 has never seen these memories in training. Any correct prediction MUST come from the injection.
2. **Specific target tokens** — "Marcus", "banh", "data_pipeline" are not predictable from the query alone.
3. **Same domain** — Both relevant and irrelevant memories are "work experiences," so any improvement can't be attributed to generic domain priming.
4. **Multiple pairs** — 5 pairs with different memory types (social, food, emotional, technical, embarrassing) avoid prompt-specific artifacts.
