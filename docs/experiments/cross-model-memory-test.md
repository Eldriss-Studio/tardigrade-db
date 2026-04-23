# Cross-Model Memory Test: Sonnet Stores, Opus Retrieves

**Date:** Planned  
**Database:** TardigradeDB (word-hash vectorization)  
**Agent 1 (Experiencer):** Claude Sonnet 4  
**Agent 2 (Rememberer):** Claude Opus 4  
**Script:** `examples/sonnet_memory_test.py`

## Hypothesis

A memory stored by one model (Sonnet) can be meaningfully retrieved by a different model (Opus). Specifically: when two models of different capability levels roleplay the same character, the more capable model should retrieve memories at least as well as the same model — and possibly better, by asking more nuanced queries.

## Why This Matters

Real-world agent systems will involve model upgrades, fallbacks, and mixed deployments:
- An agent runs on Sonnet for cost, but falls back to Opus for complex reasoning
- Memories from Tuesday (Sonnet) need to be accessible on Wednesday (Opus)
- A cheaper model handles routine memory writes; a more capable model handles retrieval when it matters

If memory is only portable within the same model, TardigradeDB becomes a single-model tool. If memory is portable across models, it becomes infrastructure.

## Test Design

### Setup

Same character as the original two-agent test: **Kael**, junior software engineer, day 3 at NovaBridge startup.

### Phase 1: Sonnet Stores (Agent 1)

Claude Sonnet 4 receives the same character prompt and generates 12 vivid experiential memories. These are stored via `sonnet_memory_test.py store "memory 1" "memory 2" ...` using word-hash vectorization.

**Vectorization:** Word-hash (deterministic, model-agnostic). Each word maps to a fixed 768-dim direction via SHA-256 seeded RNG. Similarity = word overlap. This is deliberately weak — it isolates the model-difference variable from the vectorization variable.

### Phase 2: Opus Retrieves (Agent 2)

Claude Opus 4 receives the same character identity but knows nothing about what was stored. It reflects on the day naturally — broad queries first, then specific follow-ups, then negative controls.

**Key difference from original test:** Agent 2 is a *different model* than Agent 1. It may:
- Use different vocabulary when querying (Opus tends to be more precise, Sonnet more conversational)
- Ask different kinds of follow-up questions
- Probe at different levels of abstraction

### Phase 3: Sonnet Retrieves (Control)

Repeat Phase 2 with Claude Sonnet 4 as Agent 2. This is the **same-model control** — identical to the original experiment. Comparing Opus retrieval against Sonnet retrieval isolates the effect of the model swap.

## What We Measure

### Primary: Recall Rate

| Condition | Agent 1 (Store) | Agent 2 (Retrieve) | Expected recall |
|-----------|-----------------|-------------------|----------------|
| A: Same-model (control) | Sonnet | Sonnet | ~92% (matches original) |
| B: Cross-model | Sonnet | Opus | ? |
| C: Reverse cross-model | Opus | Sonnet | ? |

### Secondary: Query Vocabulary Overlap

For each (query, correct memory) pair, measure the word overlap between the query text and the memory text. With word-hash vectorization, word overlap IS the similarity signal. If Opus uses different words than Sonnet when recalling the same experience, retrieval scores will differ.

Example:
```
Memory:  "ate a banh mi from the food cart downstairs"
Sonnet:  "what did I eat for lunch"              → overlap: "eat" (1 word)
Opus:    "what food did I consume at midday"      → overlap: "food" (1 word, different word)
```

Both queries work, but through different word overlap paths.

### Tertiary: Query Quality

Qualitative comparison of how each model queries:
- Does Opus ask more targeted questions?
- Does Opus probe for emotional/social memories differently?
- Does Opus generate more or fewer follow-up queries?

## Expected Outcomes

### Hypothesis A: No Degradation

Opus retrieves at ≥92% recall (same as Sonnet-Sonnet). Word-hash vectorization is model-agnostic, and Opus uses enough overlapping vocabulary that retrieval works identically.

**Implication:** Cross-model portability works at the text/hash level. TardigradeDB's memory is model-independent when using model-agnostic vectorization.

### Hypothesis B: Opus Retrieves Better

Opus retrieves at >92% recall — possibly catching the "Tomasz keyboard" memory that Sonnet missed. Opus may use more varied vocabulary or ask more comprehensive questions, hitting memories that Sonnet's queries missed.

**Implication:** A more capable retrieval model compensates for weak vectorization. Model-agnostic memory benefits from model diversity.

### Hypothesis C: Opus Retrieves Worse

Opus retrieves at <92% recall. Opus may use more abstract or precise vocabulary that has *less* word overlap with Sonnet's conversational memory phrasing.

**Implication:** Vocabulary mismatch is real. Cross-model portability needs better-than-word-hash vectorization (embeddings or KV cache) to work reliably.

## Execution Plan

### Prerequisites

```bash
cd ~/Dev/tardigrade-db
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml
```

### Step 1: Sonnet stores memories

Launch a Claude Sonnet subagent with the Kael character prompt. Have it generate 12 experiential memories and store them:

```bash
python examples/sonnet_memory_test.py store \
  "memory 1 text" \
  "memory 2 text" \
  ...
```

### Step 2: Opus retrieves (cross-model)

Launch a Claude Opus subagent with the same character identity. Have it query naturally:

```bash
python examples/sonnet_memory_test.py query "What happened at work today"
python examples/sonnet_memory_test.py query "Did anything embarrassing happen"
# ... follow-up queries based on results
```

### Step 3: Sonnet retrieves (control)

Repeat Step 2 with a Claude Sonnet subagent, same queries where possible.

### Step 4: Compare recall, scores, and query patterns

## Connection to Open Questions

This experiment addresses **Q5 (Cross-Model KV Portability)** from `docs/technical/open-questions.md`, but at the text/hash layer rather than the KV layer. Results inform:

1. If word-hash portability works → model-agnostic retrieval is viable, store text + metadata for cross-model compatibility
2. If vocabulary mismatch hurts → need shared embeddings or a model-agnostic embedding index alongside model-specific KV
3. Regardless of outcome → establishes baseline for future KV-level cross-model experiments (same family, different size)

## What This Does NOT Test

- **KV-level portability** — Word-hash vectors are model-agnostic by construction. This test cannot distinguish "models are compatible" from "the vectorization ignores the model." A future experiment with model-specific hidden states would test true KV portability.
- **API model internals** — Claude Opus/Sonnet are API models; we cannot access their KV cache. This test uses them as agent brains, not as tensor sources.
- **Injection** — This tests retrieval only (does the right memory come back?), not injection (does injecting KV help generation?).
