# Multi-Memory KV Injection Experiment

**Date:** April 24, 2026
**Model:** Qwen3-0.6B (596M)
**Corpus:** 10 cross-referencing fact sets (2-3 facts each)

## The Question

Can TardigradeDB inject multiple memories simultaneously and produce responses that synthesize information across them — at zero prompt token cost?

## Results

| Approach | Correct | Prompt Tokens | Notes |
|----------|---------|---------------|-------|
| Text RAG (baseline) | **10/10** | 564 | Facts pasted into prompt |
| KV Naive Concat | 3/10 | 243 | Concatenate K/V tensors per layer |
| KV Sequential Recompute | 1/10 | 243 | Process facts sequentially through model |

### Text RAG: 10/10

All 10 cross-referencing queries answered correctly. The model can reason across facts when they're in the prompt as text.

### Naive KV Concatenation: 3/10

Concatenates K/V tensors from multiple packs along the sequence dimension per layer. The model retrieves *some* signal from injected packs (it knows about the people, places, and things mentioned) but hallucinates details instead of pulling them from the second pack.

Examples of failures:
- Expected "Honda Civic" → generated "white car"
- Expected "Whiskers" → generated "Milo"
- Expected "Biscuit" → generated "Milo"
- Expected "Carmen" → generated "the information is ambiguous"

The model knows there's a cat, a dog, a car — but invents the details rather than reading them from the injected KV. This is consistent with the Knowledge Packs paper's finding that naive concatenation fails.

**Root cause:** Each pack's KV cache carries RoPE rotations computed at its original positions. Concatenating packs A (positions [0..15]) and B (positions [0..12]) creates a non-monotonic position sequence [0..15, 0..12]. The model's attention mechanism expects monotonically increasing positions in causal transformers.

### Sequential Recomputation: 1/10

Processes each fact through the model sequentially: compute fact 1's KV, then process fact 2 with fact 1's KV already in cache. This produces contiguous positions and cross-fact attention.

**Worse than naive concat.** The model generated "I don't have access to that information" for most queries.

**Root cause:** The facts are processed as system-role messages sequentially, but the query is then appended as a user-role message. The model sees the accumulated KV from facts 1+2 in its cache, but the query's tokens can only attend to the cached positions — it cannot *re-attend* to the fact content with the query in mind. The KV cache from the facts was computed without knowledge of what question would be asked. The model essentially "forgets" the specific details by the time the query arrives.

## Why Multi-Memory Injection Fails

The failure is architectural, not implementational:

1. **KV caches are computed in isolation.** Each fact's KV cache is produced by a forward pass that doesn't know about other facts or the upcoming query. The K/V representations encode the fact, but the attention patterns (how tokens relate to each other) are fixed at computation time.

2. **Cross-fact reasoning requires co-attention.** When text RAG puts "Tomoko drives a Honda Civic" and "Lucia's instructor is Tomoko" in the same prompt, the model attends to both simultaneously during the query forward pass. The query tokens can directly attend to both "Honda Civic" and "Tomoko" and connect them. With injected KV, the query tokens attend to pre-computed representations that never saw each other.

3. **RoPE breaks cross-pack attention.** Even if the model could cross-reference injected packs, the non-monotonic position IDs from concatenation corrupt the distance-based attention bias that RoPE provides. Tokens that should be "nearby" (from the same logical context) appear at inconsistent positions.

4. **Sequential recomputation doesn't help for cross-referencing.** Knowledge Packs' sequential approach was designed for *additive* context (each fact stands alone as background knowledge). Cross-referencing queries ("What car does Lucia's instructor drive?") require the model to *reason about relationships* between facts during generation — something the pre-computed KV cache cannot support.

## What This Means for TardigradeDB

**Single-memory injection is the sweet spot.** One fact → one answer → zero prompt tokens. This is validated (8/10, byte-identical to text RAG, 46% token savings).

**Multi-memory scenarios should use hybrid approaches:**
- **Retrieval:** Use TardigradeDB's latent retrieval (100% recall at 100 memories) to find the k most relevant memories
- **Delivery:** For single-fact answers, inject via KV. For multi-fact synthesis, paste the retrieved fact texts into the prompt (text RAG delivery, but TardigradeDB retrieval).

This preserves TardigradeDB's value in two ways:
1. Zero-token delivery for simple recall queries (the majority case)
2. Superior retrieval quality for finding what to deliver (latent > embedding for the model's own memories)

## Open Questions

- Is there a way to recompute cross-fact KV that preserves relational reasoning?
- Could a lightweight attention re-scoring pass over concatenated KV fix the position issue?
- Does the Knowledge Packs paper's sequential approach work for *non-relational* multi-memory? (e.g., "Tell me everything about Sonia" where facts are additive, not cross-referencing)
- Could a larger model (3B+) tolerate the RoPE position corruption better?

## Files

| File | Purpose |
|------|---------|
| `experiments/multi_memory_corpus.py` | 10 cross-referencing fact sets |
| `experiments/multi_memory_experiment.py` | A/B/C comparison runner |
| `python/tardigrade_hooks/multi_composer.py` | NaiveConcatComposer + SequentialRecomputeComposer |
| `python/tardigrade_hooks/kp_injector.py` | `retrieve_and_inject_multi()`, `generate_multi()` |
| `tests/python/test_kp_multi_inject.py` | 7 ATDD structural tests |
