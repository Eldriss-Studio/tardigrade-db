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

## Phase 30A: RoPE Position Correction (April 24, 2026)

CacheBlend (EuroSys 2025 Best Paper) showed that RoPE position corruption causes accuracy loss when concatenating independently-computed KV caches. We implemented `RoPECorrectedConcatComposer` using the existing `RoPEPositionEncoder.remap_keys()` to unrotate K vectors at old positions and re-rotate at contiguous new positions.

**Result: 3/10 — identical to naive concat.** Position correction made zero difference. Same 3 queries pass, same 7 fail with identical hallucinated responses.

This proved position corruption is NOT the primary failure mode.

## Retrieval Diagnostic: The Real Problem (April 24, 2026)

A retrieval diagnostic (`experiments/multi_memory_retrieval_debug.py`) revealed the actual cause of failure:

**0/10 queries retrieved all needed packs.** `mem_read_pack` returned only 1 pack per query. Every query found the "linking" fact but never the "detail" fact:

| Query | Found | Never Retrieved |
|-------|-------|-----------------|
| "wifi password for apartment 4B?" | "Sonia's wifi password is mango-7" | "Sonia lives in apartment 4B" |
| "car does Lucia's instructor drive?" | "Lucia's instructor is Tomoko" | "Tomoko drives Honda Civic" |
| "cat of neighbor who brought pierogi?" | "Mrs. Kowalski brought pierogi" | "Mrs. Kowalski's cat is Whiskers" |
| ... (same pattern for all 10) | linking fact found | detail fact MISSING |

**The Phase 30 multi-memory injection experiment was actually testing single-memory injection on cross-referencing queries.** The 3/10 result is expected for single-memory — the model gets one fact but not the one with the actual answer. RoPE correction, HKVD recomputation, and all injection-layer fixes are irrelevant when the needed packs were never retrieved.

### Root Cause: Multi-Hop Retrieval

The retriever scores by query-to-memory similarity. "What car does Lucia's instructor drive?" is semantically close to "Lucia's instructor is Tomoko" but NOT to "Tomoko drives a Honda Civic." The second fact relates to the query only through the entity "Tomoko" — a reasoning hop the retriever can't make.

### RAG Comparison

Standard embedding RAG (e5-small-v2) was tested on the same corpus:

| Retriever | All facts found (top-k) | Partial | Detail fact rank when missed |
|-----------|------------------------|---------|------------------------------|
| TardigradeDB (latent, Top5Avg) | **0/10** | 10/10 | Not in top-5 |
| RAG (e5-small-v2, cosine) | **5/10** | 5/10 | Rank #3 or #4 |

RAG does better (5/10 vs 0/10) because embedding similarity captures entity-name overlap that latent hidden states miss. RAG's failures are marginal — the right fact is rank 3-4, so increasing k from 2 to 4 would recover most. TardigradeDB's failures are total — the right fact doesn't appear in top-5 at all.

Multi-hop is partially universal (both fail), but RAG handles it significantly better.

### What This Means

1. Multi-memory injection was never tested — only 1 pack was injected per query
2. The injection mechanism (naive concat, RoPE correction) may work fine with correct packs — untested
3. The fix is in the retrieval layer, not the injection layer
4. Options: increase k, two-stage retrieval, or Trace graph traversal (link related facts at storage time)

## Remaining Experiments

| Experiment | Purpose | Status |
|------------|---------|--------|
| Oracle injection | Manually inject correct packs to verify injection works | Planned |
| Higher-k retrieval | Test if missing facts appear at k=10/20 | Planned |
| Trace-linked retrieval | Link related facts via causal graph, follow edges on retrieval | Planned |

## Updated File List

| Script | What it tests |
|--------|---------------|
| `experiments/multi_memory_corpus.py` | 10 cross-referencing fact sets |
| `experiments/multi_memory_experiment.py` | Naive concat vs sequential recompute vs text RAG |
| `experiments/multi_memory_rope_corrected.py` | RoPE-corrected concat vs naive vs text RAG |
| `experiments/multi_memory_retrieval_debug.py` | TardigradeDB retrieval diagnostic (which packs found?) |
| `experiments/multi_memory_rag_retrieval_debug.py` | RAG retrieval diagnostic (e5-small-v2 comparison) |
| `python/tardigrade_hooks/multi_composer.py` | NaiveConcatComposer, SequentialRecomputeComposer, RoPECorrectedConcatComposer |
| `tests/python/test_kp_multi_inject.py` | 7 ATDD structural tests for multi-memory injection |
| `tests/python/test_rope_corrected_composer.py` | 7 ATDD tests for RoPE-corrected composer |
