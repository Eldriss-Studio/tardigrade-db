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

## Oracle Injection + Higher-K Retrieval (April 24, 2026)

### Oracle Injection: 6/10

Bypassed retrieval and manually injected the correct packs for each query. Result:

| Query | Oracle | Normal | Notes |
|-------|--------|--------|-------|
| Q1: wifi password for apartment 4B | PASS | PASS | Both work — single fact sufficient |
| Q2: car Lucia's instructor drives | **MISS** | MISS | Hallucinated "white car" even with correct packs |
| Q3: cat of neighbor who brought pierogi | **MISS** | MISS | Hallucinated "Milo" even with correct packs |
| Q4: best-selling item at Eduardo's bakery | PASS | PASS | Both work |
| Q5: address of Sonia's dentist | **MISS** | MISS | "information incomplete" even with correct packs |
| Q6: dog of Lucia's piano teacher | **MISS** | MISS | Hallucinated "Milo" even with correct packs |
| Q7: leader of running group at Prospect Park | PASS | MISS | Oracle fixes retrieval-only failure |
| Q8: day Eduardo's restaurant closed | PASS | PASS | Both work |
| Q9: pharmacist at pharmacy on Western | PASS | MISS | Oracle fixes retrieval-only failure |
| Q10: birthday of Lucia's best friend | PASS | MISS | Oracle fixes retrieval-only failure |

**6/10 correct with oracle.** 3 queries (Q7, Q9, Q10) were pure retrieval failures — fixed by oracle. 4 queries (Q2, Q3, Q5, Q6) fail even with correct packs injected — genuine injection/composition failures.

### Higher-K Retrieval

| k | Queries with all facts found |
|---|------------------------------|
| 2 | 0/10 |
| 5 | 0/10 |
| 10 | 4/10 |
| 20 | 5/10 |

Even at k=20, only 5/10 queries find all needed packs. The latent retrieval key for second-hop facts has no semantic overlap with the query.

### What This Means

The ceiling for multi-memory KV injection: **~6/10** on cross-referencing queries (vs 10/10 text RAG). Two independent problems:

1. **Retrieval gap (fixable):** 5/10 second-hop facts unreachable by latent retrieval. Trace graph or entity-linked retrieval could fix this.
2. **Injection gap (harder):** 4/10 queries fail even with correct packs injected. The model hallucinates details instead of reading them from the second pack's KV cache. This is the genuine cross-attention limitation.

## Why 4 Oracle Injections Fail (April 24, 2026)

The 4 queries that fail even with correct packs share a structural pattern: **the query and the answer fact have zero direct vocabulary overlap**. The only connection is through an entity name that the query doesn't mention.

**Failing pattern — no vocabulary bridge:**
| Query asks about | Linking fact | Answer fact | Missing bridge |
|-----------------|-------------|-------------|----------------|
| "Lucia's instructor's car" | instructor = Tomoko | Tomoko drives Honda Civic | Query has no "Tomoko" |
| "neighbor's cat who brought pierogi" | Mrs. Kowalski brought pierogi | Mrs. Kowalski's cat = Whiskers | Query has no "Mrs. Kowalski" |
| "Sonia's dentist address" | dentist at Riverside Dental | Riverside Dental at 742 Elm | Query has no "Riverside Dental" |
| "Lucia's piano teacher's dog" | teacher = Mr. Yamamoto | Mr. Yamamoto has Biscuit | Query has no "Mr. Yamamoto" |

**Passing pattern — direct vocabulary overlap:**
| Query asks about | Answer fact | Shared terms |
|-----------------|-------------|--------------|
| "best-selling at bakery" | "best-selling item is cinnamon swirl" | "best-selling" |
| "running group leader" | "running group leader is Carmen" | "running group", "leader" |
| "restaurant closed day" | "Trattoria Bella closed on Mondays" | "closed" |
| "pharmacist at pharmacy" | "MedPlus pharmacist is James Chen" | "pharmacist" |
| "birthday of best friend" | "Harper's birthday is March 15th" | "birthday" |

### Root Cause: Attention Inside the Model

This is the same problem as retrieval but **inside the model's attention mechanism**. During generation, the query's Q vectors attend to injected K vectors via dot product. If the K vectors (computed from "Tomoko drives Honda Civic") have zero token overlap with the Q vectors (computed from "Lucia's instructor"), the attention score is low and the model ignores that pack's V vectors.

Text RAG succeeds because the model processes all facts as text in a single forward pass — the attention between "Lucia's instructor" and "Tomoko" is computed fresh during that pass. With KV injection, that attention was never computed.

### Architectural Verdict

Multi-memory KV injection works when:
- Facts share direct vocabulary with the query (6/10)
- OR the answer is in a fact that's self-sufficient (single-memory recall)

Multi-memory KV injection fails when:
- Cross-referencing requires entity resolution ("Lucia's instructor" = "Tomoko") that can only happen during a joint forward pass

This is a fundamental property of attention, not a fixable bug. The ceiling for multi-memory KV injection with independently-computed packs is ~60% on cross-referencing queries.

### Recommendations

1. **Single-memory injection** remains the canonical path (8/10, byte-identical to text RAG)
2. **Multi-memory with vocabulary overlap** works (6/10 on oracle) — queries like "what is X's birthday" where "birthday" appears in both query and answer fact
3. **Cross-referencing queries** requiring entity resolution should use **text RAG delivery** with TardigradeDB retrieval — retrieve the memories latently, deliver them as text in the prompt
4. **Hybrid approach**: use single-memory KV injection for direct-recall queries, fall back to text delivery for multi-hop queries. The retrieval layer (latent + Trace graph) determines which memories are relevant; the delivery method (KV injection vs text) depends on query complexity

## Phase 31: Trace-Linked Retrieval — 3/10 → 9/10 (April 24, 2026)

### The Breakthrough

Linking related facts at storage time and following links at retrieval time produces **9/10 correct on cross-referencing queries** — up from 3/10 baseline and 6/10 oracle.

| Approach | Accuracy | Prompt Tokens | Retrieval |
|----------|----------|---------------|-----------|
| Text RAG | 10/10 | 564 | N/A (all in prompt) |
| Baseline KV injection | 3/10 | 243 | 0/10 find both packs |
| Oracle injection (bypass retrieval) | 6/10 | 243 | Manual (correct packs) |
| RoPE-corrected concat | 3/10 | 243 | 0/10 (position fix irrelevant) |
| **Trace-linked injection** | **9/10** | **243** | **10/10 find both packs** |

Result verified reproducible across 2 runs. `do_sample=False` ensures deterministic generation.

### How It Works

**Storage:** `store_linked(["fact A", "fact B"])` stores both facts as KV packs and records a bidirectional link between their pack IDs in a Python-side dictionary.

**Retrieval:** `retrieve_with_trace(query, k=1)` retrieves the top-1 pack via normal latent retrieval, then looks up linked pack IDs and loads them from a local cache. Both packs are composed into a single DynamicCache via NaiveConcatComposer.

**No Rust changes, no recomputation, no training.**

### Why 9/10 Instead of 6/10 (Oracle)

The oracle experiment used `mem_read_pack(query_key, 50)` to load "correct" packs, which returns packs **ordered by retrieval score**. Trace-linked retrieval loads the second pack from a local cache with **score 0.0**, which means the packs arrive in storage order (fact A first, fact B second).

In causal attention, order matters: later tokens attend to earlier ones. When fact A (the linking fact) is first and fact B (the answer fact) is second, fact B's tokens can attend to fact A's tokens during the query's generation forward pass. The oracle's retrieval-score ordering may have placed packs in a suboptimal order for 4 of the 10 queries.

This is a significant finding: **pack ordering affects multi-memory injection quality**. The order in which facts were originally stored (and linked) may naturally reflect their logical relationship, which helps the model's causal attention.

### The One Miss: Q1

Q1 ("What is the wifi password for the person in apartment 4B?") passes with single-memory injection (3/10 baseline) but fails with trace-linked injection (9/10). The model responds: "The information provided is about Sonia's WiFi password, not [apartment 4B]."

This is a regression caused by injecting BOTH packs: the model sees KV from "Sonia's wifi password is mango-cathedral-7" AND "Sonia lives in apartment 4B" — but interprets the second fact as contradicting the query's premise rather than confirming it. The model thinks "apartment 4B" is about Sonia's location, not about whose wifi password it is.

This may be fixable by adjusting pack ordering (put the answer fact last) or by using a smarter composition strategy. It's not a fundamental limitation.

### Honest Assessment

**What's real:**
- Trace-linked retrieval solves the multi-hop retrieval problem (0/10 → 10/10 finding both packs)
- Multi-memory KV injection works at 9/10 on this corpus with correct pack ordering
- 57% fewer prompt tokens than text RAG
- No recomputation, no training, no custom kernels
- Reproducible across multiple runs (deterministic with do_sample=False)

**What to question:**
- Tested on only 10 cross-referencing queries — small corpus
- Only tested on Qwen3-0.6B (596M) — model-dependent results possible
- Python-side trace links are in-memory (not persistent across restarts)
- Pack ordering matters and isn't well-understood yet
- The Q1 regression shows that injecting more packs can sometimes hurt
- Not tested at scale (100+ linked facts, 1000+ total memories)

### Next Steps

1. **Scale test:** Run on a larger corpus (50+ cross-referencing queries, 200+ total memories)
2. **Pack ordering:** Investigate whether "answer fact last" consistently helps
3. **Rust-side trace links:** Move Python-side `_trace_links` to Rust Trace graph for persistence
4. **Persistence:** Cache pack data in Rust so `_pack_data` survives restarts
5. **Larger model:** Test on Qwen2.5-3B to see if model capacity changes the results

## Scale Test: 140 Memories, 20 Queries (April 24, 2026)

100 background single-fact memories + 20 cross-referencing fact pairs (40 facts, trace-linked) = 140 total.

| Approach | Small corpus (21 facts) | Scale (140 facts) |
|----------|------------------------|-------------------|
| Text RAG | 10/10 (100%) | 19/20 (95%) |
| Baseline (no trace) | 3/10 (30%) | ~30% (estimated) |
| Trace-linked injection | 9/10 (90%) | **11/20 (55%)** |

The 9/10 result was real but small-corpus-favorable. At 140 memories with retrieval noise from 100 unrelated facts, accuracy drops to 55%. Still 2x better than no-trace baseline, but the gap with text RAG widens from 10% to 40%.

Even text RAG degraded slightly (100% → 95%) — the 20 new queries are harder and more specific.

## Full Progression

```
Phase 24: Retrieval works ..................... 100% at 100 memories
Phase 25: Single injection works ............. 8/10, byte-identical to RAG
Phase 30: Multi injection fails .............. 3/10 (retrieval problem)
Phase 30A: RoPE correction ................... 3/10 (no effect)
Phase 30B: Oracle injection .................. 6/10 (injection ceiling)
          RAG retrieval comparison ........... 5/10 (RAG also partially fails)
          Higher-k retrieval ................. 5/10 at k=20
Phase 31: Trace-linked retrieval ............. 10/10 find both packs
          Small corpus (21 facts) ............ 9/10 (90%)
          Scale test (140 facts) ............. 11/20 (55%)
```

## Failure Decomposition (April 25, 2026)

Decomposed the 9 failures at 140 memories:

| Failure type | Count | What happened |
|---|---|---|
| Retrieval wrong | 5 | Background memory outscores cross-ref linking fact |
| Trace link failure | 0 | Python-side graph works perfectly |
| Injection failure | 4 | Correct packs found but model misformats answer |

The 5 retrieval failures: the original 100 background memories are detailed narratives ("Went to a poetry reading at a bookstore in Pilsen. A woman read a poem about...") that score higher on content similarity than the shorter cross-ref linking facts ("The bookstore in Pilsen where I did the poetry reading is called Casa Azul"). The retriever correctly picks the best content match — it just doesn't have trace links to the answer.

Of the 4 injection failures, Q20 ("14 months" vs expected "fourteen") is a format mismatch, not a real failure.

## Phase 32: Trace-Boosted Retrieval (April 25, 2026)

**Name: Trace-Boosted Retrieval.** Memories with trace connections get a score boost proportional to their link count. Connected memories are discovery hubs — they bring related facts along via trace traversal. Inspired by PageRank (linked pages rank higher) and Obsidian (backlink-dense notes are more important).

**Implementation:** After retrieving an expanded candidate set (`k * 5` packs), re-rank by:
```
final_score = retrieval_score * (1 + link_count * boost_factor)
```
Then take top-k and follow trace links. `boost_factor = 0.3` in initial test.

### Results

| Metric | Before boost | After boost (0.3) |
|---|---|---|
| Final accuracy | 11/20 (55%) | **14/20 (70%)** |
| First-hop retrieval correct | 15/20 | 15/20 (unchanged) |
| Retrieval failures | 5 | 5 (but 3 queries PASS anyway) |
| Trace failures | 0 | 0 |
| Injection failures | 4 | 1 |

The boost didn't change how many first-hop retrievals were correct (still 15/20) but changed *which* packs ranked highest among expanded candidates. 3 queries that previously failed now pass — the boost promoted different background memories that happened to contain enough information.

### Full Progression

```
No trace baseline:   ~30%
Trace-linked:         55% (11/20)
Trace-boosted (0.3):  70% (14/20)
Text RAG:             95% (19/20)
```

Each step improved. The remaining 6 failures: 5 retrieval (background still outscores despite boost) and 1 injection. Q20 ("14 months" vs "fourteen") is arguably a format match. Effective accuracy may be 75%.

## Open Research

| Question | Status |
|----------|--------|
| Does higher boost_factor (0.5, 1.0) fix more retrieval failures? | Next experiment |
| Would linking details to existing memories (not restated facts) eliminate the competition? | Not tested |
| Does a larger model (3B+) change results? | Not tested |
| Would hybrid delivery (KV + text fallback) reach 95%? | Theoretical |
| Does Rust-side Trace match Python-side links? | Not tested |

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
