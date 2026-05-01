# Experiments

Empirical tests validating TardigradeDB's core thesis: that persistent memory in latent space — using raw KV cache tensors, not text or embeddings — enables a fundamentally different kind of LLM memory.

## Completed Experiments

### [Two-Agent Memory Cycle](two-agent-memory-test.md)

**Date:** April 22, 2026  
**Status:** Complete

Two independent Claude Sonnet agents — one generating experiential memories, one retrieving them blind — communicate exclusively through TardigradeDB. Tests both a word-hash baseline and real GPT-2 KV cache tensors.

**Key findings:**
- 92% recall with word-hash vectors (weakest possible baseline)
- 80% recall with GPT-2 KV cache tensors, but with 100x stronger signal-to-noise separation
- ~1,600-point score gap between genuine memories and noise makes confidence thresholding practical
- Missed memories are model-quality limitations (GPT-2, 117M params), not architectural ones
- One missed memory (a peripheral observation about a coworker's keyboard) accidentally modeled realistic human memory decay

**Scripts:** `examples/sonnet_memory_test.py` (word-hash), `examples/kv_memory_test.py` (KV cache)

### [KV Injection Critique & Validation](kv-injection-critique.md)

**Date:** April 22, 2026  
**Status:** Complete — validated empirically

External critique questioning whether cross-context KV injection works. Three documents cover this:
- **[Critique](kv-injection-critique.md)** — The original peer review concerns and analysis
- **[Test design](kv-injection-validation-test.md)** — 6-condition experiment design with experiential memories
- **[Results](kv-injection-results.md)** — Measured outcomes across 5 prompt pairs (30 measurements)

Key findings from [validation results](kv-injection-results.md):
- **Full per-token KV injection works** — 26x to 829x improvement over baseline, matching or exceeding Text RAG
- **Mean-pooled injection is broken** — mathematical category error (hidden states ≠ K/V projection space, causes collapse to "?" token at 78-90%)
- **Q4 quantization preserves 89% of injection quality** — TardigradeDB's storage approach is viable
- **Irrelevant KV injection ≈ baseline** — confirms improvement is genuine semantic transfer, not noise
- **Selective token injection fails** — salience heuristic (hidden state norm) doesn't identify the tokens that matter for recall

**Automated test coverage:** 5 ATDD tests in `tests/python/test_kv_injector.py` validate the injection pipeline mechanism (reshape, cache extend, GPT-2 integration, multi-cell, round-trip). 5 ATDD tests in `tests/python/test_sweep.py` validate background governance sweep (Active Object pattern).

**Architectural implication:** Mean-pooled vectors work for **retrieval** (search index). Full per-token KV is necessary for **injection** (attention augmentation). The emerging architecture uses mean-pooled as the index key and full KV as the stored value.

### [Sonia Parallel Subagent Validation](sonia-subagent-parallel-test.md)

**Date:** April 23, 2026  
**Status:** Complete

Two parallel Codex subagents (different agent models) executed separate Sonia experiment scripts and reported recall/SNR outcomes independently.

**Key findings:**
- `sonia_real_kv_cache.py`: per-token real-KV beat mean-pool on recall in this run (`75.0%` vs `62.5%`, +12.5 points)
- `sonia_production_sim.py`: recall was equal between modes (`31.2%` each) but per-token showed much larger SNR separation
- Both runs completed successfully with no operational blockers

### vLLM KV Connector — End-to-End Round-Trip + Architectural Discovery

**Date:** April 26-27, 2026
**Status:** Complete — save/load plumbing validated; architectural limit of v1 API discovered

First validation that TardigradeDB plugs into a production LLM serving framework. The `tardigrade_vllm.connector.TardigradeConnector` implements vLLM's KV Connector v1 API, captures KV during generation, persists packs to TardigradeDB, and supports cross-process state sync via `Engine.refresh()`. Tested in WSL2 + Ubuntu 24.04 with an RTX 3070 Ti.

**Setup:** vLLM 0.19.1, PyTorch 2.10 (CUDA 12.8), Qwen3-0.6B (28 layers, 8 KV heads, head_dim=128, kv_dim=1024) loaded in bf16 with `enforce_eager=True` and `max_model_len=512`.

**Save path findings (validated):**
- Per-request slot extraction via `RequestSlotResolver` (Strategy + Parameter Object pattern) — saves the actual request's KV blocks, not placeholder block 0.
- One pack per request via fingerprint dedup (was 20 packs per 20-token completion).
- Cross-session persistence proven: vLLM run #1 writes packs, run #2 reads them on startup.
- `Engine.refresh()` (Memento re-application) — scheduler-side connector re-syncs from worker-side writes in place. SLB dimension auto-adapts when first cells arrive.

**Load path findings (validated mechanically):**
- `start_load_kv` correctly writes K/V data into GPU block slots — verified with Spy-pattern CPU tests using real `torch.Tensor.copy_()`.
- Per-method ATDD tests for `get_num_new_matched_tokens` catch silent-failure bugs (narrowed exception handling, structural attribute checks).

**vLLM 0.19 API drift (fixed):**
- `build_connector_meta()` must return non-None (asserted in gpu_model_runner)
- `request_finished()` must return `(bool, dict|None)` tuple
- `kv_layer` is a single Tensor `[2, blocks, bs, h, d]` (K stacked with V along dim 0)
- Layer names: `"model.layers.N.self_attn.attn"` (not `"layers.N.self_attn"`)
- bf16 must be cast to float32 in torch before `.numpy()`
- `__init__` must accept `kv_cache_config` as second arg (deprecated signature warning)

**Critical architectural discovery — vLLM v1 KV Connector is prefix-cache only:**

The v1 connector API (`base.py:451-480`) requires **token-identical prefix matching**. `get_num_new_matched_tokens` returns "how many of this prompt's first N tokens have pre-computed KV" — vLLM skips prefill for those positions and trusts the loaded KV as-if-computed. All in-tree connectors (LMCache, SharedStorage, ExampleConnector) are prefix-cache offloads.

**There is no mechanism in the v1 API for cross-prompt KV injection** — injecting KV from prompt A to influence generation of prompt B. The connector simply cannot make the model produce different output for the same prompt by loading unrelated stored KV. Verified by:
1. Reading `base.py` contract documentation (explicit: "only consider the largest prefix of prompt-tokens for which KV cache is actually available")
2. Tracing the scheduler path in `scheduler.py` (sets `num_computed_tokens = local + external`, then attention only computes positions beyond that)
3. Testing with a synthetic fact ("Zorblax discovered the moons of Quthar in 2089") — cold and primed generations are byte-identical because the connector has no way to inject cross-prompt memory through this API

**Implication:** The vLLM connector is a valid **persistent prefix-cache accelerator** (same prompt → reuse stored KV, skip prefill). It is NOT the vehicle for the "cross-session memory" pitch. That pitch is served by the HuggingFace `KnowledgePackStore` path, which passes `past_key_values` directly to `model.generate()` and is not bound by the prefix-cache contract. Documented results via that path: 8/10 novel facts byte-identical to Text RAG, 46% fewer prompt tokens.

**Tests (27 total):** 15 CPU (`test_vllm_load_path.py`: resolver, spy, per-method ATDD, signatures) + 4 CPU (`test_vllm_format.py` + `test_vllm_connector.py`) + 3 CPU (`test_engine_refresh.py`) + 6 GPU (`test_vllm_integration.py`: save, pack contents, semantic match, coherent output, synthetic-fact A/B [RED — architecturally impossible via v1], accumulation) + 1 GPU (`test_vllm_cross_session.py`: cross-process persistence).

**Next steps identified:**
1. ~~Prefix-cache reframing: per-user synthetic "memory prefix" that's token-identical across requests → stock vLLM prefix-cache serves it.~~ **Done — [Path 2 adapter built](memory-prefix-adapter.md): `MemoryPrefixBuilder` assembles governed memory prefixes. Next: wire into vLLM connector.**
2. ~~Verify KnowledgePackStore's 8/10 claim with synthetic facts (Zorblax-style) via HF directly — the real "memory" A/B test.~~ **Done — [Path 1 verified](synthetic-kv-injection.md): 9/10 with fully synthetic gibberish, 100% recall ratio vs text RAG.**
3. Research SGLang's connector contract for a potentially more flexible production serving path.
4. Optional: vLLM custom attention plugin for true cross-prompt KV injection (heavy, requires fork).

### [Path 2: Memory Prefix Adapter](memory-prefix-adapter.md)

**Date:** April 27, 2026  
**Status:** Complete — end-to-end verified on GPU with vLLM 0.19.1 + Qwen3-0.6B (4/4 tests passing)

A deployment adapter that bridges TardigradeDB's governed memory to vLLM's prefix-cache contract. `MemoryPrefixBuilder` assembles a deterministic text prefix per owner from their Core/Validated memories, ordered by importance. Because the same owner's prefix is token-identical across requests, vLLM's stock prefix-cache serves it at zero prefill cost.

**Key components:**
- **`engine.list_packs(owner=N)`** — new Rust engine API enumerating all packs for an owner with tier, importance, and text metadata. Sorted by importance descending.
- **`MemoryPrefixBuilder`** — Facade that filters by governance tier (Core always, Validated optionally, Draft never), applies optional token budget, and formats via pluggable strategies.
- **`PrefixResult`** — value object with `text`, `version` (content hash for staleness detection), `pack_ids`, and `token_estimate`.
- **Format strategies:** `BulletListFormat` ("Memory context:\n- fact1\n- fact2") and `TierAnnotatedFormat` ("- [Core] fact1\n- [Validated] fact2").

**Design decisions:**
- This is an output adapter, not a pivot. The HuggingFace direct-injection path (Path 1) coexists alongside — both read from the same engine, same storage, same governance. The prefix path just formats memories as text instead of injecting KV tensors.
- Version is a SHA-256 content hash (truncated to 64 bits) — deterministic, no external state, changes when memory content changes.
- Token budget drops lowest-importance memories first when the prefix exceeds the limit.

**Tests (15 total):** 4 Rust acceptance tests for `list_packs` (empty, all, owner filter, importance sorting) + 11 Python ATDD tests for `MemoryPrefixBuilder` (empty, tier filtering, determinism, ordering, budget, versioning, format strategy, newline escaping).

**Scripts:** `python/tardigrade_hooks/prefix_builder.py`, `python/tardigrade_hooks/prefix_format.py`

**Wired (April 27, 2026):** `VLLMMemoryClient` (`python/tardigrade_vllm/prefix_client.py`) wraps `MemoryPrefixBuilder` for vLLM serving. `prepare_prompt(query)` prepends the governed prefix as raw text; `prepare_messages(messages)` injects it as/into a system message for OpenAI-style chat APIs. 13 ATDD tests validate prompt composition, message merging, owner isolation, budget enforcement, version tracking, and input immutability.

### [Path 1: Synthetic-Fact KV Injection Verification](synthetic-kv-injection.md)

**Date:** April 27, 2026  
**Status:** Complete — **PASS**

The existential test for TardigradeDB's core value proposition: does KV injection transfer knowledge the model has *never seen*? Uses 10 fully synthetic gibberish facts (nonsense proper nouns, fake units, made-up numbers) that cannot exist in any training corpus. Compares text RAG (fact pasted in prompt) vs KV injection (stored KV tensors injected via `KnowledgePackStore`).

**Model:** Qwen3-0.6B (596M) on CPU, float32  
**Corpus:** 10 gibberish facts (e.g., "The capital of Vrenthar is Zyphlox-9", "Agent Snibblex reported that the vault code is 9-Quornth-44")

**Key findings:**
- **Text RAG: 9/10** — one miss due to `</think>` tag truncation at max_new_tokens boundary
- **KV Injection: 9/10** — matches text RAG exactly (100% recall ratio vs the 70% gate)
- **236 total prompt tokens saved** (~23.6 per query average)
- KV injection got one fact right (#9, "88.2 frenzils") that text RAG missed
- The one KV miss (#8) dropped the quantity "5 klombs" but kept the substance "purazine"
- **Verdict:** KV injection transfers truly novel knowledge. Any correct recall is unambiguous — these gibberish strings cannot come from model weights.

**Scripts:** `experiments/synthetic_kv_injection_experiment.py`, `experiments/synthetic_facts_corpus.py`  
**Tests:** 7 ATDD tests in `tests/python/test_synthetic_kv_injection.py` (6 structural on GPT-2, 1 gate on Qwen3-0.6B)

### [KV Cache Validation — Full Progression](kv-cache-validation.md)

**Date:** April 22-23, 2026
**Status:** Complete — three major discoveries

Systematic exploration of what to store and how to retrieve, tested on Sonia (16 diverse life memories) with GPT-2, Qwen3-0.6B, and Qwen2.5-3B.

**Three discoveries:**
1. **Store K projections, not hidden states** — hidden states produce gravity wells (31.2%), K projections doubled recall (62.5-75%)
2. **K*K per-token matching fails** — K vectors share a massive common component across all sequences (position-0 cross-sentence dot = 6281 for unrelated text). Per-token K*K got 25%, worse than mean-pool
3. **Query with Q, store K (Q*K)** — matches how attention actually works. The fixed Q*K per-token pipeline gets 68.8% recall with 8 unique top-1 memories, and now exercises encoded Q tokens against encoded K tokens through max-sim scoring

**Full progression:** 31.2% (hidden) → 25% (K*K per-token) → 62.5% (K*K mean-pool) → 75% (K*K per-token manual) → 68.8% (Q*K per-token pipeline, 16 memories)

**100-memory scale test:** Q*K recall dropped to **40%** at 100 memories. Gravity well returned. Traditional RAG baseline achieved 100% on the same corpus.

**Signal audit verdict: `LAYER_OR_HEAD_PROBLEM`.** The correct memories ARE in the latent signal (R@100 = 100%). Hidden states + top5_pair_avg achieves 100% recall with 10% false positive rate.

**Engine pipeline validation: 100% recall.** Hidden states + Top5Avg scoring through the full engine pipeline (Q4 quantization, INT8 scoring, per-token retrieval) achieves **30/30 recall, all at rank #1, no gravity well, 97ms latency**. This matches traditional RAG using the model's own latent representations.

**Open question: vague queries.** All test queries use specific vocabulary. Real agent queries would be vaguer ("How is Lucia doing?"). This is the next test.

**Scripts:** `experiments/scale_100_hidden_top5.py` (100% result), `experiments/scale_100_qk_diagnostics.py` (diagnostic suite), `experiments/scale_100_rag_baseline.py` (RAG baseline)

### [P1: Architectural Unification](p1-architectural-unification.md)

**Date:** April 28, 2026
**Status:** Complete — 249 Rust tests, 194 Python tests

Architectural gap analysis found 18 disconnections between TardigradeDB's layers. Priority 1 wired existing features together:

**Key changes:**
- **Active governance:** Tier-based retrieval boost (Core 1.25×, Validated 1.1×, Draft 1.0×). `evict_draft_packs()` for controlled cleanup. Two bugs found and fixed: `mem_read_pack` not re-sorting after tier boost; `mem_read` early-exit truncating before final sort.
- **WAL checkpointing:** `refresh()` truncates WAL after successful replay — prevents unbounded growth.
- **Text store consolidation:** Killed `text_registry.json` sidecar. Rust `TextStore` is sole source of truth.
- **Dead code removal:** `batch_cache.rs` and `arena.rs` stubs deleted.
- **CI fixes:** Broken cross-crate doc links and typos config fixed (CI had been red for 5+ commits).

**Verified:** Both e2e demos (GPT-2 hook pattern + Qwen3-0.6B KnowledgePackStore pipeline) pass end-to-end.

### [P2+P3: Production Story & Differentiators](p2-p3-production-and-differentiators.md)

**Date:** April 28, 2026
**Status:** Complete — 263 Rust tests, 216 Python tests

P2 closed production credibility gaps; P3 turned Rust-only features into Python-accessible capabilities:

**P2 key changes:**
- Honest docs: `spec.md` and `tdd.md` rewritten, unimplemented claims moved to "Future Work"
- `Engine::status()` for monitoring; `Engine(path, segment_size=, vamana_threshold=)` for config
- 3 retrieval key strategies (LastToken, MeanPool, Projected) with named constants

**P3 key changes:**
- Semantic edge types: `add_pack_edge` with Supports/Contradicts; `pack_supports`/`pack_contradicts` queries
- SynapticBank exposed to Python: `store_synapsis`/`load_synapsis` with f32↔f16 at boundary
- Multi-agent acceptance: 3 agents × 5 packs, 12 tests all pass immediately (owner isolation validated)

### [SGLang KV Connector Investigation](sglang-investigation.md)

**Date:** April 28, 2026
**Status:** Complete — NOT VIABLE

SGLang's RadixAttention architecture is strictly prefix-based (same as vLLM v1). `match_prefix()` operates on token sequence identity only. No mechanism for cross-prompt KV injection. Both major production LLM serving frameworks are confirmed prefix-only.

**Implication:** Path 1 (HuggingFace direct injection) remains the only working approach for zero-token KV injection. Path 2 (memory prefix) works for production serving. Path 3 (SGLang) is closed. Path 4 (custom attention plugin) is the only remaining theoretical option.

## Research Status — What's Proven, What's Not

### Proven (tested with data)

| Finding | Evidence | Confidence |
|---------|----------|------------|
| **KV injection transfers novel knowledge** | 9/10 synthetic gibberish facts recalled on Qwen3-0.6B, matching text RAG | High — nonsense strings can only come from injected KV |
| **Per-token Top5Avg retrieval works at scale** | 100% R@5 at 2,000 memories, no gravity well, no degradation | High — 30 queries, clean scaling curve |
| **Vamana acceleration works** | 1.44x latency speedup at 1K memories with zero recall loss | High — Criterion benchmarked |
| **Q4 quantization preserves retrieval** | 89% of injection quality preserved through Q4 pipeline | High — measured in injection results |
| **Mean-pooling is broken for retrieval** | 10% same-model recall (vs 100% per-token) | High — repeated across models |
| **Position 0 must be skipped** | Including attention sink drops recall from 96.7% to 3.3% | High — immediate, reproducible |
| **Same-family cross-model works** | 90% R@5 via per-token linear projection (Qwen3-0.6B → 1.7B) | High — closed-form, no training loop |
| **Cross-family cross-model is viable** | 76.7% R@5 via MLP adapter (Qwen3 → GPT-2, 400K params) | Medium — single corpus, single model pair |
| **Dimension projection is free** | Truncation and orthogonal projection have zero recall cost same-model | High — tested truncation + random orthogonal |
| **Energy distribution differs across model sizes** | Qwen3-0.6B: 71% energy in first quarter; 1.7B: 52% in last quarter | High — direct measurement |
| **Linear alignment has a ceiling** | Procrustes plateaus at ~47% cross-family (500 training samples) | High — scaling curve flattens |
| **vLLM v1 is prefix-cache only** | No mechanism for cross-prompt KV injection in the API | High — traced scheduler code + synthetic fact test |
| **Thread-safe concurrent access works** | Arc<Mutex<Engine>> with GIL release, 4 ATDD tests pass | High |
| **Crash recovery works** | Truncated segment + truncated WAL both recover cleanly | High — acceptance tests |
| **Vague queries degrade to ~46% R@5** | 100 specific queries: 100% R@5. 100 moderate queries: 45% R@5. 100 vague queries: 48% R@5. The cliff is binary — vocabulary overlap vs not — not a gradient of vagueness. | High — 100 queries per tier, 10 phrasings × 10 domains |
| **Moderate and vague are indistinguishable** | Moderate (45%) and vague (48%) show no statistical difference. The critical factor is whether the query shares exact vocabulary with the stored memory, not how specifically it's phrased. | High — 200 total non-specific queries |
| **Gravity wells in vague retrieval** | Short generic queries route to Fitness and Work domains regardless of intent. Hidden states for vague queries converge to a shared high-energy pattern. R@5 == R@10 — misses are categorical, not rank-based. | High — consistent across all phrasings |

### Not Yet Tested (roads untravelled)

#### Highest priority — determines real-world viability

| Experiment | Why it matters | Risk if untested |
|-----------|---------------|-----------------|
| ~~**Vague queries**~~ | **TESTED** — see Proven table above. 46% R@5 for non-specific queries. The cliff is vocabulary overlap, not vagueness. Retrieval needs augmentation (context signals, domain priors, re-ranking) for vague queries. | — |
| **RoPE injection** | All production models (Llama, Qwen, Mistral) use rotary position encoding. KV tensors encode position via RoPE. Injecting KV from position N into position M breaks attention if RoPE isn't handled. | KV injection might not work at all on production models without RoPE decoupling. The HF direct path bypasses this by injecting into a fresh context, but that's a workaround. |
| **Scale beyond 2K** | Proven at 2K memories. Untested at 10K, 100K. Latency is linear (3s at 2K). At 100K it would be ~150s per query without acceleration. Does Vamana's 1.44x scale, or plateau? | The "database" claim requires at least 10K-memory validation. |

#### Medium priority — improves quality and trust

| Experiment | Why it matters |
|-----------|---------------|
| **Adversarial retrieval** | What happens with contradictory memories? ("Meeting at 3pm" vs "Meeting moved to 5pm"). Does governance (recency decay, importance) surface the right one? |
| **Confidence thresholding** | When should the engine say "I don't remember"? 10% false positive rate was measured but never addressed. No calibrated threshold exists. |
| **Multi-session memory** | "What happened last week?" requires temporal awareness. Engine stores timestamps but scoring ignores time entirely. |
| **Cross-model KV injection** | We proved cross-model *retrieval* (finding memories). Never tested cross-model *injection* (putting Qwen's K/V tensors into GPT-2's attention). Retrieval uses hidden states; injection uses K/V projections — different tensors. |
| **Governance under load** | AKL promotion/demotion/decay is unit-tested. Never tested at scale. With 10K memories and continuous access, does importance scoring create its own gravity wells? |
| **Storage cost at scale** | How much disk per memory with Q4? Is 100K memories practical on a consumer SSD? Nobody measured. |

#### Lower priority — research directions

| Experiment | Why it matters |
|-----------|---------------|
| **Per-head scoring** | Current approach concatenates all attention heads. Per-head scoring might capture different relevance types (syntactic vs semantic). |
| **False positive calibration** | Score thresholding to reduce 10% negFP rate. |
| **Model version regression** | Same architecture, different training checkpoint (Qwen3-0.6B v1 vs v2). Do stored memories survive weight updates? Different from cross-model. |
| **Cross-family with more model pairs** | MLP adapter tested on one pair (Qwen→GPT-2). Does it generalize to Llama→Mistral, etc.? |
| **Canonical representation space** | Instead of N² adapters, train N adapters to a shared space. Reduces adapter count from quadratic to linear. |

### Completed Experiments Table

| Experiment | Status |
|-----------|--------|
| [Two-Agent Memory Cycle](two-agent-memory-test.md) | Complete |
| [KV Injection Critique & Validation](kv-injection-critique.md) | Complete |
| [Sonia Parallel Subagent](sonia-subagent-parallel-test.md) | Complete |
| [vLLM KV Connector](../experiments/README.md#vllm-kv-connector) | Complete |
| [Path 2: Memory Prefix Adapter](memory-prefix-adapter.md) | Complete |
| [Path 1: Synthetic-Fact KV Injection](synthetic-kv-injection.md) | Complete — 9/10 |
| [KV Cache Validation](kv-cache-validation.md) | Complete — 100% recall |
| [P1: Architectural Unification](p1-architectural-unification.md) | Complete |
| [P2+P3: Production & Differentiators](p2-p3-production-and-differentiators.md) | Complete |
| [SGLang Investigation](sglang-investigation.md) | Complete — NOT VIABLE |
| Scale recall benchmark (100→2K memories) | Complete — 100% R@5 at 2K |
| Latency benchmark (Vamana vs brute-force) | Complete — 1.44x speedup |
| Cross-model retrieval (same-family + cross-family) | Complete — 90% same-family, 77% cross-family with MLP |
| vLLM connector — semantic save | Complete |
| vLLM cross-session retrieval | Complete |
| GQA K-expansion | Complete |
| Hidden states + Top5Avg validation | Complete — 30/30 |
| 100-memory Q*K scale test | Complete — 40% (superseded by hidden states path) |
| Traditional RAG baseline | Complete — 100% |
| Vague query retrieval (100 queries per tier) | Complete — specific 100%, moderate 45%, vague 48% R@5 |

## Running Experiments

### Prerequisites

```bash
cd ~/Dev/tardigrade-db
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml

# For KV cache tests
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

### Word-hash test
```bash
python examples/sonnet_memory_test.py store "memory 1" "memory 2"
python examples/sonnet_memory_test.py query "related question"
python examples/sonnet_memory_test.py info
```

### KV cache tensor test
```bash
python examples/kv_memory_test.py store "memory 1" "memory 2"
python examples/kv_memory_test.py query "related question"
python examples/kv_memory_test.py info
```
