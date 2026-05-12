# LLM Agent Reformulation Strategy — Experiment Design

## Problem

Qwen3-0.6B LoCoMo retrieval is capped at 68.2%. Exhaustive elimination across 8 techniques — whitening, reweighting, multi-layer fusion, keyword RLS, embedding RLS, generative RLS (Qwen2.5-3B), multi-phrasing, chunked ingestion — all yielded 0% improvement. Root cause confirmed: vocabulary mismatch between queries and stored conversation text. Specific queries hit 100% recall — the latent-space geometry works when vocabulary aligns.

The existing `GenerativeReformulationStrategy` uses Qwen2.5-3B to rephrase queries, but a 3B model still lacks the world knowledge to bridge "athletic achievements" → "ultramarathon runner". No small model can reason about domain-specific vocabulary bridges.

## Hypothesis

A capable LLM (DeepSeek-chat) with genuine world knowledge can bridge vocabulary gaps that no small model can. By generating reformulated queries that use the vocabulary the stored memory actually contains, the existing RLS loop can retrieve what the original query misses.

**What this proves or disproves:** Whether vocabulary bridging powered by world knowledge is sufficient to break the 68.2% ceiling.
- **If positive (any score movement):** Agent-driven retrieval is the path forward → design the full MCP agent protocol.
- **If zero (68.2% again):** The bottleneck is deeper than vocabulary — possibly latent-space geometry fundamentally can't express these relationships at any model size. Revisit Experiment A (larger capture model) as fallback.

## Architecture — Strategy Pattern Integration

The new `LLMAgentReformulationStrategy` plugs into the existing Strategy hierarchy:

```
ReformulationStrategy (ABC)
├── KeywordExpansionStrategy        ← hand-crafted synonyms (0% on LoCoMo)
├── MultiPhrasingStrategy           ← template-based (0% on LoCoMo)
├── EmbeddingExpansionStrategy      ← embedding NN (0% on LoCoMo)
├── GenerativeReformulationStrategy ← Qwen2.5-3B local (0% on LoCoMo)
└── LLMAgentReformulationStrategy   ← NEW: DeepSeek API (world knowledge)
```

**Interface:** `reformulate(query_text: str | None) -> list[str]`

The strategy calls DeepSeek, parses the response, returns 3-5 reformulated query strings. The existing `ReflectiveLatentSearch` loop handles everything else unchanged (retrieve → confidence check → reformulate → re-retrieve → RRF fusion).

**LLM provider:** Self-contained HTTP call to the Chat Completions API within the strategy class (same shape as `DeepSeekProvider` in `python/tdb_bench/evaluators/providers.py` — `urllib.request` POST, JSON parse). No shared helper extraction — the evaluator provider is coupled to judging concerns, and extracting a common HTTP layer for a hypothesis test is premature abstraction.

**Configuration:**
- `TDB_RLS_MODE=agent` — activates the new strategy
- `DEEPSEEK_API_KEY` — already in `.env.bench`
- `TDB_RLS_AGENT_MODEL` — defaults to `deepseek-chat`

## Prompt Design — Vocabulary Bridge

```
You are helping search a memory system that stores conversations verbatim.
It retrieves by matching words in the stored text, not by understanding meaning.

Generate 3-5 alternative search queries that use different vocabulary
the stored conversation might actually contain.

Original query: "{query_text}"

Think about what specific words a real conversation would use instead of
the abstract terms in the query. Output one query per line, nothing else.
```

**Why this prompt (not generic rephrasing):** The existing `GenerativeReformulationStrategy` already does generic rephrasing with Qwen2.5-3B and got 0%. This prompt encodes our diagnosis — the problem is vocabulary mismatch specifically — and directs the model to reason about what words the stored conversation would actually use.

**Response parsing:**
- Split on newlines
- Strip numbering prefixes (`1.`, `-`, `•`)
- Filter empty lines
- Cap at 5 reformulations (defensive)
- On parse failure or API error → return empty list (RLS falls back to original query)

## ATDD Acceptance Criteria

### Unit-level (`test_rls.py`)

1. `LLMAgentReformulationStrategy.reformulate()` returns `list[str]` with 2-5 items given a query
2. Strategy returns empty list on API failure (graceful degradation)
3. Strategy respects `TDB_RLS_AGENT_MODEL` env var for model selection
4. Strategy returns empty list when `query_text` is None
5. Prompt contains vocabulary-bridging instruction (not generic rephrasing)

### Integration-level (benchmark run)

6. `TDB_RLS_MODE=agent` activates the new strategy in `ReflectiveLatentSearch`
7. Full LoCoMo benchmark completes without crash (1,542 items)
8. Output JSON records `rls_mode=agent` in metadata

### Hypothesis test (the experiment)

9. LoCoMo score with `agent` mode vs 68.2% baseline — any delta proves the model matters
10. Per-category breakdown: improvement concentrated in vague/moderate queries (not specific) confirms the vocabulary-bridging hypothesis

**Success definition:** Any score movement. Even +1.8pp (70%) is signal. 68.2% again means vocabulary bridging alone isn't sufficient.

## SOLID Analysis

- **SRP:** `LLMAgentReformulationStrategy` does one thing — call an LLM API to produce reformulated queries. Prompt construction, HTTP calls, and response parsing are its single responsibility. The RLS loop, retrieval, and fusion are handled by existing code.
- **OCP:** New strategy extends the hierarchy without modifying `ReflectiveLatentSearch` or any existing strategy. Different LLM backends via `TDB_RLS_AGENT_MODEL` env var.
- **LSP:** Substitutable for any other `ReformulationStrategy` — same `reformulate()` contract, returns `list[str]`.
- **ISP:** The `ReformulationStrategy` interface is already minimal (one method).
- **DIP:** `ReflectiveLatentSearch` depends on the `ReformulationStrategy` abstraction, not on `LLMAgentReformulationStrategy` concretely.

## Cost Estimate

- ~150 tokens per DeepSeek call (prompt + response)
- 1,542 LoCoMo items × 1 call each = ~231K tokens (upper bound — only called when confidence check fails)
- DeepSeek-chat: ~$0.07/1M input tokens → well under $1 for the full run
- No retry logic. Individual API failures don't matter across 1,542 items.

## Files

- **Modify:** `python/tardigrade_hooks/rls.py` — add `LLMAgentReformulationStrategy`, register `"agent"` in mode map
- **Modify:** `tests/python/test_rls.py` — 5 ATDD unit tests for the new strategy
- **No new files.** Everything fits in existing modules.

## Scope

### In scope
- `LLMAgentReformulationStrategy` implementation (~60 lines)
- ATDD tests (5 unit tests)
- `agent` mode registration in RLS mode map
- Full LoCoMo benchmark run
- Results documentation

### Out of scope (explicitly deferred)
- **Larger capture model (original Experiment A):** Deprioritized — evidence says latent-space geometry isn't the bottleneck. Preserved as fallback if this experiment yields 0%.
- **MCP agent protocol:** Only worth designing if this experiment shows score movement.
- **New benchmark adapter:** Not needed — existing `TardigradeAdapter` + RLS loop is sufficient.
- **Retry/circuit-breaker for API calls:** Overkill for a hypothesis test.

---

## Appendix: Fallback — Larger Capture Model

If the LLM agent reformulation yields 0%, the next experiment is swapping the capture model from Qwen3-0.6B to Qwen2.5-3B (float16). Zero code changes — just env vars:

```bash
TDB_BENCH_MODEL=Qwen/Qwen2.5-3B TDB_BENCH_DEVICE=mps \
TDB_REFINEMENT_MODE=centered \
PYTHONPATH=python python -m tdb_bench run --mode full --system tardigrade \
  --config python/tdb_bench/config/default.json \
  --output target/bench-3b-capture.json
```

**Risk:** Qwen2.5-3B in float16 on MPS may be slow for 1,542 items × chunked ingestion. Test on a subset first.
