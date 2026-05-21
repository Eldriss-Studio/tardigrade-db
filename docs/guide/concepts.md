# Concepts

This page is the vocabulary you need to read the rest of the documentation, and the design rationale behind the most important pieces. It assumes you've already skimmed the [README](../../README.md) — you know roughly what TardigradeDB is — and you want the conceptual picture before going deeper into a specific guide.

## KV Cache Injection

A language model's *KV cache* is the running internal state it builds up while processing tokens. Every token the model reads produces a key and value vector at each attention layer; the KV cache is just the stack of all those vectors so far. The model uses the cache to decide what to generate next, and once generation ends the cache is normally thrown away.

Traditional agent memory systems work around this loss by storing the *text* of past conversations, retrieving the relevant text on each new turn, and pasting it back into the prompt — which means the model re-tokenizes and reprocesses the same facts every time, paying prompt tokens for every recall. TardigradeDB stores the KV cache itself instead. When the Python API retrieves a memory, it reinjects the stored cache directly into the model's attention, contributing zero additional prompt tokens because the cache is restored rather than re-tokenized. (The query text on the new turn still costs tokens; only the recalled facts come for free.) When the MCP server retrieves a memory, it delivers the text back in the tool response because MCP can't carry tensors — that path pays the normal prompt-token cost in exchange for universal LLM compatibility.

On the Python path, the recall produces byte-identical generation to having the fact in the prompt, while consuming roughly 46% fewer total prompt tokens than the equivalent text-RAG call. This is most useful for single-fact recall queries — *"what's the user's preference?"*, *"what did they say about X?"* — rather than for queries that require reasoning across many facts simultaneously, where text delivery is still more reliable.

## Reflective Latent Search (RLS)

> ⚠️ **Honest status (2026-05-14):** RLS is documented because the API exists and the design has independent value as a teaching example, but **it does not currently improve retrieval over the no-RLS baseline on clean LoCoMo data**. The 2026-05-14 bench audit found that every RLS mode (keyword, multi-phrasing, embedding, generative, LLM-agent) underperforms the no-RLS baseline; the DeepSeek agent reformulator specifically loses 12.7 percentage points. The earlier 68.2% LoCoMo ceiling against which RLS was being measured was also retracted in the same audit (the runs used the lexical fallback adapter on a corrupted dataset, not the native KV engine). Honest native-engine baseline on clean data: ~36% R@1 at 50-item scale. See [`docs/experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md) for the full record. Read what follows knowing the technique didn't deliver; the design is preserved for future work and because the API surface is still in the codebase.

Latent-space retrieval works well for specific queries but tends to weaken on vague queries — *"what outdoor activities does this person enjoy?"* — where the query's vocabulary doesn't overlap with the stored memory's vocabulary. RLS was the attempt to recover from those misses.

The loop is **retrieve, evaluate, reformulate, re-retrieve, fuse**. The engine does a single forward pass and scores stored memories with per-token Top5Avg; that's the retrieve step. It then evaluates how confident that retrieval is by computing the ratio of the top score to the second-best score — if rank 1 is much stronger than rank 2 (ratio ≥ 1.10), the retrieval is unambiguous and the loop returns immediately. Otherwise the query is ambiguous, and RLS reformulates: it asks one or more configured strategies to produce query variants, scores each variant independently against the same memory store, and finally merges all the ranked result lists using Reciprocal Rank Fusion (RRF) — a rank-based merge that combines results by their positions in each list rather than by raw score magnitude, so a memory that appears at rank 2 across three different query variants outranks a memory that appears at rank 1 only once.

The gating on the confidence ratio is what keeps the loop cheap on average: high-confidence queries bypass the reformulation work entirely. The choice of RRF rather than score-based fusion is what lets diverse query forms coexist without one variant's score magnitudes dominating the merge.

**Strategy cost ladder** (choose by latency budget):

| Strategy | Model needed | Latency |
|----------|-------------|---------|
| `KeywordExpansionStrategy` | none | <1 ms |
| `MultiPhrasingStrategy` | none | <1 ms |
| `EmbeddingExpansionStrategy` | embedding table | ~5 ms |
| `GenerativeReformulationStrategy` | local LLM (e.g. Qwen2.5-3B) | ~500 ms |
| `LLMAgentReformulationStrategy` | external API (DeepSeek) | ~1–2 s (network) |

## Multi-view Consolidation

A single fact can be queried in many forms: "Tomoko teaches swimming" vs "Who teaches swimming?" vs "Nishida's instruction". Latent-space retrieval matches well when the query resembles the stored text — but paraphrases can miss.

Multi-view consolidation creates multiple retrieval surfaces for the same canonical memory. Each view is a text reframing stored as a linked pack:

```
Canonical pack (KV tensor)
  └── Supports edge → Summary view:   "Tomoko Nishida teaches swimming"
  └── Supports edge → Question view:  "What did Tomoko Nishida do?"
  └── Supports edge → Paraphrase view: "Nishida Tomoko relocated to the center"
```

The canonical pack holds the actual KV tensor used for injection. The views are lightweight retrieval surfaces that point back to the same fact. Querying from any framing can now surface the memory.

**Three rule-based framings** (no model required):
- `summary` — extractive first clause
- `question` — WH-question with extracted subject
- `paraphrase` — clause reordering + synonym substitution

**LLM framing** (`llm_question`) — uses a local model to generate diverse questions via the HyPE pattern. More creative, but requires a model.

**Idempotency:** `MemoryConsolidator` tracks views already attached and skips packs that have already been consolidated. Safe to call repeatedly without generating duplicate views.

**Governance gating:** Only packs at Validated tier or above are consolidated (configurable via `min_tier`). Draft memories are ephemeral candidates, not worth the consolidation overhead.

## Trace Links

Memories can be connected via trace links — durable graph edges stored in the engine. When the agent retrieves one memory, it can follow trace links to discover related memories.

**Example:**
```
Memory A: "Went to a bookstore in Pilsen"
    └── trace link
Memory B: "The bookstore is called Casa Azul"
```

A query about "the bookstore" finds Memory A, then follows the trace link to discover Memory B.

**Who creates links?** The agent decides. TardigradeDB provides `store_and_link()` and `store_linked()` — the agent calls them when it knows two facts are related.

**Why not auto-link?** Latent similarity can't distinguish "same event" from "same topic." Two cooking memories score as similar as a fact and its related detail. Auto-linking with latent similarity was tested and doesn't work (see `docs/experiments/multi-memory-injection.md`).

## Trace-Boosted Retrieval

When retrieving memories, TardigradeDB boosts scores for memories that have trace links. Connected memories are "discovery hubs" — finding one leads to related facts.

A memory with 3 trace links gets a higher effective score than an isolated memory with slightly higher content similarity. This helps the retriever prefer connected memories over isolated ones.

## Governance (AKL)

The Adaptive Knowledge Lifecycle automatically manages memory importance:

- **Importance (0-100):** +3 on read, +5 on write, decays daily (factor 0.995)
- **Tiers:** Draft → Validated (importance ≥ 65) → Core (≥ 85). Hysteresis prevents oscillation.
- **Recency decay:** `exp(-days/30)` — memories unused for a month have their retrieval score halved

Governance runs automatically. The agent doesn't need to manage memory lifecycle.

## KV Packs

A KV Pack is the unit of storage. It contains the complete KV cache from one model forward pass — all layers, all heads. For Qwen3-0.6B, that's 28 layers of K+V tensors.

Packs are:
- **Stored atomically** (single fsync for all layers)
- **Retrieved as units** (all layers come back together)
- **Governed together** (one importance score per pack)
- **Linkable** (trace edges connect packs to each other)

## Storage vs Retrieval vs Injection

| Layer | What it does | Key metric |
|-------|-------------|------------|
| **Storage** | Q4 quantize KV tensors, persist to disk | 730 KB per memory |
| **Retrieval** | Per-token Top5Avg scoring in latent space | 100% recall at 100 memories |
| **RLS** | Confidence-gated reformulation + RRF fusion | Closes vocab-gap misses |
| **Injection** | Reconstruct DynamicCache, inject into model.generate() | 8/10, byte-identical to text RAG |
| **Trace** | Follow graph edges to discover related memories | 70% on cross-referencing at 140 memories |

## When to Use Text RAG Instead

TardigradeDB's KV injection works best for single-fact recall. For queries that need the model to reason across multiple facts simultaneously, text delivery (paste facts into the prompt) is more reliable. The agent can use TardigradeDB's retrieval to find the right memories, then choose the delivery method based on query complexity.

## MCP vs Python API

| Path | Delivery | Token cost | Requires |
|------|----------|------------|----------|
| MCP server | Text in tool response | Normal prompt tokens | Any LLM client |
| Python API (`TardigradeClient`) | KV cache injection | Zero prompt tokens | Model access (HuggingFace) |

Both paths use TardigradeDB's latent-space retrieval to find the right memories. The difference is delivery: the MCP server returns memory text in its tool response (which the LLM reads as prompt tokens), while the Python API injects KV cache directly into the model's attention (bypassing the prompt entirely).

Choose MCP for convenience with any model. Choose the Python API when you control the model and want zero-token injection.
