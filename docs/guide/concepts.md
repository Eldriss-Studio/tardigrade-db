# Concepts

## KV Cache Injection

Traditional agent memory systems store text, retrieve text, paste text into the prompt. This consumes prompt tokens.

TardigradeDB stores the model's own KV cache tensors — the internal state the model computes during a forward pass. When using the Python API, retrieved memories are injected directly into the model's attention as KV cache, consuming zero prompt tokens. When using the MCP server, memories are delivered as text for universal LLM compatibility (standard prompt token cost).

**Result (Python API):** Byte-identical output to having the text in the prompt, at 46% fewer prompt tokens.

**When to use:** Single-fact recall queries. "What's the user's preference?" "What did they say about X?"

## Reflective Latent Search (RLS)

Latent-space retrieval works well for specific queries, but vague queries ("What outdoor activities does this person enjoy?") often return weak results — the vocabulary in the query doesn't overlap with the vocabulary in the stored memory.

RLS runs a RETRIEVE → EVALUATE → REFORMULATE → RE-RETRIEVE → FUSE loop to recover from those misses.

```
1. RETRIEVE  — single forward pass, Top5Avg scoring
2. EVALUATE  — confidence = score[rank-1] / score[rank-2]
               if ratio ≥ threshold (1.10): return immediately
3. REFORMULATE — run configured strategies to generate query variants
4. RE-RETRIEVE — score each variant independently
5. FUSE      — Reciprocal Rank Fusion over all result lists
```

**Why confidence-ratio gating?** If rank-1 is clearly better than rank-2, the retrieval is unambiguous and reformulation only adds noise. Gating on ratio lets high-confidence queries bypass the overhead entirely.

**Why Reciprocal Rank Fusion?** Different reformulations may surface different relevant memories. RRF combines ranked lists by position, not by raw score magnitude, so results from diverse query forms can coexist without one dominating.

**Strategy cost ladder** (choose based on latency budget):

| Strategy | Model needed | Latency |
|----------|-------------|---------|
| `KeywordExpansionStrategy` | none | <1ms |
| `MultiPhrasingStrategy` | none | <1ms |
| `EmbeddingExpansionStrategy` | embedding table | ~5ms |
| `GenerativeReformulationStrategy` | local LLM (e.g. Qwen2.5-3B) | ~500ms |
| `LLMAgentReformulationStrategy` | external API (DeepSeek) | ~1-2s (network) |

**Benchmark result:** RLS keyword/embedding strategies produced 0% lift on LoCoMo (2,042 items), confirming that the 68.2% ceiling is the model capability limit, not a retrieval failure. The agent strategy (vocabulary bridging) is the active research path.

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
