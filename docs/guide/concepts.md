# Concepts

## KV Cache Injection

Traditional agent memory systems store text, retrieve text, paste text into the prompt. This consumes prompt tokens.

TardigradeDB stores the model's own KV cache tensors — the internal state the model computes during a forward pass. When a memory is retrieved, its KV cache is injected directly into the model's attention, consuming zero prompt tokens.

**Result:** Byte-identical output to having the text in the prompt, at 46% fewer prompt tokens.

**When to use:** Single-fact recall queries. "What's the user's preference?" "What did they say about X?"

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
| **Injection** | Reconstruct DynamicCache, inject into model.generate() | 8/10, byte-identical to text RAG |
| **Trace** | Follow graph edges to discover related memories | 70% on cross-referencing at 140 memories |

## When to Use Text RAG Instead

TardigradeDB's KV injection works best for single-fact recall. For queries that need the model to reason across multiple facts simultaneously, text delivery (paste facts into the prompt) is more reliable. The agent can use TardigradeDB's retrieval to find the right memories, then choose the delivery method based on query complexity.
