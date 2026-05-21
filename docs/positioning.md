# Positioning — Why TardigradeDB?

Current agent memory systems (Mem0, Letta, Zep) rely on text retrieval — tokenize, embed, search, detokenize. This creates a lossy round-trip through representations the model never asked for. TardigradeDB eliminates that by persisting the model's own internal state and restoring it directly into the attention stack.

This document explains what we chose differently from embedding RAG, where the choice pays off, and where RAG remains stronger. For measured numbers on the latency / footprint / KV-native dimensions, see [`docs/positioning/latency_first.md`](positioning/latency_first.md).

## How retrieval differs from semantic search

Semantic search (vector DBs, RAG) and TardigradeDB both find "relevant stuff" from stored data. The difference is in what is being compared:

```
Semantic search (Mem0, Letta, Pinecone):
  text → separate embedding model → vector → store
  query → same embedding model → vector → cosine similarity
  The embedding model is a translator between text and numbers.
  The LLM that is actually thinking never touches this process.
  Retrieved text → re-tokenize → paste into prompt → consume context window

TardigradeDB:
  LLM is already thinking → hidden state tensors exist as a byproduct → store
  LLM thinks about query → hidden state tensors → per-token Top5Avg scoring
  Retrieved KV cache → inject directly into attention → zero prompt tokens consumed
  No translator. No separate model. The comparison happens in the same
  mathematical space the model uses to think. The search IS attention.
```

Semantic search outsources retrieval to a separate system. TardigradeDB does retrieval inside the model's own representation space — the model searches its own memories using its internal activations. When a memory is found, it's injected as pre-computed KV tensors directly into the attention cache, consuming zero prompt tokens.

## What TardigradeDB is NOT trying to be

TardigradeDB is **not** a faster, higher-recall RAG. It is an architecturally different choice with its own trade-offs. Comparing the two head-to-head on "% recall on a benchmark" misses the point.

### Where TardigradeDB chooses differently

| Concern | Embedding RAG | TardigradeDB |
|---------|---------------|--------------|
| Representation cost | Pay an embedding-model forward per write AND per query | Reuse the LLM's own internal state — no separate model |
| Prompt-token budget | Retrieved text consumes prompt tokens | Retrieved KV is injected into attention — zero prompt tokens |
| Round-trip fidelity | text → embed → search → text → re-tokenize | Native tensor path, no encode/decode hops |
| Index target | Document chunks (text-shaped) | Activation cells (model-shaped) |

### Where RAG remains stronger today (honest)

- **Vague-query recall on small corpora.** Embedding models are explicitly trained for paraphrase robustness. TardigradeDB's latent-space scoring inherits the LLM's own representational geometry, which is not optimized for that task.
- **Multi-hop retrieval.** Engineered RAG pipelines with re-ranking and hybrid search outperform the current Trace + Top5Avg pipeline.
- **Mature ecosystem.** Vector DBs, eval harnesses, and tooling are battle-tested; TardigradeDB is a research kernel.

### Where the choice pays off

- **Cross-session memory persistence inside the model's own state.** The ability to save and re-inject pre-computed KV is fundamentally not something a vector DB can do — it would require re-running the model.
- **Zero prompt-token cost on injection.** Long-running agents with large memory trails don't pay context-window tax for prior knowledge.
- **One representation surface.** No separate embedding model to fine-tune, drift-monitor, or version-pin against the inference model.

If your problem is "find the right document chunk and paste it into a prompt", use RAG. If your problem is "make an LLM behave as if it had already lived through prior conversations", that's the question TardigradeDB is investigating.

## Three-way comparison

| Dimension | Embedding RAG / vector memory | Traditional KV cache | TardigradeDB |
|-----------|-------------------------------|----------------------|--------------|
| Primary stored unit | Text chunks + embedding vectors | Attention K/V tensors for active context window | Quantized K/V tensors as durable memory cells |
| Retrieval signal | ANN / cosine similarity in embedding space | Usually none (append + replay only) | Attention-native latent scoring (`q · k / √d_k`) |
| Semantic recall | Strong for text-level similarity | Not a retrieval system by itself | Designed for latent semantic recall directly on K/V |
| Persistence scope | External DB persistence is common | Usually process / session-local and ephemeral | Built for cross-session persistence in the engine |
| Context usage pattern | Retrieve text, then re-tokenize into prompt | Replay prior cache pages, often all-or-nothing | Retrieve and inject selected memory slices |
| Lifecycle / governance | App-defined policies, often ad hoc | None by default | AKL promotion / demotion / decay |
| Causal / episodic structure | Optional metadata graph | None by default | Trace graph + WAL recovery model |
| Representation round-trip | text → embed → search → text | none, but no semantic retrieval layer | native tensor path (no mandatory text / embedding round-trip) |
| Status | Mature ecosystem / pattern | Core inference primitive | Research-grade preview under active validation |

## "Isn't this just a KV cache?"

Yes at the data level; no at the system level.

A raw KV cache is append-and-replay state for one running model session. Compared with raw KV cache and text / embedding memory stacks, TardigradeDB demonstrates:

- **Attention-native semantic retrieval** (`q · k / √d_k`), so recall is based on latent similarity rather than text keyword overlap.
- **Selective memory injection**, fetching only relevant KV slices instead of replaying full history.
- **Durable compressed persistence** (Q4 / Q8) across sessions / runs, not process-local ephemeral cache pages.
- **Memory lifecycle control** (AKL promotion, demotion, decay), avoiding unmanaged growth.
- **Causal / episodic structure + recovery model** (Trace + WAL + rebuildable derived state).
- **Cross-agent memory boundary** via a shared engine API, instead of one cache per process.
- **Native tensor path**, with no mandatory text → embedding → ANN → text round-trip.

In short: TardigradeDB stores KV tensors, but is aiming to behave like a managed long-term memory kernel.

## Naming trivia

The name *TardigradeDB* is a metaphor; these are the development pillars behind it:

- **Cryptobiosis → dormant memory revival**: quantized KV state can be persisted, then "reanimated" by retrieval and reinjection later.
- **Resilience under stress → recovery-first design**: WAL + rebuildable derived state + fail-fast replay boundaries.
- **Tiny footprint → compressed survival**: Q4 / Q8 compression keeps memory practical under constrained capacity.
- **Adaptive survival → memory lifecycle control**: AKL promotion / demotion / decay keeps useful memory active and stale memory fading.

## See also

- [`docs/positioning/latency_first.md`](positioning/latency_first.md) — measured numbers (sub-millisecond p99 retrieval, 751 B per cell).
- [`docs/architecture.md`](architecture.md) — the four-layer Aeon architecture.
- [`docs/research-log.md`](research-log.md) — historical experiments and benchmark progression.
- [`docs/competitors/`](competitors/) — direct competitor analysis (MemArt, ByteRover, Letta, LMCache, …).
