# TardigradeDB Learning Roadmap

A structured guide for understanding the LLM-native concepts behind TardigradeDB — from transformer internals to the full Aeon Architecture. Each level builds on the previous and unlocks the ability to contribute to different parts of the codebase.

> **Core mental model:** TardigradeDB is not a database that happens to store tensors — it's an extension of the model's own memory system that happens to persist to disk. Every design decision follows from that inversion.

---

## Level 0 — How Transformers Actually Work (The Internals, Not the API)

Before anything else, you need to understand what TardigradeDB is actually *storing*. Not "transformers predict the next token" — the mechanical details of what happens inside.

### What to Learn

- **Self-attention mechanics** — Q, K, V projections. How a query vector dot-products against key vectors to produce attention weights, which then weight the value vectors. This is the single most important concept — TardigradeDB's entire retrieval layer is "do attention against stored K/V tensors instead of current-context tokens."

- **The KV cache** — During autoregressive generation, the model caches K and V tensors from all previous tokens so it doesn't recompute them. Understand *what* is cached (per-layer K and V matrices), *why* (avoid O(n) recomputation), and *how big it gets* (scales with sequence length × layers × head dim).

- **Multi-head attention** — Each head has its own Q/K/V projections with different learned subspaces. TardigradeDB stores per-head K/V, so you need to know what "head dimension" (`d_k`, `d_v`) means concretely.

- **Position encoding** — How the model knows token order. Especially **RoPE (Rotary Position Embeddings)** — understand why naively injecting old KV blocks breaks position information, and why "decoupled position encoding" is necessary to fix it (this is central to the Storage layer).

### Resources

- Jay Alammar's "The Illustrated Transformer"
- Andrej Karpathy's "Let's build GPT from scratch" (YouTube)
- The RoPE paper (Su et al., "RoFormer") — focus on how rotation is applied to Q and K

### You're Ready When

You can explain why restoring a KV cache from a previous session requires re-encoding position IDs, and why you can't just concatenate old K/V with new K/V naively.

---

## Level 1 — KV Cache Management & Quantization

This is where you go from "I understand the theory" to "I understand what TardigradeDB physically stores on disk."

### What to Learn

- **KV cache memory math** — For a model with L layers, H heads, dimension d, sequence length S: cache size = `2 × L × H × d × S × bytes_per_element`. Do the math for Llama-70B at 8K context in FP16 vs Q4. This is why quantization matters.

- **Quantization fundamentals** — What FP16, INT8, and INT4 (Q4) mean. How you map a range of floats into a small set of integers (scale + zero-point). The `half` crate in the workspace handles FP16; `tdb-storage/quantization.rs` handles Q4.

- **Safetensors format** — The serialization format TardigradeDB uses. Understand its header structure (JSON metadata + flat tensor data). It's designed for zero-copy mmap, which connects directly to the GPU DMA path.

- **The cost of quantization error** — Q4 KV is lossy. Understand *where* the error shows up (attention score distortion) and why it's acceptable for memory (the "gist" is preserved even if exact values shift).

### Resources

- Hugging Face's safetensors specification
- "GPTQ" and "AWQ" papers for intuition on weight quantization (same principles apply to KV)
- The MemArt paper — specifically how they handle compressed keys

### You're Ready When

You can calculate how many agent contexts fit in 24GB VRAM at Q4 vs FP16, and explain why `tdb-core/types.rs` defines tensor shapes the way it does.

### Codebase Connection

| File | What It Teaches |
|------|-----------------|
| `crates/tdb-core/src/types.rs` | Fundamental tensor shape definitions |
| `crates/tdb-storage/src/quantization.rs` | Q4 quantization implementation |

---

## Level 2 — Latent-Space Retrieval (The MemArt Idea)

This is TardigradeDB's core differentiator versus every text-based memory system.

### What to Learn

- **Why Flat RAG is wasteful** — Traditional RAG: text → tokenize → embed → cosine search → retrieve text → tokenize again → inject into prompt → model recomputes K/V from scratch. TardigradeDB: retrieve pre-computed K/V → inject directly into attention. No round-trip through text.

- **Multi-Token Aggregation** — Instead of embedding a whole passage into one vector (like sentence-transformers do), MemArt keeps the *individual* K vectors from each token. Retrieval computes attention scores between the current query's Q vectors and stored K vectors directly. This is why the 91–135x prefill reduction is possible — you skip the entire "embed and compare" pipeline.

- **The Semantic Lookaside Buffer (SLB)** — A CPU-side cache that exploits conversational locality (recent topics tend to recur). Stored as INT8-quantized key summaries for fast approximate matching. When it hits, retrieval is sub-5μs because you skip the GPU entirely.

- **RelayCaching** — In multi-agent scenarios, Agent B can reuse Agent A's decode-phase KV cache during its own prefill. Only the *divergent* prefix positions get recomputed. Understand why this cuts TTFT by 4.7x.

### Resources

- The MemArt / CacheBlend paper lineage — search for "KV cache reuse" and "prefix caching" in the context of vLLM/SGLang
- LMCache project — infrastructure-level KV sharing, same idea, different scope
- The RelayCaching paper

### You're Ready When

You can explain why TardigradeDB doesn't need an embedding model at all, and trace the data flow from "agent asks a question" → "relevant KV blocks are identified" → "blocks are injected into the attention stack."

### Codebase Connection

| File | What It Teaches |
|------|-----------------|
| `crates/tdb-retrieval/src/attention.rs` | Latent-space attention scoring |
| `crates/tdb-retrieval/src/slb.rs` | Semantic Lookaside Buffer |
| `crates/tdb-retrieval/src/int8_quant.rs` | INT8 quantization for SLB |
| `crates/tdb-retrieval/src/simd_distance.rs` | SIMD-accelerated distance computation |

---

## Level 3 — Systems-Level: How the Storage Engine Works

Now you bridge the LLM knowledge into the Rust codebase.

### What to Learn

- **Memory-mapped I/O (mmap)** — How `memmap2` maps files directly into virtual address space. Why this enables zero-copy reads and connects to GPU DMA. The blob arena and segment manager use this heavily.

- **GPU Direct Memory Access** — NVMe → GPU without CPU bounce buffers. Understand the `cuda/` directory's purpose — this is where the DMA kernel will live.

- **Block pool design** — A pool of quantized KV blocks, similar to a slab allocator but for tensor-shaped data. Each block represents a captured KV snapshot from a specific inference step.

- **Lock-free concurrency** — `crossbeam-epoch` enables epoch-based reclamation (like a garbage collector for lock-free data structures). This is how the index achieves 750ns P99 under 16-thread contention.

### Resources

- The `memmap2` crate documentation
- "Lock-Free Data Structures Using STMs in Haskell" (Crossbeam's epoch-based reclamation is the Rust equivalent)
- Any GPU DMA / GPUDirect Storage overview from NVIDIA

### You're Ready When

You can trace how a KV block goes from "captured during inference" → serialized to Q4 → written to a segment file → indexed → later retrieved via mmap → DMA'd to GPU.

### Codebase Connection

| File | What It Teaches |
|------|-----------------|
| `crates/tdb-storage/src/block_pool.rs` | Quantized KV block management |
| `crates/tdb-storage/src/arena.rs` | Append-only mmap-backed blob arena |
| `crates/tdb-storage/src/segment.rs` | Segment file management |

---

## Level 4 — Organization: The Neuro-Symbolic Graph

### What to Learn

- **Why vector search alone fails ("Vector Haze")** — Cosine similarity retrieves *topically similar* but *causally disconnected* facts. An agent needs to know that "the patient reported chest pain" CAUSED "the doctor ordered an ECG" — pure similarity search can't express this.

- **The Atlas Index** — A Page-Clustered Vector Index combining small-world graph navigation (like HNSW) with B+ Tree-style disk locality. The Vamana algorithm (from Microsoft's DiskANN) powers the graph navigation.

- **The Trace graph** — An episodic graph that tracks causal edges between KV blocks. Not a general knowledge graph; specifically tracks "this memory led to that memory."

- **Write-Ahead Log** — Crash recovery for the graph with <1% overhead. Standard database technique, applied to a tensor-native context.

### Resources

- DiskANN / Vamana paper — for the graph index algorithm
- Any intro to Write-Ahead Logging (the concept transfers directly from traditional databases)
- Genesys-Memory paper — a text-based causal graph for comparison; TardigradeDB does the same over KV blocks

### You're Ready When

You can explain why TardigradeDB needs *both* a spatial index (Atlas) and a causal graph (Trace), and what queries each one answers.

### Codebase Connection

| File | What It Teaches |
|------|-----------------|
| `crates/tdb-index/src/vamana/mod.rs` | Vamana small-world graph index |
| `crates/tdb-index/src/trace.rs` | Causal episodic graph |
| `crates/tdb-index/src/wal.rs` | Write-Ahead Log for crash recovery |

---

## Level 5 — Governance: The Adaptive Knowledge Lifecycle (AKL)

### What to Learn

- **Importance scoring** — The formula: `ι ∈ [0, 100]`, +3 on access, +5 on update, ×0.995 daily decay. Simple but effective — frequently-used memories survive, neglected ones fade.

- **Maturity tiers with hysteresis** — Draft → Validated (at ι ≥ 65, demote < 35), Validated → Core (at ι ≥ 85, demote < 60). The *gap* between promote and demote thresholds prevents oscillation — this is a control theory concept (Schmitt trigger / hysteresis).

- **Recency decay** — `r = exp(-Δt / τ)`, τ = 30 days. This gives a ~21-day half-life. Understand why exponential decay is chosen over linear: it never reaches zero, degrades gracefully, and is computationally cheap.

- **ByteRover 2.0 comparison** — ByteRover's AKL operates on text files and a human-readable Context Tree. TardigradeDB's AKL operates on tensor blocks. Same algorithm, fundamentally different substrate — that's the key insight.

### Resources

- The ByteRover 2.0 analysis in `docs/refs/AI Agentic Memory System Efficiency.md` — describes the AKL algorithm origin
- Any control theory intro covering hysteresis / Schmitt triggers
- Exponential decay in the context of LRU cache eviction policies

### You're Ready When

You can explain why a memory block with ι = 64 stays as "draft" even though 64 is close to the 65 promotion threshold, and what would happen without hysteresis gaps.

### Codebase Connection

| File | What It Teaches |
|------|-----------------|
| `crates/tdb-governance/src/scoring.rs` | Importance score computation |
| `crates/tdb-governance/src/tiers.rs` | Maturity tier promotion/demotion with hysteresis |
| `crates/tdb-governance/src/decay.rs` | Temporal recency decay |

---

## Level 6 — The Synaptic Bank & Adapter Concept

This is the most forward-looking part of the design — where episodic memory (KV cache) meets procedural memory (weights).

### What to Learn

- **LoRA (Low-Rank Adaptation)** — How you modify a model's behavior by adding small rank-r matrices (A and B) to existing weight matrices: `W' = W + αBA`. The full model stays frozen; only A and B are trained/stored per agent.

- **Why this is "weight-level memory"** — KV cache stores *episodic* memory (specific past interactions). Adapters store *procedural/semantic* memory (learned patterns and preferences). Together they mirror the hippocampus (episodic) + neocortex (consolidated) split from neuroscience.

- **Hot-swapping adapters** — At serving time, loading a different agent's adapter bank is just swapping a few small matrices. LRAgent does something similar with shared base KV + per-agent LoRA caches.

- **Online learning** — No full backprop; cheap updates. After a successful completion that should become "habit," compute a small gradient for just the adapters, clip, quantize, and persist. Over time, each agent develops personalized weights.

### Resources

- The LoRA paper (Hu et al., 2021)
- LRAgent paper — shared base KV + low-rank per-agent caches
- MemoryLLM (Apple) — treats FFN parameters as token-indexed memory
- PRIME — dual-memory personalization framework (episodic + semantic)

### You're Ready When

You can explain the difference between "remembering what happened" (KV bank) and "learning how to behave" (synaptic bank), and why TardigradeDB needs both.

### Codebase Connection

| File | What It Teaches |
|------|-----------------|
| `crates/tdb-core/src/synaptic_bank.rs` | Per-agent adapter storage |
| `crates/tdb-core/src/memory_cell.rs` | The MemoryCell struct (KV counterpart) |

---

## Level 7 — Concurrency & Scheduling

### What to Learn

- **BatchQuantizedKVCache** — How to handle multiple agents doing Q4 inference concurrently without contention on shared memory.

- **Interleaved prefill/decode scheduling** — One agent decodes (fast, sequential) while another prefills (slow, parallel). The scheduler interleaves them to hide the 500ms warm-reload latency behind productive work. Same concept as GPU kernel interleaving in CUDA streams.

### Resources

- vLLM's continuous batching paper (Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention")
- Orca paper (Yu et al.) — iteration-level scheduling for LLM inference
- CUDA Streams documentation — for the general concept of overlapping compute and memory operations

### You're Ready When

You can explain why a naive "one agent at a time" approach wastes GPU cycles, and how interleaving prefill and decode phases keeps the pipeline saturated.

### Codebase Connection

| File | What It Teaches |
|------|-----------------|
| `crates/tdb-engine/src/batch_cache.rs` | Concurrent Q4 batch cache |
| `crates/tdb-engine/src/scheduler.rs` | Interleaved prefill/decode scheduler |
| `crates/tdb-engine/src/engine.rs` | Top-level orchestration |

---

## Suggested Codebase Reading Order

Once you've built up the conceptual foundation, read the source files in this order — each builds on the previous:

| # | File | Concepts Required |
|---|------|-------------------|
| 1 | `crates/tdb-core/src/types.rs` | Level 0–1 |
| 2 | `crates/tdb-core/src/memory_cell.rs` | Level 0–1 |
| 3 | `crates/tdb-core/src/error.rs` | — |
| 4 | `crates/tdb-storage/src/quantization.rs` | Level 1 |
| 5 | `crates/tdb-storage/src/block_pool.rs` | Level 1, 3 |
| 6 | `crates/tdb-storage/src/arena.rs` | Level 3 |
| 7 | `crates/tdb-storage/src/segment.rs` | Level 3 |
| 8 | `crates/tdb-retrieval/src/attention.rs` | Level 2 |
| 9 | `crates/tdb-retrieval/src/int8_quant.rs` | Level 1–2 |
| 10 | `crates/tdb-retrieval/src/slb.rs` | Level 2 |
| 11 | `crates/tdb-retrieval/src/simd_distance.rs` | Level 2–3 |
| 12 | `crates/tdb-index/src/vamana/mod.rs` | Level 4 |
| 13 | `crates/tdb-index/src/trace.rs` | Level 4 |
| 14 | `crates/tdb-index/src/wal.rs` | Level 4 |
| 15 | `crates/tdb-governance/src/scoring.rs` | Level 5 |
| 16 | `crates/tdb-governance/src/decay.rs` | Level 5 |
| 17 | `crates/tdb-governance/src/tiers.rs` | Level 5 |
| 18 | `crates/tdb-core/src/synaptic_bank.rs` | Level 6 |
| 19 | `crates/tdb-engine/src/batch_cache.rs` | Level 7 |
| 20 | `crates/tdb-engine/src/scheduler.rs` | Level 7 |
| 21 | `crates/tdb-engine/src/engine.rs` | All levels |

---

## Competitive Context

Understanding what exists helps clarify what TardigradeDB is *not*:

| Project | What It Does | How TardigradeDB Differs |
|---------|-------------|--------------------------|
| **Mem0, Letta, Zep** | Text-based agent memory with vector/graph search | TardigradeDB operates on KV tensors, not text |
| **LMCache, LRAgent** | Infrastructure-level KV cache sharing | Plumbing, not a cognitive memory engine |
| **MemArt** | Latent-space retrieval via compressed keys | Retrieval only, not a full storage/governance system |
| **ByteRover 2.0** | AKL governance over text-file Context Trees | Same governance algorithm, but over text, not tensors |
| **Genesys-Memory** | Causal graph with autonomous pruning | Text-based graph, not KV-native |
| **MemoryLLM (Apple)** | FFN parameters as interpretable memory | Research on weights-as-memory, not productized |
| **SpaceTimeDB** | WASM reducers inside the database kernel | Relevant execution model, not LLM-specific |

No existing project unifies: persistent quantized KV storage + latent-space retrieval + neuro-symbolic organization + adaptive lifecycle + per-agent adapter banks into a single kernel. That's the gap TardigradeDB fills.

---

## Key Papers & References

| Topic | Paper / Resource |
|-------|-----------------|
| Transformer internals | "Attention Is All You Need" (Vaswani et al., 2017) |
| Rotary position encoding | "RoFormer" (Su et al., 2021) |
| KV cache quantization | GPTQ, AWQ papers |
| Latent-space retrieval | MemArt / CacheBlend lineage |
| Paged KV management | PagedAttention / vLLM (Kwon et al., 2023) |
| Graph indexing | DiskANN / Vamana (Subramanya et al., 2019) |
| Low-rank adaptation | LoRA (Hu et al., 2021) |
| Weights as memory | MemoryLLM (Apple), PRIME |
| Agent memory survey | "Architecture of Continuous Agentic Memory Systems" (2026) |
| Iteration-level scheduling | Orca (Yu et al., 2022) |
| Prefix caching | LMCache, RelayCaching |
