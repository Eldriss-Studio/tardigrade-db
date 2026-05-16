**TardigradeDB: Project Summary**

**Overview**

TardigradeDB is a from-scratch, LLM-native memory kernel designed as the persistent memory system for autonomous AI agents. It operates directly on the model's Key-Value (KV) cache tensors in latent space — memory is stored, retrieved, and organized as quantized neural activations, not text.

> **Status: experimental prototype.** APIs, on-disk formats, and guarantees may change. Current results are from controlled experiments and benchmarks.

> **Positioning:** see [`docs/positioning/latency_first.md`](../positioning/latency_first.md) for the measured-latency / footprint / KV-native-API axes TardigradeDB legitimately competes on (sub-millisecond p99 retrieval at 5K cells, 751 B per cell on disk). LoCoMo Judge scoring is a separate in-flight initiative, not the current claim.

**What Works Today**

1. **Persistent Quantized KV Storage:**
   KV cache blocks are persisted to disk as 4-bit quantized tensors in a custom mmap-backed arena. Q4 quantization preserves 89% of injection quality while fitting 4x more context into fixed storage. Append-only segments with per-write fsync. Engine is single-threaded (`&mut self`); multi-agent isolation is via logical owner filtering, not concurrent access.

2. **Latent-Space Retrieval:**
   Per-token Top5Avg scoring computes dot products between individual query and memory hidden-state tokens — 100% recall at 100 memories through the full engine pipeline (Q4 quantization, INT8 scoring, 97ms latency). Semantic Lookaside Buffer (SLB) provides sub-5μs retrieval using INT8 scalar quantization. Three-stage pipeline: SLB → PerTokenRetriever → BruteForceRetriever.

3. **Neuro-Symbolic Organization:**
   Vamana graph index (DiskANN-style) for approximate nearest neighbor search. Trace graph for causal relationships between KV blocks. Decoupled WAL for crash recovery.

4. **Self-Curating Governance (Adaptive Knowledge Lifecycle):**
   Every stored block carries an importance score (ι ∈ [0,100]) that increases on access (+3) and update (+5), with 0.995 daily decay. Memories autonomously promote/demote across maturity tiers (draft → validated → core) with hysteresis thresholds. Tier-based retrieval boost (Core 1.25×, Validated 1.1×).

5. **Verified KV Injection:**
   Fully synthetic gibberish facts (nonsense proper nouns, fake units) injected via KV tensors and recalled by Qwen3-0.6B at 9/10, matching text RAG at 100% recall ratio. Any correct recall is unambiguous — these strings cannot come from model weights.

**Deployment Paths**

- **Path 1 (HuggingFace direct injection):** `KnowledgePackStore` passes `past_key_values` directly to `model.generate()`. This is the path with the strongest results (9/10 synthetic fact recall). Not integrated into production serving frameworks.
- **Path 2 (vLLM prefix adapter):** `MemoryPrefixBuilder` assembles governed memory text prefixes. vLLM's stock prefix-cache serves them automatically at zero prefill cost. This is a text adapter, not KV injection.
- **vLLM KV Connector:** Implements vLLM's KV Connector v1 API as a persistent prefix-cache accelerator. The v1 API is prefix-cache only — it cannot do cross-prompt KV injection.

**Architectural Vision (Not Yet Implemented)**

These are design targets from the technical design document, not current capabilities:

- **GPU DMA:** Direct NVMe→GPU transfers via cuFile/GDS, bypassing CPU.
- **Decoupled position encoding:** RoPE remapping for safe historical KV block reuse across contexts.
- **BatchQuantizedKVCache:** Concurrent Q4 inference with interleaved prefill/decode scheduling.
- **Cross-agent KV reuse (RelayCaching):** Sharing KV cache across agents for estimated TTFT reduction.

**Value Proposition**

TardigradeDB replaces text retrieval with direct cache restoration. Instead of tokenize → embed → search → detokenize → re-tokenize, it stores the model's own internal state and restores it directly into the attention stack. The search IS attention — relevance is computed in the same mathematical space the model uses to think.

No existing project unifies persistent quantized KV storage, latent-space retrieval, neuro-symbolic organization, and adaptive lifecycle into a single kernel. The closest overlaps are partial: LMCache (persistent KV, but prefix-cache only), MemArt (latent retrieval, but not a full engine), ByteRover 2.0 (adaptive lifecycle, but text-based).
