### Technical Design Document: Latent Space Agent-Native Database Kernel (Aeon Architecture)

**1. Architecture Overview**

TardigradeDB operates as a persistent memory kernel for autonomous AI agents. It persists and retrieves the LLM's Key-Value (KV) cache tensors in quantized form, using latent-space attention scoring for retrieval. The system is organized into four layers — Storage, Retrieval, Organization, and Governance — coordinated by an Engine facade that provides crash-recoverable state management via the Memento pattern.

**2. Storage Layer: Persistent Quantized KV-Cache**

The foundation is a persistent block pool that stores each agent's KV cache durably across restarts.

- **Format & Quantization:** KV caches are stored in a custom binary segment format using 4-bit group-wise quantization (Q4_0, same as llama.cpp). 4× compression vs FP32 with MSE < 0.01. Position encodings stored unquantized for exact reproduction.
- **Append-Only Segments:** 256MB segments with length-prefixed binary records. `sync_data()` on every write. Partial records at EOF are silently discarded on recovery.
- **Text Store:** Durable append-only binary sidecar for fact text associated with KV packs. Single source of truth (replaces the original JSON sidecar).
- **Deletion Log:** Append-only log of deleted pack IDs, applied during `refresh()`.

**3. Retrieval Engine: Latent Space Attention & Caching**

Retrieval computes relevance directly in the model's latent representation space, without external embedding models.

- **Per-Token Top5Avg Scoring:** Hidden-state vectors stored per-token. Retrieval scores by averaging the top 5 dot products across all query-memory token pairs. Achieves 100% recall at 100 memories through the full pipeline (Q4, INT8, per-token encoding). Validated on Qwen3-0.6B.
- **Semantic Lookaside Buffer (SLB):** Fixed-capacity LRU cache with symmetric INT8 quantized keys. Sub-5μs retrieval. NEON SDOT intrinsics on ARM, auto-vectorized fallback on x86.
- **Three-Stage Pipeline:** SLB (hot) → PerTokenRetriever (primary) → BruteForce (fallback). Chain of Responsibility with CellId deduplication.
- **Active Governance:** Retrieval scores are adjusted by recency decay × tier boost (Core 1.25×, Validated 1.1×). Results re-sorted after adjustment.

**4. Organization Layer: Neuro-Symbolic Topology**

Memory is structured via a dual topology for both similarity search and causal reasoning.

- **Vamana Graph Index:** DiskANN-style single-layer graph with robust pruning. Lazy activation at configurable cell count threshold. Supports batch and incremental build.
- **Trace (Episodic Graph):** Directed causal edges (CausedBy, Follows, Supports, Contradicts) with BFS transitive traversal. Used for trace-boosted retrieval — connected memories rank higher.
- **Write-Ahead Log (WAL):** Every trace mutation is logged with fsync before in-memory application. Replayed on open, checkpointed after successful refresh.

**5. Governance Layer: Adaptive Knowledge Lifecycle (AKL)**

Autonomous lifecycle management prevents infinite accumulation and surfaces proven memories.

- **Importance Scoring:** ι ∈ [0, 100]. +3 on read, +5 on write. Daily decay ×0.995 (~138-day half-life).
- **Maturity Tiers:** Draft → Validated (ι ≥ 65) → Core (ι ≥ 85) with hysteresis gaps preventing oscillation.
- **Tier-Based Retrieval Boost:** Core 1.25×, Validated 1.1×, Draft 1.0× — makes tiers behaviorally meaningful.
- **Recency Decay:** r = exp(−Δt/30) applied as retrieval score multiplier. ~21-day half-life.
- **Eviction:** `evict_draft_packs(threshold, owner)` removes low-importance Draft packs. Validated/Core never evicted.

**6. Engine Facade**

- **Memento Pattern:** On open/refresh, all in-memory state is rebuilt from durable sources (segments, WAL, deletion log, text store). Crash at any point is recoverable.
- **Cross-Process Sync:** `Engine::refresh()` re-syncs from disk without dropping the instance. Used by vLLM connector for scheduler ↔ worker coordination.
- **WAL Checkpointing:** WAL truncated after successful replay in refresh(). Prevents unbounded growth.
- **Status API:** `Engine::status()` returns a snapshot of cell count, pack count, segments, SLB occupancy, Vamana state, pipeline stages, governance entries, and trace edges.

**7. Future Work (Not Yet Implemented)**

- **GPU DMA Offloading:** Direct NVMe→GPU transfers via cuFile/GDS.
- **Decoupled Position Encoding:** RoPE remapping for cross-prompt KV injection.
- **RelayCaching:** Cross-agent KV cache reuse (estimated 4.7× TTFT reduction from literature).
- **BatchQuantizedKVCache:** Concurrent Q4 inference across multiple agents.
- **Vamana Edge Persistence:** Avoid O(n²) rebuild on refresh.
- **Segment Compaction:** Reclaim space from deleted packs.
