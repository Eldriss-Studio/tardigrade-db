### 1. Storage Layer: Persistent Quantized KV-Cache

The storage engine persists the LLM's Key-Value (KV) cache directly to disk as quantized binary segments.

- **Data Format & Quantization:** KV tensors are stored in 4-bit group-wise quantization (Q4_0, same scheme as llama.cpp). 4× compression vs FP32 with MSE < 0.01 on typical activation distributions. Position encodings are stored unquantized (f32) for exact reproduction.
- **Append-Only Segments:** 256MB segment files with binary length-prefixed records. `sync_data()` on every write for crash durability. Partial record detection on recovery discards incomplete writes.
- **Segment Scanning Recovery:** On open, segments are scanned to rebuild the `CellId → (segment, offset)` index. O(n) in cell count, not cell size. No external dependency — the segment files are the source of truth.
- **Text Store:** Append-only binary sidecar for fact text associated with KV packs. Single source of truth (JSON sidecar removed in P1).
- **Deletion Log:** Durable append-only log of deleted pack IDs. Applied during `refresh()` to filter deleted packs from in-memory state.
- **Segment Compaction:** Mark-Sweep GC rewrites segments below 50% live ratio, reclaiming space from deleted packs. `Engine::compact()` exposed to Python. Crash-safe (segments are rebuilt atomically; partial work is discarded on recovery).

### 2. Retrieval Layer: Latent Space Attention & Semantic Lookaside Buffers

Retrieval computes relevance directly in the model's latent space using per-token attention scoring.

- **Per-Token Top5Avg Scoring:** Stores per-token hidden state vectors (not mean-pooled). Scores by computing dot products between all query and memory token pairs, then averaging the top 5. Achieves 100% recall at 100 memories through the full engine pipeline (Q4 quantized, INT8 scoring). Validated on Qwen3-0.6B.
- **Semantic Lookaside Buffer (SLB):** Fixed-capacity LRU cache storing mean-pooled keys in symmetric INT8 quantization. Sub-5μs retrieval at 4096 entries using NEON SDOT intrinsics (ARM) with auto-vectorized fallback (x86).
- **Three-Stage Retrieval Pipeline:** SLB (hot, INT8) → PerTokenRetriever (primary, Top5Avg) → BruteForce (fallback, exact). Chain of Responsibility pattern with CellId deduplication.
- **Active Governance Integration:** Retrieval scores are multiplied by recency decay and tier boost (Core 1.25×, Validated 1.1×, Draft 1.0×). Results are re-sorted by adjusted score after governance.

### 3. Organization Layer: Vamana Graph Index + Causal Trace + WAL

KV blocks are organized via a dual topology: a navigable ANN graph for similarity and a causal graph for episodic relationships.

- **Vamana Graph Index:** DiskANN-style single-layer graph with robust pruning (angular diversity). Supports batch build and incremental `insert_online`. Lazily activated when cell count crosses a configurable threshold. Rebuilt from scratch on `refresh()` (no edge persistence yet).
- **Trace Causal Graph:** Directed edges with four types: CausedBy (parent tracking), Follows (pack links), Supports, Contradicts (defined, not yet auto-populated by the engine). BFS transitive ancestor traversal.
- **Write-Ahead Log (WAL):** Every trace edge is logged (with fsync) before in-memory application. Replayed on engine open for crash recovery. Checkpointed after successful `refresh()` to prevent unbounded growth. Lenient recovery: partial records are discarded.

### 4. Governance Layer: Adaptive Knowledge Lifecycle (AKL)

Autonomous memory management that prevents infinite accumulation and surfaces proven memories.

- **Importance Scoring (ι):** ι ∈ [0, 100]. Read access: +3. Write/update: +5. Daily decay: ×0.995 (~138-day half-life for untouched cells).
- **Recency Decay:** r = exp(−Δt / 30), applied as a retrieval score multiplier at query time. ~21-day half-life. Does not modify stored ι — only affects ranking.
- **Maturity Tiers with Hysteresis:** Draft → Validated (ι ≥ 65, demote < 35) → Core (ι ≥ 85, demote < 60). 30-point and 25-point hysteresis gaps prevent oscillation. Skip-tier promotion supported.
- **Tier-Based Retrieval Boost:** Core memories receive 1.25× retrieval score boost, Validated 1.1×, Draft 1.0×. This makes tiers behaviorally meaningful — proven memories rank higher.
- **Eviction:** `evict_draft_packs(threshold, owner)` removes Draft-tier packs below importance threshold. Validated and Core packs are never evicted. Owner-scoped.

## Future Work

These capabilities are described in the original TDD but are **not yet implemented**:

- **GPU DMA Offloading:** Direct NVMe→GPU transfers via cuFile/GDS, bypassing CPU bounce buffers. Requires CUDA SDK integration.
- **Decoupled Position Encoding:** RoPE remapping for safe cross-prompt KV injection. Designed but not needed for current deployment paths.
- **RelayCaching:** Cross-agent KV cache reuse for multi-agent handoffs. Estimated 4.7× TTFT reduction (from literature). Requires multi-agent scheduling logic.
- **BatchQuantizedKVCache:** Concurrent Q4 inference across multiple agents. Stub only.
- **Vamana Edge Persistence:** Serializing graph edges to disk for O(n) load on refresh instead of O(n²) rebuild.
