//! Latent-space retrieval for `TardigradeDB` — find relevant memory cells by attention scoring.
//!
//! `TardigradeDB` does not use cosine similarity over text embeddings. Relevance is computed
//! as **scaled dot-product attention** directly in the model's latent space:
//!
//! ```text
//! score(q, k) = (q · k) / √d_k
//! ```
//!
//! This is the same formula used inside a transformer's attention head. Querying memory
//! with a key vector from a live inference pass means the retrieval metric is identical
//! to how the model itself relates tokens — no separate embedding model required.
//!
//! # Two-Level Retrieval
//!
//! ```text
//! query key vector
//!       │
//!       ▼
//! ┌─────────────────────────────────────────┐
//! │  SemanticLookasideBuffer (hot path)     │
//! │  • Fixed-capacity LRU, INT8 keys        │
//! │  • Target: < 5 μs via SIMD dot product  │
//! │  • HIT → return immediately             │
//! └─────────────┬───────────────────────────┘
//!               │ MISS
//!               ▼
//! ┌─────────────────────────────────────────┐
//! │  BruteForceRetriever (cold path)        │
//! │  • Full f32 exhaustive scan             │
//! │  • Owner-filtered per-agent partitions  │
//! │  • Valid up to ~10K cells per agent     │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Why Brute Force, Not ANN?
//!
//! The [MemArt paper] (which this architecture cites) demonstrates that at typical
//! per-agent memory scales (<10K blocks), SIMD brute-force matmul outperforms HNSW
//! and other approximate nearest-neighbor indexes. ANN indexes pay a build-time and
//! index-memory overhead that only pays off beyond ~100K vectors. For larger cold
//! indexes, `tdb-index` provides a Vamana graph (DiskANN-style).
//!
//! Additionally, HNSW is unreliable for Q/K distributions that shift over time
//! (which happens as the model processes different contexts), because its graph
//! edges are built against a snapshot distribution.
//!
//! ## Semantic Lookaside Buffer
//!
//! The [`SemanticLookasideBuffer`] is modeled on a CPU's Translation Lookaside Buffer:
//! a fast fixed-size cache for the most recently used entries. It stores keys in
//! **symmetric INT8 quantization** — one scale per full vector instead of per-group —
//! which allows the dot product to be computed as integer arithmetic using NEON
//! `SDOT` intrinsics (ARM) or equivalent auto-vectorized code on x86.
//!
//! # Usage
//!
//! ```rust
//! use tdb_retrieval::attention::BruteForceRetriever;
//!
//! let mut retriever = BruteForceRetriever::new();
//!
//! // Index a set of key vectors (from stored cells).
//! retriever.insert(/* cell_id */ 0, /* owner */ 1, /* layer */ 12, &[1.0, 0.0, 0.0, 0.0]);
//! retriever.insert(/* cell_id */ 1, /* owner */ 1, /* layer */ 12, &[0.0, 1.0, 0.0, 0.0]);
//!
//! // Query with a key from the current inference pass.
//! let results = retriever.query(&[1.0, 0.0, 0.0, 0.0], /* k */ 1, /* owner_filter */ None);
//! assert_eq!(results[0].cell_id, 0); // exact-match cell scores highest
//! ```
//!
//! [`SemanticLookasideBuffer`]: slb::SemanticLookasideBuffer
//! [MemArt paper]: https://arxiv.org/abs/2409.17264

pub mod attention;
pub mod int8_quant;
pub mod per_token;
pub mod pipeline;
pub mod retriever;
pub mod simd_distance;
pub mod slb;
