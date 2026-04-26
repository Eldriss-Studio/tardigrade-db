//! KV Pack — a complete multi-layer KV cache stored as a single memory entity.
//!
//! A KV Pack captures the full `past_key_values` from one model inference pass.
//! It contains per-layer K+V payloads for injection and a retrieval key (hidden
//! state summary) for searching.
//!
//! ```text
//! Model inference on "Sonia's wifi password is mango-cathedral-7"
//!   │
//!   ├── Layer 0:  K+V tensors ─┐
//!   ├── Layer 1:  K+V tensors  │
//!   ├── ...                    ├── KVPack (stored atomically, retrieved as unit)
//!   ├── Layer 27: K+V tensors ─┘
//!   │
//!   └── Hidden states ──────────── Retrieval key (per-token encoded)
//! ```
//!
//! Repository pattern: the engine treats a pack as a single entity for storage,
//! retrieval, governance, and deletion — not as 28 scattered cells.

use crate::{OwnerId, Tier};

/// Unique identifier for a KV Pack.
pub type PackId = u64;

/// Per-layer K+V payload within a KV Pack.
#[derive(Debug, Clone)]
pub struct KVLayerPayload {
    /// Transformer layer index.
    pub layer_idx: u16,
    /// Flattened `[K_flat | V_flat]` where each is `(seq_len * kv_heads * head_dim)` floats.
    pub data: Vec<f32>,
}

/// A complete multi-layer KV cache captured from one inference pass.
///
/// Stored atomically (single fsync), retrieved as a unit, governed as one memory.
/// Optionally carries the original fact text for durable text persistence
/// alongside the tensor data.
///
/// # Examples
///
/// Construct a pack with the original fact text — the engine persists it
/// in the durable text store alongside the tensor data:
///
/// ```rust
/// use tdb_core::kv_pack::{KVLayerPayload, KVPack};
///
/// let pack = KVPack {
///     id: 0, // assigned by the engine on write
///     owner: 1,
///     retrieval_key: vec![0.0f32; 128],
///     layers: vec![KVLayerPayload {
///         layer_idx: 0,
///         data: vec![0.1f32; 64],
///     }],
///     salience: 80.0,
///     text: Some("User prefers morning meetings".into()),
/// };
///
/// assert_eq!(pack.text.as_deref(), Some("User prefers morning meetings"));
/// ```
///
/// `text: None` is valid for packs that don't carry source text (e.g.,
/// pre-Item-3 databases or pure-tensor workflows).
#[derive(Debug, Clone)]
pub struct KVPack {
    /// Unique pack identifier (assigned by engine on write).
    pub id: PackId,
    /// Agent/user that owns this memory.
    pub owner: OwnerId,
    /// Per-token encoded hidden state summary for retrieval scoring.
    ///
    /// The encoded key uses the retrieval crate's 64-float Q4-safe header:
    /// sentinel at index 0, metadata at indices 32 and 33, token data at 64+.
    pub retrieval_key: Vec<f32>,
    /// Per-layer K+V payloads for injection.
    pub layers: Vec<KVLayerPayload>,
    /// Initial importance hint (0-100).
    pub salience: f32,
    /// Original fact text, persisted in a durable append-only text store.
    ///
    /// `None` for packs created before text storage was added (backward compatible).
    pub text: Option<String>,
}

/// Metadata about a stored pack, returned by retrieval.
#[derive(Debug)]
pub struct PackReadResult {
    /// The complete KV Pack with all layer payloads.
    pub pack: KVPack,
    /// Retrieval score (attention-space similarity).
    pub score: f32,
    /// Current governance tier.
    pub tier: Tier,
}
