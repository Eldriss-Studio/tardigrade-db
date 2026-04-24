//! Shared types, error definitions, and the fundamental storage unit of `TardigradeDB`.
//!
//! `tdb-core` is the vocabulary crate — it defines the language every other crate
//! speaks. No business logic lives here. If you are adding a new field to a stored
//! tensor or a new error variant, this is where you start.
//!
//! # What is a [`MemoryCell`]?
//!
//! `TardigradeDB` does not store text or embeddings. It stores **KV-cache tensors** —
//! the exact `(key, value)` vectors a transformer produces at a specific layer during
//! inference. A [`MemoryCell`] is the on-disk representation of one such tensor pair,
//! along with enough metadata for the engine to manage its lifecycle autonomously.
//!
//! ```text
//! Transformer attention head (layer L)
//!   ┌──────────────────────────────────┐
//!   │  Q · K^T / √d_k → softmax → · V │
//!   │      ↑              ↑            │
//!   │    key[L]        value[L]        │
//!   └──────┬──────────────┬────────────┘
//!          │              │    captured as f32 vectors
//!          └──────┬───────┘
//!                 ▼
//!           MemoryCell { id, owner, layer, key, value, pos_encoding, meta }
//! ```
//!
//! Key vectors are used for **retrieval** (attention scoring against a query).
//! Value vectors are used for **injection** (inserted directly into the attention
//! stack at read time, bypassing tokenization entirely).
//!
//! # Maturity Lifecycle — [`Tier`]
//!
//! Every cell is assigned a maturity tier that governs its storage priority and
//! eviction eligibility:
//!
//! ```text
//! Draft ──(ι ≥ 65)──► Validated ──(ι ≥ 85)──► Core
//!       ◄──(ι < 35)──            ◄──(ι < 60)──
//! ```
//!
//! Hysteresis gaps (30 points for Draft↔Validated, 25 for Validated↔Core) prevent
//! oscillation when scores hover near a boundary. See `tdb-governance` for the
//! importance score (ι) algorithm.
//!
//! # Error Handling
//!
//! All fallible operations across the workspace return [`error::Result<T>`], which
//! wraps [`TardigradeError`]. Errors are defined here so that `tdb-storage`,
//! `tdb-retrieval`, and other crates can return the same type without circular deps.
//!
//! # Crate Map
//!
//! | Crate | Role |
//! |---|---|
//! | **tdb-core** (this) | Shared types — `MemoryCell`, `Tier`, `TardigradeError` |
//! | `tdb-storage` | Persist cells to append-only segments (Q4 quantized) |
//! | `tdb-retrieval` | Retrieve cells by latent-space attention scoring |
//! | `tdb-index` | Vamana ANN graph + causal Trace graph + WAL |
//! | `tdb-governance` | Importance scoring, tier transitions, recency decay |
//! | `tdb-engine` | Top-level facade orchestrating all layers |
//!
//! # Quick Example
//!
//! ```rust
//! use tdb_core::memory_cell::MemoryCellBuilder;
//! use tdb_core::types::Tier;
//!
//! // Capture a KV pair from layer 12 of a 4096-dim model.
//! let cell = MemoryCellBuilder::new(
//!     /* id */ 1,
//!     /* owner */ 42,
//!     /* layer */ 12,
//!     /* key */ vec![0.1f32; 128],
//!     /* value */ vec![0.2f32; 128],
//! )
//! .token_span(0, 64)
//! .importance(50.0)
//! .tier(Tier::Draft)
//! .build();
//!
//! assert_eq!(cell.layer, 12);
//! assert_eq!(cell.meta.tier, Tier::Draft);
//! ```

#![deny(unsafe_code)]

pub mod error;
pub mod kv_pack;
pub mod memory_cell;
pub mod synaptic_bank;
pub mod types;

pub use error::TardigradeError;
pub use kv_pack::{KVLayerPayload, KVPack, PackId, PackReadResult};
pub use memory_cell::MemoryCell;
pub use synaptic_bank::SynapticBankEntry;
pub use types::{CellId, LayerId, OwnerId, SynapticId, TagBits, Tier};
