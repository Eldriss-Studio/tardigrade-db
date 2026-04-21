//! Top-level engine — the unified entry point for `TardigradeDB`.
//!
//! `tdb-engine` exposes a single [`Engine`] struct that coordinates the four
//! architectural layers behind two operations: `MEM.WRITE` and `MEM.READ`.
//! Everything else — quantization, indexing, importance scoring, tier transitions —
//! happens automatically as a side effect of those two calls.
//!
//! # System Overview
//!
//! ```text
//!                       ┌──────────────────────────┐
//!    inference pass ──► │   Engine (Facade)        │ ◄── query key
//!                       │                          │
//!                       │  mem_write(owner, layer, │
//!                       │    key, value, salience) │
//!                       │  mem_read(query, k,      │
//!                       │    owner_filter)         │
//!                       └──────────┬───────────────┘
//!                                  │ coordinates
//!                    ┌─────────────┼──────────────────┐
//!                    ▼             ▼                  ▼
//!              BlockPool    BruteForceRetriever   ImportanceScorer
//!              (tdb-storage) (tdb-retrieval)    + TierStateMachine
//!                                               (tdb-governance)
//! ```
//!
//! # Write Path — `mem_write`
//!
//! ```text
//! mem_write(owner, layer, key, value, salience)
//!   1. Assign a monotone CellId
//!   2. Build a MemoryCell with current timestamp
//!   3. BlockPool::append  → Q4-compress and fsync to segment file
//!   4. BruteForceRetriever::insert  → index key for latent-space queries
//!   5. ImportanceScorer::new(salience) + on_update (+5)
//!   6. TierStateMachine::evaluate  → may immediately promote to Validated/Core
//! ```
//!
//! # Read Path — `mem_read`
//!
//! ```text
//! mem_read(query_key, k, owner_filter)
//!   1. BruteForceRetriever::query(k*2)  → raw attention scores
//!   2. recency_decay(days_since_update)  → down-weight stale cells
//!   3. adjusted_score = raw_score × recency_factor
//!   4. BlockPool::get  → dequantize cell from disk
//!   5. ImportanceScorer::on_access (+3)  → boost accessed cells
//!   6. TierStateMachine::evaluate  → potential promotion
//!   7. Return top-k by adjusted score
//! ```
//!
//! # Current Limitations (Phase 1)
//!
//! The engine is **partially ephemeral**: cells are durable on disk (via `BlockPool`),
//! but the retrieval index and governance state are rebuilt in-memory from scratch
//! on each `Engine::open`. A recovery pass that scans the `BlockPool` and rehydrates
//! the `BruteForceRetriever` and `HashMap<CellId, CellGovernance>` is planned for
//! Phase 2 (see the `TODO` in [`engine::Engine::open`]).
//!
//! # Usage
//!
//! ```rust,no_run
//! use tdb_engine::engine::Engine;
//!
//! let dir = std::path::Path::new("/tmp/tardigrade");
//! let mut engine = Engine::open(dir).unwrap();
//!
//! // Capture a KV pair from layer 12.
//! let cell_id = engine.mem_write(
//!     /* owner */ 42,
//!     /* layer */ 12,
//!     /* key */ &[0.1f32; 128],
//!     /* value */ vec![0.2f32; 128],
//!     /* salience */ 60.0,
//! ).unwrap();
//!
//! // Retrieve the most relevant cell for a query key.
//! let results = engine.mem_read(&[0.1f32; 128], /* k */ 5, None).unwrap();
//! if let Some(top) = results.first() {
//!     println!("top cell: id={} score={:.3} tier={:?}", top.cell.id, top.score, top.tier);
//! }
//! ```
//!
//! [`Engine`]: engine::Engine
//! [`engine::Engine::open`]: engine::Engine::open

#![deny(unsafe_code)]

pub mod batch_cache;
pub mod engine;
pub mod scheduler;
