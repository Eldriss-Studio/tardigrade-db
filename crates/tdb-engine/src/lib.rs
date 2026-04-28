//! The brain of `TardigradeDB` — a unified facade that turns five independent
//! subsystems into a single, coherent memory for autonomous AI agents.
//!
//! # Why does this crate exist?
//!
//! An LLM agent that can reason over its own history needs four things to happen
//! on every interaction: **persist** the current KV cache, **retrieve** relevant
//! past context, **organize** causal relationships between memories, and **govern**
//! which memories survive over time. Each of these is handled by a separate crate
//! in `TardigradeDB` — but no agent should need to coordinate them manually.
//!
//! `tdb-engine` is the **Facade** that hides that complexity. It exposes two
//! primary operations — [`mem_write`] and [`mem_read`] — and everything else
//! (quantization, indexing, importance scoring, tier promotion, WAL logging,
//! causal edge tracking) happens as automatic side effects.
//!
//! # What happens when you write
//!
//! A single call to [`mem_write`] triggers a cascade across all four layers:
//!
//! ```text
//! mem_write(owner, layer, key, value, salience, parent)
//!   │
//!   ├─ Governance:  ImportanceScorer::new(salience) + on_update(+5)
//!   │               TierStateMachine::evaluate → Draft / Validated / Core
//!   │
//!   ├─ Storage:     BlockPool::append → Q4-compress key+value, fsync to segment
//!   │               (on-disk metadata includes importance + tier for crash recovery)
//!   │
//!   ├─ Retrieval:   Pipeline::insert → PerTokenRetriever + BruteForce
//!   │               SLB::insert → cache mean-pooled key for hot-path lookups
//!   │
//!   ├─ Index:       if Vamana active → insert (graph ANN)
//!   │               if cell_count ≥ threshold → lazy-activate Vamana
//!   │
//!   └─ Trace:       if parent_cell_id provided →
//!                     WAL::append(AddEdge) → fsync (Observer pattern)
//!                     TraceGraph::add_edge(child → parent, CausedBy)
//! ```
//!
//! Governance is computed **before** persistence so the on-disk metadata matches
//! the in-memory state — this is required for correct Memento-pattern recovery
//! when the engine reopens.
//!
//! # What happens when you read
//!
//! A single call to [`mem_read`] runs a three-stage retrieval chain
//! (**Chain of Responsibility** pattern), merging results across stages:
//!
//! ```text
//! mem_read(query_key, k, owner_filter)
//!   │
//!   ├─ SLB (hot cache, separate from pipeline)
//!   │  INT8 quantized mean-pooled keys, NEON SDOT / AVX2
//!   │  Sub-5μs per query at 4096 entries
//!   │  Exploits conversational locality — recent cells are cached
//!   │
//!   ├─ Pipeline Stage 1: PerTokenRetriever (primary)
//!   │  Inverted Multi-Key Index with Top5Avg scoring
//!   │  INT8 per-token dot products, tuneable via PerTokenConfig
//!   │  100% recall at 100 memories (validated on Qwen3-0.6B)
//!   │
//!   ├─ Pipeline Stage 2: BruteForce (fallback)
//!   │  Exhaustive scaled dot-product: score = (q · k) / √d_k
//!   │  Exact results, owner-filtered per-agent partitions
//!   │
//!   └─ Pipeline Stage 3: Vamana (optional, lazy-activated)
//!      DiskANN-style graph with robust pruning
//!      Activated when cell count crosses threshold
//!
//!   Then for each result:
//!     → recency_decay(Δt) multiplier (21-day half-life)
//!     → BlockPool::get → dequantize full cell from disk
//!     → ImportanceScorer::on_access(+3) → potential tier promotion
//!     → SLB::insert → cache for future hot-path hits
//! ```
//!
//! Results are deduplicated by `CellId`, sorted by decay-adjusted score, and
//! truncated to k.
//!
//! # Crash recovery and cross-process sync (Memento pattern)
//!
//! On [`Engine::open`] and [`refresh`], the engine rebuilds all in-memory
//! derived state from durable sources:
//!
//! ```text
//! Engine::open(dir) / Engine::refresh()
//!   ├─ BlockPool::open/refresh_index → scan segments
//!   ├─ Pipeline rebuild → fresh PerTokenRetriever + BruteForce
//!   │  (Vamana reset; re-activated lazily if cell count ≥ threshold)
//!   ├─ For each persisted cell:
//!   │    Pipeline::insert(key) + SLB::insert(mean_pooled_key)
//!   │    ImportanceScorer::new(persisted_importance)
//!   │    TierStateMachine::with_tier(persisted_tier)
//!   ├─ WAL::replay → rebuild TraceGraph from logged edges
//!   ├─ PackDirectory::from_cells → rebuild pack membership
//!   ├─ DeletionLog::refresh → filter deleted packs
//!   └─ next_id = max(persisted_ids) + 1
//! ```
//!
//! `refresh()` provides the same guarantee as a fresh `open()`: two Engine
//! handles sharing a path stay synchronized. The vLLM connector relies on
//! this for scheduler ↔ worker coordination.
//!
//! The engine can crash at any point and recover all state that was durably
//! committed (fsync'd). In-flight writes that didn't complete fsync are
//! discarded — partial record detection in segment scanning handles this.
//!
//! # Causal memory (Trace graph)
//!
//! When a cell is written with `parent_cell_id = Some(parent)`, the engine
//! records a directed `CausedBy` edge in the Trace graph, durably logged via
//! the WAL before being applied in memory. This enables causal chain queries:
//!
//! ```text
//! trace_ancestors(cell_c) → [cell_b, cell_a]
//! ```
//!
//! If cell C was caused by B, and B was caused by A, the engine traverses
//! the causal graph transitively.
//!
//! # Synaptic bank (`LoRA` adapters)
//!
//! Beyond episodic KV memory, the engine also persists per-agent **`LoRA` adapters**
//! via [`store_synapsis`] / [`load_synapsis`]. These encode stable preferences and
//! patterns as low-rank weight deltas applied to the base model at runtime —
//! the "slow memory" complement to the fast episodic KV cache.
//!
//! # Python bridge
//!
//! The `tdb-python` crate wraps this engine via `PyO3`. From Python:
//!
//! ```python
//! import tardigrade_db as tdb
//! engine = tdb.Engine("/tmp/tardigrade")
//! cell_id = engine.mem_write(owner=42, layer=12, key=np_key, value=np_val, salience=80.0, parent_cell_id=None)
//! results = engine.mem_read(query_key=np_query, k=5, owner=None)
//! ancestors = engine.trace_ancestors(cell_id)
//! ```
//!
//! # Usage (Rust)
//!
//! ```rust,no_run
//! use tdb_engine::engine::Engine;
//!
//! let dir = std::path::Path::new("/tmp/tardigrade");
//! let mut engine = Engine::open(dir).unwrap();
//!
//! // Capture a KV pair from transformer layer 12.
//! let cell_id = engine.mem_write(
//!     /* owner */ 42,
//!     /* layer */ 12,
//!     /* key */ &[0.1f32; 128],
//!     /* value */ vec![0.2f32; 128],
//!     /* salience */ 60.0,
//!     /* parent */ None,
//! ).unwrap();
//!
//! // Retrieve the most relevant cells for a query key.
//! let results = engine.mem_read(&[0.1f32; 128], /* k */ 5, None).unwrap();
//! if let Some(top) = results.first() {
//!     println!("cell {} score={:.3} tier={:?}", top.cell.id, top.score, top.tier);
//! }
//!
//! // Query causal ancestors.
//! let ancestors = engine.trace_ancestors(cell_id);
//!
//! // Close and reopen — all state is recovered from disk.
//! drop(engine);
//! let engine2 = Engine::open(dir).unwrap();
//! assert_eq!(engine2.cell_count(), 1);
//! ```
//!
//! [`Engine`]: engine::Engine
//! [`Engine::open`]: engine::Engine::open
//! [`mem_write`]: engine::Engine::mem_write
//! [`mem_read`]: engine::Engine::mem_read
//! [`refresh`]: engine::Engine::refresh
//! [`store_synapsis`]: engine::Engine::store_synapsis
//! [`load_synapsis`]: engine::Engine::load_synapsis

#![deny(unsafe_code)]

pub mod engine;
pub(crate) mod pack_directory;
pub(crate) mod pack_materialization;
#[cfg(test)]
pub(crate) mod pack_read_profile;
pub mod scheduler;

pub use tdb_index::trace::EdgeType;
