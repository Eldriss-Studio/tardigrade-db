//! `CellSource` — the storage seam for lazy retrievers.
//!
//! Retrievers that want to score cell data on demand (rather than holding
//! a full in-memory copy) depend on this trait instead of `BlockPool`
//! directly. The trait is intentionally minimal: one operation, one
//! return type, no error channel beyond `Option` for "not found." Bigger
//! surface would tie retrievers to storage semantics (segments, fsync,
//! recovery) they have no business knowing.
//!
//! # Ownership model
//!
//! Sources are passed **by borrowed reference** to retriever and pipeline
//! query methods. The retriever does not store the source; it is borrowed
//! for the duration of one query. This keeps `BlockPool` ownership in
//! the engine and avoids `Arc<RwLock<BlockPool>>` plumbing through the
//! retrieval crate.
//!
//! # Design pattern
//!
//! Strategy: callers depend on the trait; concrete storage backends
//! provide the impl. Test doubles implement the trait directly without
//! any storage layer.
//!
//! # Example test double
//!
//! ```
//! use std::collections::HashMap;
//! use tdb_core::CellId;
//! use tdb_retrieval::cell_source::CellSource;
//!
//! struct InMemoryCellSource {
//!     cells: HashMap<CellId, Vec<f32>>,
//! }
//!
//! impl CellSource for InMemoryCellSource {
//!     fn get_encoded_key(&self, id: CellId) -> Option<Vec<f32>> {
//!         self.cells.get(&id).cloned()
//!     }
//! }
//! ```

use tdb_core::CellId;

/// Storage seam for retrievers that score cells on demand.
///
/// Implementations must be safe to call from multiple threads — the
/// retriever holds an `Arc<dyn CellSource>` and may dispatch from
/// concurrent query contexts.
pub trait CellSource: Send + Sync {
    /// Return a clone of the cell's encoded retrieval key.
    ///
    /// Returns `None` if the cell does not exist (e.g., deleted,
    /// never written, or `id` outside the source's domain). Callers
    /// must treat `None` as "skip this candidate," not as an error.
    fn get_encoded_key(&self, id: CellId) -> Option<Vec<f32>>;
}
