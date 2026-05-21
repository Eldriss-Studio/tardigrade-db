//! Retriever trait — Strategy pattern for pluggable retrieval backends.
//!
//! Unifies [`BruteForceRetriever`](crate::attention::BruteForceRetriever) and
//! [`SemanticLookasideBuffer`](crate::slb::SemanticLookasideBuffer) behind a
//! common interface so the engine can compose retrieval strategies via
//! [`RetrieverPipeline`](crate::pipeline::RetrieverPipeline).
//!
//! Each retriever receives an `owner_filter` parameter. Retrievers that don't
//! support internal owner filtering (SLB, Vamana) ignore it — the pipeline
//! handles post-filtering when needed.

use std::any::Any;
use std::collections::HashSet;

use tdb_core::{CellId, OwnerId};

use crate::attention::RetrievalResult;
use crate::cell_source::CellSource;

/// Strategy interface for retrieval backends.
///
/// Implementations must support both insertion (indexing) and querying.
/// Query methods take `&self` so multiple readers can run in parallel
/// under an outer `RwLock::read()` guard; implementations that need to
/// update bookkeeping on access (e.g. SLB LRU, `PerTokenRetriever`
/// metrics) do so via interior mutability with `Relaxed` atomics. The
/// outer writer lock provides the happens-before edge that makes the
/// `Relaxed` ordering safe.
///
/// Mutating methods (`insert`, `remove`, `as_any_mut`) keep `&mut self`
/// — they run under the outer writer lock with no concurrent reader.
///
/// The trait is intentionally **not** split into separate Read / Write
/// halves: every production impl needs both, the pipeline stores a
/// single `Vec<Box<dyn Retriever>>`, and a sub-trait split would just
/// reproduce today's interface in two pieces with no caller benefit.
///
/// Requires `Send + Sync` because the engine may be wrapped in a `PyO3`
/// `#[pyclass]` or shared across threads.
pub trait Retriever: Send + Sync {
    /// Query for the top-k most relevant cells by attention score.
    ///
    /// `owner_filter`: if `Some(id)`, prefer results from that owner.
    /// Retrievers that don't support internal filtering may return
    /// results from any owner — the caller is responsible for final filtering.
    fn query(
        &self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult>;

    /// Insert a key vector for a cell into the retrieval index.
    fn insert(&mut self, cell_id: CellId, owner: OwnerId, key: &[f32]);

    /// Remove a cell from the retrieval index.
    ///
    /// Implementations that don't support removal (e.g., Vamana) should no-op.
    /// The caller is responsible for filtering deleted cells at a higher level.
    fn remove(&mut self, cell_id: CellId);

    /// Number of entries in this retriever.
    fn len(&self) -> usize;

    /// Whether this retriever has no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Mutable downcast access for stage-specific operations (e.g. refinement
    /// hooks reaching into [`PerTokenRetriever`](crate::per_token::PerTokenRetriever)).
    /// Default returns `None`; impls override to expose their concrete type.
    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        None
    }

    /// Query with optional upstream candidate context and a [`CellSource`].
    ///
    /// Stages that don't need either of these (SLB, Vamana, brute force)
    /// keep the default impl, which ignores both parameters and delegates
    /// to [`Retriever::query`]. Stages that benefit from one or both
    /// (e.g. `PerTokenRetriever` in lazy mode) override.
    ///
    /// `candidates`: the set of cell IDs already yielded by upstream
    /// stages. Stages that respect this filter will not score cells
    /// outside the set.
    ///
    /// `source`: where to load cell data on demand. Stages that store
    /// everything in RAM ignore this. Stages that don't store data at
    /// all (lazy retrievers) require it.
    fn query_with_source(
        &self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
        _candidates: Option<&HashSet<CellId>>,
        _source: Option<&dyn CellSource>,
    ) -> Vec<RetrievalResult> {
        self.query(query_key, k, owner_filter)
    }
}
