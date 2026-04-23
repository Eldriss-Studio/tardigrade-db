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

use tdb_core::{CellId, OwnerId};

use crate::attention::RetrievalResult;

/// Strategy interface for retrieval backends.
///
/// Implementations must support both insertion (indexing) and querying.
/// The `query` method takes `&mut self` because some implementations
/// (e.g., SLB) update internal state on access (LRU tracking).
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
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult>;

    /// Insert a key vector for a cell into the retrieval index.
    fn insert(&mut self, cell_id: CellId, owner: OwnerId, key: &[f32]);

    /// Number of entries in this retriever.
    fn len(&self) -> usize;

    /// Whether this retriever has no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
