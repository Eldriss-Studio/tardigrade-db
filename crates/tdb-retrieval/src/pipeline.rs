//! `RetrieverPipeline` — Chain of Responsibility via trait objects.
//!
//! Tries each retriever in order, merging results by `CellId`.
//! Short-circuits when `k` unique results have been collected.
//!
//! ```text
//! query → Stage 0 (SLB) → Stage 1 (Vamana) → Stage 2 (BruteForce)
//!                  │                │                    │
//!                  └── merge unique results, stop when k reached
//! ```

use std::collections::HashSet;

use tdb_core::{CellId, OwnerId};

use crate::attention::RetrievalResult;
use crate::cell_source::CellSource;
use crate::retriever::Retriever;

/// Chain of Responsibility combinator: tries retrievers in order, short-circuits at k.
///
/// Each stage queries for `k * oversample_factor` candidates (default 2x)
/// to account for deduplication and owner filtering removing some results.
pub struct RetrieverPipeline {
    stages: Vec<Box<dyn Retriever>>,
    /// Multiplier for how many candidates each stage requests (default: 2).
    oversample_factor: usize,
}

impl RetrieverPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new(), oversample_factor: 2 }
    }

    /// Add a retriever stage to the end of the pipeline.
    pub fn add_stage(&mut self, retriever: Box<dyn Retriever>) {
        self.stages.push(retriever);
    }

    /// Query the pipeline: tries each stage in order, short-circuits at k unique results.
    ///
    /// Backward-compat shim: calls [`Self::query_with_source`] with no
    /// source. Stages that need a source (e.g. lazy `PerTokenRetriever`)
    /// will fall back to their default eager path.
    pub fn query(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        self.query_with_source(query_key, k, owner_filter, None)
    }

    /// Query the pipeline with an optional [`CellSource`] for lazy stages.
    ///
    /// Threads the accumulated `seen` cell-ID set forward as
    /// `upstream_candidates` so downstream stages can narrow their scan
    /// to cells nominated by earlier stages (Chain of Responsibility with
    /// enriched context).
    pub fn query_with_source(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
        source: Option<&dyn CellSource>,
    ) -> Vec<RetrievalResult> {
        let mut seen: HashSet<CellId> = HashSet::new();
        let mut candidates: Vec<RetrievalResult> = Vec::new();

        for stage in &mut self.stages {
            if stage.is_empty() {
                continue;
            }

            let upstream = if seen.is_empty() { None } else { Some(&seen) };
            let stage_results = stage.query_with_source(
                query_key,
                k * self.oversample_factor,
                owner_filter,
                upstream,
                source,
            );
            for r in stage_results {
                if seen.insert(r.cell_id) {
                    candidates.push(r);
                }
            }

            if candidates.len() >= k {
                break;
            }
        }

        // Sort by score descending, truncate to k.
        candidates
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Insert into all stages.
    pub fn insert(&mut self, cell_id: CellId, owner: OwnerId, key: &[f32]) {
        for stage in &mut self.stages {
            stage.insert(cell_id, owner, key);
        }
    }

    /// Number of stages in the pipeline.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Remove all stages (Memento: discard derived state for rebuild).
    pub fn clear_stages(&mut self) {
        self.stages.clear();
    }

    /// Mutable access to the first stage that downcasts to `T`, if any.
    ///
    /// Used by refinement strategies that need direct access to a specific
    /// retriever (e.g. [`PerTokenRetriever`](crate::per_token::PerTokenRetriever)
    /// for corpus-mean access and PRF re-query).
    pub fn first_stage_as_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.stages.iter_mut().find_map(|stage| stage.as_any_mut()?.downcast_mut::<T>())
    }
}

impl std::fmt::Debug for RetrieverPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetrieverPipeline")
            .field("stages", &self.stages.len())
            .field("oversample_factor", &self.oversample_factor)
            .finish()
    }
}

impl Default for RetrieverPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Retriever for RetrieverPipeline {
    fn query(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        RetrieverPipeline::query(self, query_key, k, owner_filter)
    }

    fn insert(&mut self, cell_id: CellId, owner: OwnerId, key: &[f32]) {
        RetrieverPipeline::insert(self, cell_id, owner, key);
    }

    fn remove(&mut self, cell_id: CellId) {
        for stage in &mut self.stages {
            stage.remove(cell_id);
        }
    }

    fn len(&self) -> usize {
        self.stages.iter().map(|s| s.len()).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::attention::BruteForceRetriever;
    use crate::cell_source::CellSource;

    const PROBE_QUERY_K: usize = 5;
    const NOMINATED_CELL_ID_A: CellId = 1;
    const NOMINATED_CELL_ID_B: CellId = 2;
    const NOMINATED_CELL_ID_C: CellId = 3;
    const FIXED_RESULT_SCORE: f32 = 1.0;
    const FIXED_OWNER: OwnerId = 1;
    const SEED_CELL_ID: CellId = 0;
    const SEED_OWNER: OwnerId = 1;
    const UNUSED_QUERY: &[f32] = &[FIXED_RESULT_SCORE, 0.0, 0.0, 0.0];

    #[test]
    fn test_pipeline_clear_stages_empties_pipeline() {
        let mut pipeline = RetrieverPipeline::new();
        pipeline.add_stage(Box::new(BruteForceRetriever::new()));
        pipeline.add_stage(Box::new(BruteForceRetriever::new()));
        pipeline.insert(SEED_CELL_ID, SEED_OWNER, UNUSED_QUERY);
        assert_eq!(pipeline.stage_count(), 2);
        assert!(!pipeline.is_empty());

        pipeline.clear_stages();

        assert_eq!(pipeline.stage_count(), 0);
        assert!(pipeline.is_empty());
        let results = pipeline.query(UNUSED_QUERY, PROBE_QUERY_K, None);
        assert!(results.is_empty());
    }

    // ---------- AT-4 helpers ----------

    /// Stage that yields a fixed set of cell IDs regardless of query.
    /// Seeds the pipeline's `seen` set so downstream stages receive
    /// non-empty `upstream_candidates`.
    struct FixedYieldStage {
        ids: Vec<CellId>,
    }

    impl Retriever for FixedYieldStage {
        fn query(
            &mut self,
            _query_key: &[f32],
            _k: usize,
            _owner_filter: Option<OwnerId>,
        ) -> Vec<RetrievalResult> {
            self.ids
                .iter()
                .map(|cell_id| RetrievalResult {
                    cell_id: *cell_id,
                    owner: FIXED_OWNER,
                    score: FIXED_RESULT_SCORE,
                })
                .collect()
        }
        fn insert(&mut self, _cell_id: CellId, _owner: OwnerId, _key: &[f32]) {}
        fn remove(&mut self, _cell_id: CellId) {}
        fn len(&self) -> usize {
            self.ids.len()
        }
    }

    /// Records what `upstream_candidates` and `source` it was called with
    /// so the test can assert the pipeline forwarded the right context.
    struct RecordingStage {
        captured: Arc<Mutex<Vec<(Option<HashSet<CellId>>, bool)>>>,
    }

    impl Retriever for RecordingStage {
        fn query(
            &mut self,
            _query_key: &[f32],
            _k: usize,
            _owner_filter: Option<OwnerId>,
        ) -> Vec<RetrievalResult> {
            Vec::new()
        }
        fn insert(&mut self, _cell_id: CellId, _owner: OwnerId, _key: &[f32]) {}
        fn remove(&mut self, _cell_id: CellId) {}
        // Non-zero so the pipeline doesn't skip us via `is_empty`.
        fn len(&self) -> usize {
            1
        }
        fn query_with_source(
            &mut self,
            _query_key: &[f32],
            _k: usize,
            _owner_filter: Option<OwnerId>,
            candidates: Option<&HashSet<CellId>>,
            source: Option<&dyn CellSource>,
        ) -> Vec<RetrievalResult> {
            self.captured.lock().expect("mutex").push((candidates.cloned(), source.is_some()));
            Vec::new()
        }
    }

    struct EmptySource;
    impl CellSource for EmptySource {
        fn get_encoded_key(&self, _id: CellId) -> Option<Vec<f32>> {
            None
        }
    }

    #[test]
    fn forwards_upstream_candidates_and_source_to_downstream_stage() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = RetrieverPipeline::new();
        pipeline.add_stage(Box::new(FixedYieldStage {
            ids: vec![NOMINATED_CELL_ID_A, NOMINATED_CELL_ID_B, NOMINATED_CELL_ID_C],
        }));
        pipeline.add_stage(Box::new(RecordingStage { captured: Arc::clone(&captured) }));

        let source = EmptySource;
        let _ = pipeline.query_with_source(UNUSED_QUERY, PROBE_QUERY_K, None, Some(&source));

        let calls = captured.lock().expect("mutex");
        assert_eq!(calls.len(), 1, "recording stage must be invoked exactly once");
        let (forwarded_candidates, source_seen) = &calls[0];

        let expected: HashSet<CellId> =
            [NOMINATED_CELL_ID_A, NOMINATED_CELL_ID_B, NOMINATED_CELL_ID_C].into_iter().collect();
        assert_eq!(
            forwarded_candidates.as_ref(),
            Some(&expected),
            "downstream stage must receive upstream cell IDs as candidates",
        );
        assert!(*source_seen, "downstream stage must receive the source the pipeline was given");
    }

    #[test]
    fn passes_none_candidates_to_first_stage_when_pipeline_starts_empty() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = RetrieverPipeline::new();
        pipeline.add_stage(Box::new(RecordingStage { captured: Arc::clone(&captured) }));

        let _ = pipeline.query_with_source(UNUSED_QUERY, PROBE_QUERY_K, None, None);

        let calls = captured.lock().expect("mutex");
        assert_eq!(calls.len(), 1);
        let (forwarded_candidates, source_seen) = &calls[0];
        assert!(
            forwarded_candidates.is_none(),
            "first stage must see no upstream candidates (nothing nominated yet)",
        );
        assert!(!*source_seen, "no source passed in => downstream sees None");
    }
}
