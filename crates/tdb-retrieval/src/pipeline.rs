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
    pub fn query(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        let mut seen = HashSet::new();
        let mut candidates: Vec<RetrievalResult> = Vec::new();

        for stage in &mut self.stages {
            if stage.is_empty() {
                continue;
            }

            let stage_results = stage.query(query_key, k * self.oversample_factor, owner_filter);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::BruteForceRetriever;

    #[test]
    fn test_pipeline_clear_stages_empties_pipeline() {
        // GIVEN a pipeline with 2 stages containing data
        let mut pipeline = RetrieverPipeline::new();
        pipeline.add_stage(Box::new(BruteForceRetriever::new()));
        pipeline.add_stage(Box::new(BruteForceRetriever::new()));
        pipeline.insert(0, 1, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(pipeline.stage_count(), 2);
        assert!(!pipeline.is_empty());

        // WHEN clear_stages()
        pipeline.clear_stages();

        // THEN stage_count() == 0 AND query returns empty
        assert_eq!(pipeline.stage_count(), 0);
        assert!(pipeline.is_empty());
        let results = pipeline.query(&[1.0, 0.0, 0.0, 0.0], 5, None);
        assert!(results.is_empty());
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
        // Report max across stages (pipeline has data if any stage does).
        self.stages.iter().map(|s| s.len()).max().unwrap_or(0)
    }
}
