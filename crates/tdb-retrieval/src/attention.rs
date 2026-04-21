//! Brute-force attention scoring over stored key vectors.
//!
//! Computes `score = q · k / √d_k` for each stored key. At per-agent scale (<10K blocks),
//! SIMD brute-force matmul is faster than any ANN index (validated by `MemArt` paper).

use tdb_core::{CellId, OwnerId};

use crate::simd_distance::DotProduct;

/// A single retrieval result: cell ID, owner, and attention score.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub cell_id: CellId,
    pub owner: OwnerId,
    pub score: f32,
}

/// Entry in the retriever's key store.
#[derive(Debug)]
struct StoredKey {
    cell_id: CellId,
    owner: OwnerId,
    key: Vec<f32>,
}

/// Brute-force retriever over in-memory key vectors.
///
/// Stores key vectors and computes attention scores via exhaustive dot product.
/// Suitable for per-agent partitions with <10K blocks.
#[derive(Debug)]
pub struct BruteForceRetriever {
    entries: Vec<StoredKey>,
}

impl BruteForceRetriever {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Insert a key vector for a cell.
    pub fn insert(&mut self, cell_id: CellId, owner: OwnerId, _layer: u16, key: &[f32]) {
        self.entries.push(StoredKey { cell_id, owner, key: key.to_vec() });
    }

    /// Query for the top-k most relevant cells by attention score.
    ///
    /// Score = `q · k / √d_k` (scaled dot product, as in transformer attention).
    /// If `owner_filter` is `Some(id)`, only cells belonging to that owner are considered.
    pub fn query(
        &self,
        query: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        if query.is_empty() {
            return Vec::new();
        }
        let d_k = query.len() as f32;
        let inv_sqrt_dk = 1.0 / d_k.sqrt();

        let mut scores: Vec<RetrievalResult> = self
            .entries
            .iter()
            .filter(|e| match owner_filter {
                Some(owner) => e.owner == owner,
                None => true,
            })
            .filter(|e| e.key.len() == query.len())
            .map(|e| {
                let dot = DotProduct::f32_dot(query, &e.key);
                RetrievalResult { cell_id: e.cell_id, owner: e.owner, score: dot * inv_sqrt_dk }
            })
            .collect();

        // Sort by score descending, take top-k.
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Number of entries in the retriever.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for BruteForceRetriever {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_retriever() {
        let retriever = BruteForceRetriever::new();
        let results = retriever.query(&[1.0, 0.0], 5, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_exact_match_ranks_first() {
        let mut retriever = BruteForceRetriever::new();
        retriever.insert(0, 1, 0, &[1.0, 0.0]);
        retriever.insert(1, 1, 0, &[0.0, 1.0]);
        retriever.insert(2, 1, 0, &[0.7, 0.7]);

        let results = retriever.query(&[1.0, 0.0], 3, None);
        assert_eq!(results[0].cell_id, 0); // exact match should be first
    }
}
