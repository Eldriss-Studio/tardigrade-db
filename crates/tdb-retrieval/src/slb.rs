//! Semantic Lookaside Buffer — fixed-size LRU cache of recently accessed cells.
//!
//! Stores key vectors in symmetric INT8 quantized format for maximum SIMD throughput.
//! Target: sub-5μs retrieval latency using NEON (ARM) / auto-vectorized (x86) dot products.
//!
//! Design: Proxy pattern — sits in front of the block pool, intercepting queries.

use std::collections::HashMap;

use tdb_core::{CellId, OwnerId};

use crate::attention::RetrievalResult;
use crate::int8_quant::{Int8Quantizer, QuantizedInt8Vec};
use crate::simd_distance::DotProduct;

/// Entry stored in the SLB.
#[derive(Debug)]
struct SlbEntry {
    cell_id: CellId,
    owner: OwnerId,
    quantized_key: QuantizedInt8Vec,
    /// Tracks access order. Higher = more recent.
    access_order: u64,
}

/// Semantic Lookaside Buffer: fixed-capacity LRU cache with INT8-quantized keys.
#[derive(Debug)]
pub struct SemanticLookasideBuffer {
    entries: HashMap<CellId, SlbEntry>,
    capacity: usize,
    dim: usize,
    access_counter: u64,
}

impl SemanticLookasideBuffer {
    /// Create a new SLB with the given capacity and vector dimensionality.
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self { entries: HashMap::with_capacity(capacity), capacity, dim, access_counter: 0 }
    }

    /// Insert or update a cell in the SLB.
    /// If at capacity, evicts the least-recently-used entry.
    pub fn insert(&mut self, cell_id: CellId, owner: OwnerId, key: &[f32]) {
        self.access_counter += 1;

        if self.entries.contains_key(&cell_id) {
            // Update existing entry.
            let entry = self.entries.get_mut(&cell_id).unwrap();
            entry.quantized_key = Int8Quantizer::quantize(key);
            entry.access_order = self.access_counter;
            return;
        }

        // Evict LRU if at capacity.
        if self.entries.len() >= self.capacity {
            self.evict_lru();
        }

        self.entries.insert(
            cell_id,
            SlbEntry {
                cell_id,
                owner,
                quantized_key: Int8Quantizer::quantize(key),
                access_order: self.access_counter,
            },
        );
    }

    /// Query for the top-k most relevant cells by INT8 dot product.
    ///
    /// # Panics
    /// Panics if `query.len() != self.dim`.
    pub fn query(&mut self, query: &[f32], k: usize) -> Vec<RetrievalResult> {
        assert_eq!(
            query.len(),
            self.dim,
            "query dimension {} does not match SLB dimension {}",
            query.len(),
            self.dim
        );
        let query_q = Int8Quantizer::quantize(query);
        let d_k = self.dim as f32;
        let inv_sqrt_dk = 1.0 / d_k.sqrt();

        let mut results: Vec<RetrievalResult> = self
            .entries
            .values()
            .filter(|e| e.quantized_key.values.len() == query_q.values.len())
            .map(|e| {
                let dot = DotProduct::int8_dot(&query_q, &e.quantized_key);
                RetrievalResult { cell_id: e.cell_id, owner: e.owner, score: dot * inv_sqrt_dk }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        // Mark returned cells as recently accessed.
        self.access_counter += 1;
        for r in &results {
            if let Some(entry) = self.entries.get_mut(&r.cell_id) {
                entry.access_order = self.access_counter;
            }
        }

        results
    }

    /// Check if a cell is present in the SLB.
    pub fn contains(&self, cell_id: CellId) -> bool {
        self.entries.contains_key(&cell_id)
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Evict the least-recently-used entry.
    ///
    /// NOTE: O(n) scan over `HashMap`. Acceptable for <10K entries.
    /// If SLB capacity targets grow beyond 10K, replace with a dual-map
    /// (`BTreeMap`<`access_order`, `CellId`>) or linked-list LRU for O(1) eviction.
    fn evict_lru(&mut self) {
        if let Some((&lru_id, _)) = self.entries.iter().min_by_key(|(_, entry)| entry.access_order)
        {
            self.entries.remove(&lru_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_contains() {
        let mut slb = SemanticLookasideBuffer::new(10, 4);
        slb.insert(1, 1, &[1.0, 2.0, 3.0, 4.0]);
        assert!(slb.contains(1));
        assert!(!slb.contains(2));
    }

    #[test]
    fn test_capacity_enforcement() {
        let mut slb = SemanticLookasideBuffer::new(3, 2);
        slb.insert(1, 1, &[1.0, 0.0]);
        slb.insert(2, 1, &[0.0, 1.0]);
        slb.insert(3, 1, &[1.0, 1.0]);
        assert_eq!(slb.len(), 3);

        slb.insert(4, 1, &[0.5, 0.5]); // should evict cell 1 (LRU)
        assert_eq!(slb.len(), 3);
        assert!(!slb.contains(1));
        assert!(slb.contains(4));
    }
}
