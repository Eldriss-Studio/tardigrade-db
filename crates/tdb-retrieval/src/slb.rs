//! Semantic Lookaside Buffer — fixed-size LRU cache of recently accessed cells.
//!
//! Stores key vectors in symmetric INT8 quantized format for maximum SIMD throughput.
//! Target: sub-10μs retrieval latency at 4096 entries using NEON (ARM) / auto-vectorized (x86).
//!
//! ## Design: Composite (Dual-Index LRU)
//!
//! Three structures, each handling one concern:
//! - `Vec<SlbSlot>` — contiguous, cache-friendly storage (sequential iteration for dot products)
//! - `HashMap<CellId, usize>` — O(1) lookup by cell ID into the Vec
//! - LRU tracking via `access_order` field + O(1) eviction of the global min
//!
//! The key optimization: `query()` iterates a contiguous `Vec` (cache-line friendly)
//! and uses partial sort (`select_nth_unstable_by`) for O(n + k log k) instead of
//! full sort O(n log n).
//!
//! ## Concurrency
//!
//! [`SemanticLookasideBuffer::query`] takes `&self` so multiple readers can
//! run in parallel under an outer `RwLock::read()` guard. LRU bookkeeping
//! (`access_counter` and per-slot `access_order`) uses [`AtomicU64`] with
//! `Relaxed` ordering.
//!
//! `Relaxed` is sufficient because LRU ordering does not need to synchronize
//! with any other state — it only needs to be approximately monotonic. The
//! happens-before edge that makes writer-side `recompute_lru` see all reader
//! stores comes from the **outer `RwLock`**: when a writer acquires
//! `RwLock::write()` after readers release `RwLock::read()`, the lock
//! implementation provides an acquire-release pair that publishes every
//! reader's `Relaxed` store to the writer. Without that outer lock the
//! Relaxed ordering would not be safe.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use tdb_core::{CellId, OwnerId};

use crate::attention::RetrievalResult;
use crate::int8_quant::{Int8Quantizer, QuantizedInt8Vec};
use crate::simd_distance::DotProduct;

/// A slot in the contiguous storage Vec.
///
/// `access_order` is [`AtomicU64`] so [`SemanticLookasideBuffer::query`]
/// can stamp it through a shared `&SlbSlot` reference. See the
/// module-level Concurrency note for the ordering argument.
#[derive(Debug)]
struct SlbSlot {
    cell_id: CellId,
    owner: OwnerId,
    quantized_key: QuantizedInt8Vec,
    access_order: AtomicU64,
    /// False if this slot has been evicted (tombstone).
    active: bool,
}

/// Semantic Lookaside Buffer: fixed-capacity LRU cache with INT8-quantized keys.
///
/// Composite pattern: dual-index (`Vec` + `HashMap`) for cache-friendly iteration
/// with O(1) lookup and O(1) amortized eviction.
#[derive(Debug)]
pub struct SemanticLookasideBuffer {
    /// Contiguous storage — iterated sequentially during query for cache locality.
    slots: Vec<SlbSlot>,
    /// O(1) lookup: `CellId` → index into `slots`.
    index: HashMap<CellId, usize>,
    /// Index of the LRU slot (lowest `access_order` among active slots).
    lru_slot: Option<usize>,
    capacity: usize,
    dim: usize,
    access_counter: AtomicU64,
    /// Number of active (non-tombstone) entries.
    active_count: usize,
}

impl SemanticLookasideBuffer {
    /// Create a new SLB with the given capacity and vector dimensionality.
    #[must_use]
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
            lru_slot: None,
            capacity,
            dim,
            access_counter: AtomicU64::new(0),
            active_count: 0,
        }
    }

    /// The vector dimensionality this SLB was constructed with.
    ///
    /// Used by `Engine::refresh` (in `tdb-engine`) to detect when the SLB needs to be
    /// rebuilt at a different dimension (e.g., when the engine was opened
    /// empty with the default dim=128 and the first cells written by
    /// another handle have a different key dimension).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Capacity (max number of active entries before LRU eviction).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Insert or update a cell in the SLB.
    /// If at capacity, evicts the least-recently-used entry.
    pub fn insert(&mut self, cell_id: CellId, owner: OwnerId, key: &[f32]) {
        let new_counter = self.access_counter.fetch_add(1, Ordering::Relaxed) + 1;

        // Update existing entry (O(1) via index).
        if let Some(&slot_idx) = self.index.get(&cell_id) {
            let slot = &mut self.slots[slot_idx];
            slot.quantized_key = Int8Quantizer::quantize(key);
            slot.access_order.store(new_counter, Ordering::Relaxed);
            self.refresh_lru_after_touch(slot_idx);
            return;
        }

        // Evict LRU if at capacity.
        if self.active_count >= self.capacity {
            self.evict_lru();
        }

        let quantized_key = Int8Quantizer::quantize(key);
        let slot_idx = self.slots.len();
        self.slots.push(SlbSlot {
            cell_id,
            owner,
            quantized_key,
            access_order: AtomicU64::new(new_counter),
            active: true,
        });
        self.index.insert(cell_id, slot_idx);
        self.active_count += 1;

        // Update LRU tracking.
        self.update_lru_on_insert(slot_idx);
    }

    /// Query for the top-k most relevant cells by INT8 dot product.
    ///
    /// Uses contiguous Vec iteration (cache-friendly) and partial sort
    /// for O(n + k log k) instead of full O(n log n).
    ///
    /// Takes `&self` so multiple readers can score in parallel. LRU
    /// stamping uses [`AtomicU64`] stores with `Relaxed` ordering — see
    /// the module-level Concurrency note for why that is sufficient.
    ///
    /// # Panics
    /// Panics if `query.len() != self.dim`.
    pub fn query(&self, query: &[f32], k: usize) -> Vec<RetrievalResult> {
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

        // Contiguous iteration over Vec (cache-line friendly).
        let mut scored: Vec<(usize, f32)> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.active && s.quantized_key.values.len() == query_q.values.len())
            .map(|(idx, s)| {
                let dot = DotProduct::int8_dot(&query_q, &s.quantized_key);
                (idx, dot * inv_sqrt_dk)
            })
            .collect();

        // Partial sort: O(n + k log k) instead of full O(n log n).
        let take = k.min(scored.len());
        if take > 0 && take < scored.len() {
            scored.select_nth_unstable_by(take - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        scored.truncate(take);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Mark returned cells as recently accessed. `fetch_add` returns a
        // unique-per-call counter even under concurrent readers; each
        // returned slot is stamped with the calling thread's counter.
        let new_counter = self.access_counter.fetch_add(1, Ordering::Relaxed) + 1;
        for &(idx, _) in &scored {
            self.slots[idx].access_order.store(new_counter, Ordering::Relaxed);
        }

        scored
            .iter()
            .map(|&(idx, score)| {
                let slot = &self.slots[idx];
                RetrievalResult { cell_id: slot.cell_id, owner: slot.owner, score }
            })
            .collect()
    }

    /// Check if a cell is present in the SLB.
    #[must_use]
    pub fn contains(&self, cell_id: CellId) -> bool {
        self.index.get(&cell_id).is_some_and(|&idx| self.slots[idx].active)
    }

    /// Current number of active entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.active_count
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.active_count == 0
    }

    /// Evict the LRU entry. O(1) when `lru_slot` is tracked.
    fn evict_lru(&mut self) {
        if let Some(lru_idx) = self.lru_slot {
            let slot = &mut self.slots[lru_idx];
            slot.active = false;
            self.index.remove(&slot.cell_id);
            self.active_count -= 1;

            // Find new LRU (scan active slots — infrequent, only on eviction).
            self.recompute_lru();
        }
    }

    /// After touching a slot (insert or update), check if LRU needs updating.
    fn refresh_lru_after_touch(&mut self, touched_idx: usize) {
        // If the touched slot WAS the LRU, we need a new LRU.
        if self.lru_slot == Some(touched_idx) {
            self.recompute_lru();
        }
    }

    /// Track LRU on new insert.
    ///
    /// Runs under the outer writer lock, so the `Relaxed` loads on
    /// `access_order` see every reader's `Relaxed` stores (the writer
    /// lock's acquire pairs with the readers' release on `RwLock::read`).
    fn update_lru_on_insert(&mut self, new_idx: usize) {
        match self.lru_slot {
            None => self.lru_slot = Some(new_idx),
            Some(current_lru) => {
                let new_order = self.slots[new_idx].access_order.load(Ordering::Relaxed);
                let current_order = self.slots[current_lru].access_order.load(Ordering::Relaxed);
                if new_order < current_order {
                    self.lru_slot = Some(new_idx);
                }
            }
        }
    }

    /// Recompute the LRU slot by scanning active entries.
    /// Called only when the current LRU is evicted or touched — not on every query.
    fn recompute_lru(&mut self) {
        self.lru_slot = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.active)
            .min_by_key(|(_, s)| s.access_order.load(Ordering::Relaxed))
            .map(|(idx, _)| idx);
    }
}

impl crate::retriever::Retriever for SemanticLookasideBuffer {
    fn query(
        &self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        let results = SemanticLookasideBuffer::query(self, query_key, k);
        // SLB doesn't filter internally — apply owner filter post-retrieval.
        match owner_filter {
            Some(owner) => results.into_iter().filter(|r| r.owner == owner).collect(),
            None => results,
        }
    }

    fn insert(&mut self, cell_id: CellId, owner: OwnerId, key: &[f32]) {
        SemanticLookasideBuffer::insert(self, cell_id, owner, key);
    }

    fn remove(&mut self, cell_id: CellId) {
        if let Some(&slot_idx) = self.index.get(&cell_id) {
            if self.slots[slot_idx].active {
                self.slots[slot_idx].active = false;
                self.active_count = self.active_count.saturating_sub(1);
            }
            self.index.remove(&cell_id);
        }
    }

    fn len(&self) -> usize {
        SemanticLookasideBuffer::len(self)
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

    #[test]
    fn test_query_returns_correct_top_k() {
        let mut slb = SemanticLookasideBuffer::new(10, 2);
        slb.insert(0, 1, &[1.0, 0.0]);
        slb.insert(1, 1, &[0.0, 1.0]);
        slb.insert(2, 1, &[0.7, 0.7]);

        let results = slb.query(&[1.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cell_id, 0); // exact match
    }

    #[test]
    fn test_lru_evicts_oldest() {
        let mut slb = SemanticLookasideBuffer::new(3, 2);
        slb.insert(1, 1, &[1.0, 0.0]); // oldest
        slb.insert(2, 1, &[0.0, 1.0]);
        slb.insert(3, 1, &[1.0, 1.0]);

        // Touch cell 1 to make it recent.
        slb.insert(1, 1, &[1.0, 0.0]);

        // Cell 2 is now LRU.
        slb.insert(4, 1, &[0.5, 0.5]); // evicts cell 2
        assert!(slb.contains(1));
        assert!(!slb.contains(2));
        assert!(slb.contains(3));
        assert!(slb.contains(4));
    }

    /// Concurrent queries against a quiescent SLB must return identical
    /// result sets. INT8 dot products are deterministic on the same data,
    /// and the only reader-side mutation (LRU stamping via `AtomicU64`)
    /// does not affect the returned `(cell_id, score)` tuples.
    ///
    /// This catches torn reads of slot state — if the `Relaxed` ordering
    /// were unsafe, two threads could observe different `quantized_key`
    /// or `active` flags and disagree on the result set.
    #[test]
    fn concurrent_queries_against_quiescent_slb_return_identical_result_sets() {
        use std::sync::Arc;
        use std::thread;

        const CELL_COUNT: usize = 64;
        const DIM: usize = 16;
        const READERS: usize = 8;
        const QUERIES_PER_READER: usize = 50;
        const K: usize = 5;

        let mut slb = SemanticLookasideBuffer::new(CELL_COUNT, DIM);
        for cell_id in 0..CELL_COUNT {
            let key: Vec<f32> = (0..DIM).map(|d| ((cell_id + d) as f32 * 0.1).sin()).collect();
            slb.insert(cell_id as u64, 1, &key);
        }
        let slb = Arc::new(slb);

        let query: Vec<f32> = (0..DIM).map(|d| (d as f32 * 0.3).cos()).collect();

        let mut handles = Vec::with_capacity(READERS);
        for _ in 0..READERS {
            let slb = Arc::clone(&slb);
            let query = query.clone();
            handles.push(thread::spawn(move || {
                let mut all = Vec::with_capacity(QUERIES_PER_READER);
                for _ in 0..QUERIES_PER_READER {
                    all.push(slb.query(&query, K));
                }
                all
            }));
        }

        let per_thread: Vec<Vec<Vec<RetrievalResult>>> =
            handles.into_iter().map(|h| h.join().expect("thread")).collect();

        let baseline = &per_thread[0][0];
        for thread_results in &per_thread {
            for results in thread_results {
                assert_eq!(
                    results.len(),
                    baseline.len(),
                    "result-set size diverged under concurrent reads"
                );
                for (got, want) in results.iter().zip(baseline.iter()) {
                    assert_eq!(got.cell_id, want.cell_id, "cell ordering diverged");
                    assert!(
                        (got.score - want.score).abs() < f32::EPSILON,
                        "score diverged: got {} want {}",
                        got.score,
                        want.score
                    );
                }
            }
        }
    }
}
