//! Vamana graph index (DiskANN-style, NOT HNSW).
//!
//! Single-layer graph for approximate nearest neighbor search over key vectors.
//! Uses greedy search with a medoid start node. Suitable for cold-path retrieval
//! at >10K blocks where brute-force becomes too slow.
//!
//! Key parameters:
//! - `R` (max_degree): maximum out-degree per node. Higher = better recall, more memory.
//! - `L` (search_list_size): beam width during greedy search. Higher = better recall, slower.

use std::collections::{HashMap, HashSet};

use tdb_core::CellId;

/// A node in the Vamana graph.
struct VamanaNode {
    id: CellId,
    vector: Vec<f32>,
    /// Neighbor indices into `VamanaIndex::nodes` (not CellIds).
    neighbors: Vec<usize>,
}

/// Result from a Vamana query: (cell_id, dot_product_score).
pub type VamanaResult = (CellId, f32);

/// Vamana graph index for approximate nearest neighbor search.
pub struct VamanaIndex {
    nodes: Vec<VamanaNode>,
    id_to_idx: HashMap<CellId, usize>,
    dim: usize,
    max_degree: usize,
    medoid_idx: Option<usize>,
}

impl VamanaIndex {
    pub fn new(dim: usize, max_degree: usize) -> Self {
        Self {
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            dim,
            max_degree,
            medoid_idx: None,
        }
    }

    /// Add a vector to the index (does not build edges — call `build()` after all inserts).
    ///
    /// # Panics
    /// Panics if `vector.len() != dim` or if `id` has already been inserted.
    pub fn insert(&mut self, id: CellId, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim);
        assert!(
            !self.id_to_idx.contains_key(&id),
            "duplicate CellId {id} in VamanaIndex"
        );
        let idx = self.nodes.len();
        self.nodes.push(VamanaNode {
            id,
            vector: vector.to_vec(),
            neighbors: Vec::new(),
        });
        self.id_to_idx.insert(id, idx);
    }

    /// Build the graph by connecting each node to its R nearest neighbors.
    ///
    /// Must be called after all inserts and before queries. Uses brute-force
    /// neighbor computation (O(n²) — acceptable for <100K nodes in Phase 3;
    /// incremental DiskANN build is a future optimization).
    pub fn build(&mut self) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }

        // Compute medoid (closest to centroid).
        self.update_medoid();

        // For each node, find its R nearest neighbors by brute-force.
        for i in 0..n {
            let mut scored: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, dot(&self.nodes[i].vector, &self.nodes[j].vector)))
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(self.max_degree);

            self.nodes[i].neighbors = scored.into_iter().map(|(j, _)| j).collect();
        }
    }

    /// Query for the top-k nearest neighbors by dot product.
    pub fn query(&self, query: &[f32], k: usize) -> Vec<VamanaResult> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let search_beam = k.max(self.max_degree) * 2;
        let candidates = self.greedy_search(query, search_beam);

        candidates
            .into_iter()
            .take(k)
            .map(|(idx, score)| (self.nodes[idx].id, score))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Greedy beam search (DiskANN-style).
    ///
    /// Maintains a sorted candidate list `L` of size `beam_size`. Iteratively expands
    /// the closest unvisited candidate, exploring all its neighbors. Terminates when
    /// all candidates in L have been visited.
    fn greedy_search(&self, query: &[f32], beam_size: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut candidates: Vec<(usize, f32, bool)> = Vec::new(); // (idx, score, expanded)

        // Seed with multiple entry points for better coverage of disconnected clusters.
        let n = self.nodes.len();
        let num_seeds = (n / 100).clamp(1, 10); // ~1% of nodes, at least 1, at most 10
        let stride = n / num_seeds;
        for s in 0..num_seeds {
            let idx = if s == 0 {
                self.medoid_idx.unwrap_or(0)
            } else {
                s * stride
            };
            if visited.insert(idx) {
                let score = dot(query, &self.nodes[idx].vector);
                candidates.push((idx, score, false));
            }
        }

        loop {
            // Find the best unexpanded candidate.
            let best_unexpanded = candidates
                .iter()
                .enumerate()
                .filter(|(_, (_, _, expanded))| !expanded)
                .max_by(|a, b| {
                    a.1.1
                        .partial_cmp(&b.1.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            let expand_pos = match best_unexpanded {
                Some((pos, _)) => pos,
                None => break, // All candidates expanded — search complete.
            };

            let expand_idx = candidates[expand_pos].0;
            candidates[expand_pos].2 = true; // Mark as expanded.

            // Explore all neighbors of the expanded node.
            for &neighbor_idx in &self.nodes[expand_idx].neighbors {
                if !visited.insert(neighbor_idx) {
                    continue;
                }

                let score = dot(query, &self.nodes[neighbor_idx].vector);
                candidates.push((neighbor_idx, score, false));
            }

            // Keep only the top `beam_size` candidates by score.
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(beam_size);
        }

        candidates
            .into_iter()
            .map(|(idx, score, _)| (idx, score))
            .collect()
    }

    /// Update medoid to the node closest to the centroid of all vectors.
    fn update_medoid(&mut self) {
        if self.nodes.is_empty() {
            return;
        }

        // Compute centroid.
        let n = self.nodes.len() as f32;
        let mut centroid = vec![0.0f32; self.dim];
        for node in &self.nodes {
            for (c, v) in centroid.iter_mut().zip(node.vector.iter()) {
                *c += v / n;
            }
        }

        // Find node with highest dot product to centroid.
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (idx, node) in self.nodes.iter().enumerate() {
            let score = dot(&centroid, &node.vector);
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        self.medoid_idx = Some(best_idx);
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_index() {
        let index = VamanaIndex::new(4, 8);
        assert!(index.query(&[1.0, 0.0, 0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn test_single_insert_and_query() {
        let mut index = VamanaIndex::new(4, 8);
        index.insert(42, &[1.0, 0.0, 0.0, 0.0]);
        index.build();

        let results = index.query(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
    }

    #[test]
    fn test_nearest_neighbor_found() {
        let mut index = VamanaIndex::new(4, 8);
        index.insert(0, &[1.0, 0.0, 0.0, 0.0]);
        index.insert(1, &[0.0, 1.0, 0.0, 0.0]);
        index.insert(2, &[0.9, 0.1, 0.0, 0.0]);
        index.build();

        let results = index.query(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_max_degree_enforced() {
        let mut index = VamanaIndex::new(2, 3);
        for i in 0..20u64 {
            index.insert(i, &[(i as f32).sin(), (i as f32).cos()]);
        }
        index.build();

        for node in &index.nodes {
            assert!(
                node.neighbors.len() <= 3,
                "Node {} has {} neighbors (max 3)",
                node.id,
                node.neighbors.len()
            );
        }
    }
}
