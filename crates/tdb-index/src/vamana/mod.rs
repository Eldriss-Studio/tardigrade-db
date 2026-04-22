//! Vamana graph index (`DiskANN`-style, NOT HNSW).
//!
//! Single-layer graph for approximate nearest neighbor search over key vectors.
//! Uses greedy search with a medoid start node. Suitable for cold-path retrieval
//! at >10K blocks where brute-force becomes too slow.
//!
//! Key parameters:
//! - `R` (`max_degree`): maximum out-degree per node. Higher = better recall, more memory.
//! - `L` (`search_list_size`): beam width during greedy search. Higher = better recall, slower.

pub mod prune;

use std::collections::{HashMap, HashSet};

use tdb_core::CellId;

use self::prune::robust_prune;

/// A node in the Vamana graph.
#[derive(Debug)]
struct VamanaNode {
    id: CellId,
    vector: Vec<f32>,
    /// Neighbor indices into `VamanaIndex::nodes` (not `CellIds`).
    neighbors: Vec<usize>,
}

/// Result from a Vamana query: (`cell_id`, `dot_product_score`).
pub type VamanaResult = (CellId, f32);

/// Vamana graph index for approximate nearest neighbor search.
pub struct VamanaIndex {
    nodes: Vec<VamanaNode>,
    id_to_idx: HashMap<CellId, usize>,
    dim: usize,
    max_degree: usize,
    medoid_idx: Option<usize>,
    /// Diversity parameter for robust pruning (`DiskANN` α, typically 1.2).
    alpha: f32,
}

impl VamanaIndex {
    pub fn new(dim: usize, max_degree: usize) -> Self {
        Self {
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            dim,
            max_degree,
            medoid_idx: None,
            alpha: 1.2,
        }
    }

    /// Add a vector to the index without building edges.
    /// Call `build()` after all inserts for batch construction.
    ///
    /// # Panics
    /// Panics if `vector.len() != dim` or if `id` has already been inserted.
    pub fn insert(&mut self, id: CellId, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim);
        assert!(!self.id_to_idx.contains_key(&id), "duplicate CellId {id} in VamanaIndex");
        let idx = self.nodes.len();
        self.nodes.push(VamanaNode { id, vector: vector.to_vec(), neighbors: Vec::new() });
        self.id_to_idx.insert(id, idx);
    }

    /// Insert a vector and immediately wire it into the graph (incremental build).
    ///
    /// Uses greedy search to find candidate neighbors, then robust prune (Strategy pattern)
    /// to select diverse neighbors. Bidirectional edges are added, and overfull neighbors
    /// are pruned.
    ///
    /// # Panics
    /// Panics if `vector.len() != dim` or if `id` has already been inserted.
    pub fn insert_online(&mut self, id: CellId, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim);
        assert!(!self.id_to_idx.contains_key(&id), "duplicate CellId {id} in VamanaIndex");

        let new_idx = self.nodes.len();
        self.nodes.push(VamanaNode { id, vector: vector.to_vec(), neighbors: Vec::new() });
        self.id_to_idx.insert(id, new_idx);

        if self.nodes.len() == 1 {
            self.medoid_idx = Some(0);
            return;
        }

        // Greedy search to find candidate neighbors.
        let search_beam = self.max_degree * 3;
        let candidates = self.greedy_search(vector, search_beam);
        let candidate_indices: Vec<usize> = candidates.iter().map(|&(idx, _)| idx).collect();

        // Robust prune to select diverse neighbors (Strategy pattern).
        let all_vecs: Vec<&[f32]> = self.nodes.iter().map(|n| n.vector.as_slice()).collect();
        let selected =
            robust_prune(vector, &candidate_indices, &all_vecs, self.alpha, self.max_degree);

        // Wire bidirectional edges.
        for &neighbor_idx in &selected {
            self.add_edge(new_idx, neighbor_idx);
            self.add_edge(neighbor_idx, new_idx);
        }

        // Prune overfull neighbors.
        for &neighbor_idx in &selected {
            self.prune_node_if_needed(neighbor_idx);
        }

        // Periodically update medoid for better search entry point.
        if self.nodes.len() % 500 == 0 {
            self.update_medoid();
        }
    }

    /// Build the graph by connecting each node to its R nearest neighbors.
    ///
    /// Template Method: reimplemented as batch `insert_online` when graph is empty,
    /// or O(n²) brute-force for initial builds. For already-wired graphs, this is a no-op.
    pub fn build(&mut self) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }

        // If the graph already has edges (from insert_online), skip.
        let has_edges = self.nodes.iter().any(|n| !n.neighbors.is_empty());
        if has_edges {
            self.update_medoid();
            return;
        }

        self.update_medoid();

        // O(n²) brute-force for initial batch build.
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

        candidates.into_iter().take(k).map(|(idx, score)| (self.nodes[idx].id, score)).collect()
    }

    /// Number of neighbors for a given cell ID.
    pub fn neighbor_count(&self, id: CellId) -> usize {
        self.id_to_idx.get(&id).map_or(0, |&idx| self.nodes[idx].neighbors.len())
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Greedy beam search (DiskANN-style).
    fn greedy_search(&self, query: &[f32], beam_size: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut candidates: Vec<(usize, f32, bool)> = Vec::new();

        // Seed with multiple entry points for better coverage.
        let n = self.nodes.len();
        let num_seeds = (n / 100).clamp(1, 10);
        let stride = n / num_seeds;
        for s in 0..num_seeds {
            let idx = if s == 0 { self.medoid_idx.unwrap_or(0) } else { s * stride };
            if visited.insert(idx) {
                let score = dot(query, &self.nodes[idx].vector);
                candidates.push((idx, score, false));
            }
        }

        loop {
            let best_unexpanded = candidates
                .iter()
                .enumerate()
                .filter(|(_, (_, _, expanded))| !expanded)
                .max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap_or(std::cmp::Ordering::Equal));

            let Some((expand_pos, _)) = best_unexpanded else {
                break;
            };

            let expand_idx = candidates[expand_pos].0;
            candidates[expand_pos].2 = true;

            for &neighbor_idx in &self.nodes[expand_idx].neighbors {
                if !visited.insert(neighbor_idx) {
                    continue;
                }

                let score = dot(query, &self.nodes[neighbor_idx].vector);
                candidates.push((neighbor_idx, score, false));
            }

            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(beam_size);
        }

        candidates.into_iter().map(|(idx, score, _)| (idx, score)).collect()
    }

    fn update_medoid(&mut self) {
        if self.nodes.is_empty() {
            return;
        }

        let n = self.nodes.len() as f32;
        let mut centroid = vec![0.0f32; self.dim];
        for node in &self.nodes {
            for (c, v) in centroid.iter_mut().zip(node.vector.iter()) {
                *c += v / n;
            }
        }

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

    fn add_edge(&mut self, from: usize, to: usize) {
        if !self.nodes[from].neighbors.contains(&to) {
            self.nodes[from].neighbors.push(to);
        }
    }

    /// Prune a node's neighbors if it exceeds `max_degree` using robust pruning.
    fn prune_node_if_needed(&mut self, idx: usize) {
        if self.nodes[idx].neighbors.len() <= self.max_degree {
            return;
        }

        let node_vec = self.nodes[idx].vector.clone();
        let candidate_indices: Vec<usize> = self.nodes[idx].neighbors.clone();
        let all_vecs: Vec<&[f32]> = self.nodes.iter().map(|n| n.vector.as_slice()).collect();

        let selected =
            robust_prune(&node_vec, &candidate_indices, &all_vecs, self.alpha, self.max_degree);

        self.nodes[idx].neighbors = selected;
    }
}

impl std::fmt::Debug for VamanaIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VamanaIndex")
            .field("nodes", &self.nodes.len())
            .field("dim", &self.dim)
            .field("max_degree", &self.max_degree)
            .finish_non_exhaustive()
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

    #[test]
    fn test_insert_online_basic() {
        let mut index = VamanaIndex::new(4, 8);
        index.insert_online(0, &[1.0, 0.0, 0.0, 0.0]);
        index.insert_online(1, &[0.0, 1.0, 0.0, 0.0]);
        index.insert_online(2, &[0.9, 0.1, 0.0, 0.0]);

        let results = index.query(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(results[0].0, 0);
    }
}
