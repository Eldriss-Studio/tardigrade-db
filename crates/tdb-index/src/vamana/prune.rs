//! Robust pruning strategy for Vamana graph neighbor selection.
//!
//! Implements the `DiskANN` neighbor selection algorithm: iteratively select the
//! candidate with the highest dot product to the node, then eliminate candidates
//! that are "too similar" to already-selected neighbors (controlled by α).
//!
//! This produces angularly diverse neighbor lists that improve graph navigability
//! compared to simple nearest-neighbor selection.

/// Select up to `max_degree` diverse neighbors from `candidates` for `node_vector`.
///
/// # Algorithm (`DiskANN` robust prune)
///
/// 1. Sort candidates by descending dot product with `node_vector`.
/// 2. For each candidate (best first):
///    - If no already-selected neighbor `s` satisfies `dot(candidate, s) > α · dot(candidate, node)`,
///      select this candidate.
///    - Otherwise, skip it (too similar to an existing selection).
/// 3. Stop when `max_degree` neighbors are selected or candidates are exhausted.
///
/// `α` (typically 1.0–1.5) controls the diversity/quality tradeoff:
/// - α = 1.0: only prune candidates closer to an existing neighbor than to the node itself
/// - α > 1.0: more aggressive pruning, producing more diverse but potentially less precise neighbors
///
/// # Arguments
/// - `node_vector`: the vector of the node being pruned
/// - `candidate_indices`: indices into `all_vectors` of candidate neighbors
/// - `all_vectors`: all vectors in the graph, indexed by candidate index
/// - `alpha`: diversity parameter (typically 1.2)
/// - `max_degree`: maximum number of neighbors to select (R)
pub fn robust_prune(
    node_vector: &[f32],
    candidate_indices: &[usize],
    all_vectors: &[&[f32]],
    alpha: f32,
    max_degree: usize,
) -> Vec<usize> {
    if candidate_indices.is_empty() {
        return Vec::new();
    }

    // Score each candidate by dot product with the node.
    let mut scored: Vec<(usize, f32)> =
        candidate_indices.iter().map(|&idx| (idx, dot(node_vector, all_vectors[idx]))).collect();

    // Sort by descending score (best candidates first).
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<usize> = Vec::with_capacity(max_degree);

    for (candidate_idx, candidate_score) in scored {
        if selected.len() >= max_degree {
            break;
        }

        // Check if this candidate is too similar to any already-selected neighbor.
        let too_similar = selected.iter().any(|&sel_idx| {
            let similarity = dot(all_vectors[candidate_idx], all_vectors[sel_idx]);
            similarity > alpha * candidate_score
        });

        if !too_similar {
            selected.push(candidate_idx);
        }
    }

    selected
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prune_selects_up_to_max_degree() {
        let node = vec![1.0, 0.0];
        let vecs: Vec<Vec<f32>> =
            (0..10).map(|i| vec![(i as f32 * 0.1).cos(), (i as f32 * 0.1).sin()]).collect();
        let all: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let indices: Vec<usize> = (0..10).collect();

        let selected = robust_prune(&node, &indices, &all, 1.2, 4);
        assert!(selected.len() <= 4);
        assert!(!selected.is_empty());
    }

    #[test]
    fn test_prune_empty_candidates() {
        let selected = robust_prune(&[1.0], &[], &[], 1.2, 4);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_prune_favors_diversity() {
        // Node equidistant from two clusters.
        let node = vec![0.5, 0.5];
        // Cluster A: strongly [1,0]-aligned. Cluster B: strongly [0,1]-aligned.
        // Intra-cluster dot products are very high (~1.0), so α=1.0 will prune
        // duplicates within the same cluster.
        let vecs = vec![
            vec![1.0, 0.0],  // cluster A
            vec![0.99, 0.0], // cluster A (very similar to A0)
            vec![0.98, 0.0], // cluster A
            vec![0.0, 1.0],  // cluster B
            vec![0.0, 0.99], // cluster B (very similar to B0)
        ];
        let all: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let indices: Vec<usize> = (0..5).collect();

        // α=1.0: prune if dot(candidate, selected) > dot(candidate, node)
        let selected = robust_prune(&node, &indices, &all, 1.0, 3);

        let from_a = selected.iter().filter(|&&i| i < 3).count();
        let from_b = selected.iter().filter(|&&i| i >= 3).count();
        assert!(from_a >= 1, "Should select at least 1 from cluster A");
        assert!(from_b >= 1, "Should select at least 1 from cluster B");
    }
}
