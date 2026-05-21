//! Retrieval-key computation strategies (Strategy pattern).
//!
//! A retrieval key is the vector that identifies a stored pack in latent
//! space. For the vLLM connector path, the scheduler does not have hidden
//! states — only token IDs. The token-embedding table is used as a cheap
//! proxy: look up each token's row, then fold the rows into one vector
//! according to the chosen strategy.
//!
//! Both save and load sides use the same strategy so the resulting keys
//! live in the same space and dot-product retrieval is meaningful.
//!
//! # Strategies
//!
//! - [`LastTokenStrategy`] — return the last in-range token's embedding row.
//! - [`MeanPoolStrategy`]  — mean of all in-range token embedding rows.
//! - [`ProjectedStrategy`] — `W · last_token_embedding` for `hidden != kv_dim`.
//!
//! # Usage
//!
//! ```
//! use tdb_retrieval::retrieval_key::{EmbeddingTable, LastTokenStrategy, RetrievalKeyStrategy};
//!
//! let table = EmbeddingTable::new(vec![0.0, 1.0,  2.0, 3.0,  4.0, 5.0], 3, 2);
//! let key = LastTokenStrategy.compute(&[0, 2], &table).unwrap();
//! assert_eq!(key, vec![4.0, 5.0]);
//! ```

/// Token-embedding table owned by the engine.
///
/// Stored row-major as a flat `Vec<f32>` of length `vocab_size * hidden_size`.
/// Row `i` lives at `weights[i * hidden_size .. (i + 1) * hidden_size]`.
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    weights: Vec<f32>,
    vocab_size: usize,
    hidden_size: usize,
}

impl EmbeddingTable {
    /// Construct from a flat row-major buffer.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != vocab_size * hidden_size`.
    #[must_use]
    pub fn new(weights: Vec<f32>, vocab_size: usize, hidden_size: usize) -> Self {
        assert_eq!(
            weights.len(),
            vocab_size * hidden_size,
            "weights buffer length must equal vocab_size * hidden_size",
        );
        Self { weights, vocab_size, hidden_size }
    }

    #[inline]
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[inline]
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Borrow the embedding row for `token_id`, or `None` if out of range.
    #[inline]
    #[must_use]
    pub fn row(&self, token_id: i64) -> Option<&[f32]> {
        if token_id < 0 {
            return None;
        }
        let tid = token_id as usize;
        if tid >= self.vocab_size {
            return None;
        }
        let start = tid * self.hidden_size;
        Some(&self.weights[start..start + self.hidden_size])
    }
}

/// Strategy interface: fold token IDs into one retrieval-key vector.
pub trait RetrievalKeyStrategy {
    /// Compute the key, or return `None` if the input does not contain at
    /// least one in-range token.
    fn compute(&self, token_ids: &[i64], table: &EmbeddingTable) -> Option<Vec<f32>>;
}

/// Last in-range token's embedding row (most common strategy).
#[derive(Debug, Default, Clone, Copy)]
pub struct LastTokenStrategy;

impl RetrievalKeyStrategy for LastTokenStrategy {
    fn compute(&self, token_ids: &[i64], table: &EmbeddingTable) -> Option<Vec<f32>> {
        let last = token_ids.iter().rev().find_map(|&tid| table.row(tid))?;
        Some(last.to_vec())
    }
}

/// Mean of all in-range token embedding rows.
#[derive(Debug, Default, Clone, Copy)]
pub struct MeanPoolStrategy;

impl RetrievalKeyStrategy for MeanPoolStrategy {
    fn compute(&self, token_ids: &[i64], table: &EmbeddingTable) -> Option<Vec<f32>> {
        let hidden = table.hidden_size();
        let mut acc = vec![0.0f32; hidden];
        let mut count: usize = 0;
        for &tid in token_ids {
            if let Some(row) = table.row(tid) {
                for (a, r) in acc.iter_mut().zip(row.iter()) {
                    *a += *r;
                }
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        let inv = 1.0 / count as f32;
        for a in &mut acc {
            *a *= inv;
        }
        Some(acc)
    }
}

/// Linear projection `W · embedding(last_token)` for `hidden_size` != `kv_dim`.
///
/// `projection` is row-major `kv_dim × hidden_size`; the result vector has
/// length `kv_dim`.
#[derive(Debug, Clone)]
pub struct ProjectedStrategy {
    projection: Vec<f32>,
    kv_dim: usize,
    hidden_size: usize,
}

impl ProjectedStrategy {
    /// Construct from a flat row-major projection matrix.
    ///
    /// # Panics
    ///
    /// Panics if `projection.len() != kv_dim * hidden_size`.
    #[must_use]
    pub fn new(projection: Vec<f32>, kv_dim: usize, hidden_size: usize) -> Self {
        assert_eq!(
            projection.len(),
            kv_dim * hidden_size,
            "projection length must equal kv_dim * hidden_size",
        );
        Self { projection, kv_dim, hidden_size }
    }
}

impl RetrievalKeyStrategy for ProjectedStrategy {
    fn compute(&self, token_ids: &[i64], table: &EmbeddingTable) -> Option<Vec<f32>> {
        assert_eq!(
            self.hidden_size,
            table.hidden_size(),
            "projection hidden_size must match embedding table hidden_size",
        );
        let last = token_ids.iter().rev().find_map(|&tid| table.row(tid))?;
        let mut out = vec![0.0f32; self.kv_dim];
        for (i, out_i) in out.iter_mut().enumerate().take(self.kv_dim) {
            let row = &self.projection[i * self.hidden_size..(i + 1) * self.hidden_size];
            let mut acc = 0.0f32;
            for (w, x) in row.iter().zip(last.iter()) {
                acc += *w * *x;
            }
            *out_i = acc;
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_table() -> EmbeddingTable {
        EmbeddingTable::new(
            vec![
                1.0, 2.0, // tid 0
                3.0, 4.0, // tid 1
                5.0, 6.0, // tid 2
                7.0, 8.0, // tid 3
            ],
            4,
            2,
        )
    }

    #[test]
    fn last_token_returns_last_in_range_row() {
        let key = LastTokenStrategy.compute(&[0, 2, 99, 1], &small_table()).unwrap();
        assert_eq!(key, vec![3.0, 4.0]);
    }

    #[test]
    fn last_token_skips_negative_and_oob_ids() {
        let key = LastTokenStrategy.compute(&[-1, 2, 99], &small_table()).unwrap();
        assert_eq!(key, vec![5.0, 6.0]);
    }

    #[test]
    fn last_token_returns_none_when_all_oob() {
        assert!(LastTokenStrategy.compute(&[99, -1], &small_table()).is_none());
    }

    #[test]
    fn last_token_returns_none_on_empty_input() {
        assert!(LastTokenStrategy.compute(&[], &small_table()).is_none());
    }

    #[test]
    fn mean_pool_averages_in_range_rows() {
        let key = MeanPoolStrategy.compute(&[0, 2, 99], &small_table()).unwrap();
        assert_eq!(key, vec![3.0, 4.0]);
    }

    #[test]
    fn projected_identity_equals_last_token() {
        let identity = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];
        let strategy = ProjectedStrategy::new(identity, 2, 2);
        let key = strategy.compute(&[0, 2], &small_table()).unwrap();
        assert_eq!(key, vec![5.0, 6.0]);
    }

    #[test]
    fn projected_changes_dimension() {
        let projection = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            1.0, 1.0,
        ];
        let strategy = ProjectedStrategy::new(projection, 3, 2);
        let key = strategy.compute(&[2], &small_table()).unwrap();
        assert_eq!(key.len(), 3);
        assert_eq!(key, vec![5.0, 6.0, 11.0]);
    }
}
