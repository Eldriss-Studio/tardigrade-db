//! Per-token retriever — Inverted Multi-Key Index pattern.
//!
//! Stores individual token K vectors (INT8 quantized) mapped back to their
//! parent `CellId`. Retrieval uses max-sim aggregation: for each cell, the
//! score is the maximum dot product across all query-token / stored-token
//! pairs. This preserves token-level discriminative signal that mean-pooling
//! destroys.
//!
//! Based on FIER (2025) and max-sim scoring, adapted for
//! the `Retriever` trait.
//!
//! ## Encoding Convention
//!
//! Per-token keys are packed into a flat `&[f32]` slice with a 2-float header:
//!
//! ```text
//! [token_count as f32::from_bits(N)]
//! [dimension as f32::from_bits(D)]
//! [N × D floats: concatenated per-token K vectors]
//! ```

use std::collections::HashMap;

use tdb_core::{CellId, OwnerId};

use crate::attention::RetrievalResult;
use crate::int8_quant::{Int8Quantizer, QuantizedInt8Vec};
use crate::retriever::Retriever;
use crate::simd_distance::DotProduct;

/// A single token's K vector, quantized to INT8, with its parent cell reference.
#[derive(Debug)]
struct TokenEntry {
    cell_id: CellId,
    owner: OwnerId,
    quantized_key: QuantizedInt8Vec,
}

/// Per-token retriever using Inverted Multi-Key Index + max-sim scoring.
///
/// Multiple token entries point back to the same `CellId`. On query,
/// each query token scores against all stored tokens. The best match
/// per cell determines that cell's ranking.
#[derive(Debug)]
pub struct PerTokenRetriever {
    tokens: Vec<TokenEntry>,
    /// Track token count per cell for diagnostics.
    cell_token_count: HashMap<CellId, usize>,
    /// Dimension of each token vector (detected on first insert).
    dim: Option<usize>,
}

impl PerTokenRetriever {
    /// Create a new empty per-token retriever.
    pub fn new() -> Self {
        Self { tokens: Vec::new(), cell_token_count: HashMap::new(), dim: None }
    }

    /// Number of individual token entries stored.
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Number of distinct cells indexed.
    pub fn cell_count(&self) -> usize {
        self.cell_token_count.len()
    }
}

impl Default for PerTokenRetriever {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode multiple per-token K vectors into a flat f32 slice with header.
///
/// Format: `[token_count_bits, dim_bits, k0_0, k0_1, ..., k0_D, k1_0, ...]`
pub fn encode_per_token_keys(token_keys: &[&[f32]]) -> Vec<f32> {
    if token_keys.is_empty() {
        return Vec::new();
    }

    let n = token_keys.len();
    let d = token_keys[0].len();
    let mut encoded = Vec::with_capacity(2 + n * d);

    // Header: token count and dimension as reinterpreted f32 bits.
    encoded.push(f32::from_bits(n as u32));
    encoded.push(f32::from_bits(d as u32));

    // Concatenated token vectors.
    for tk in token_keys {
        encoded.extend_from_slice(tk);
    }

    encoded
}

/// Decode a per-token encoded key slice into token count, dim, and flat data.
///
/// Returns `None` if the slice is too short or doesn't match the encoding convention.
pub fn decode_per_token_keys(encoded: &[f32]) -> Option<(usize, usize, &[f32])> {
    if encoded.len() < 2 {
        return None;
    }

    let n = encoded[0].to_bits() as usize;
    let d = encoded[1].to_bits() as usize;

    if n == 0 || d == 0 {
        return None;
    }

    let data = &encoded[2..];
    if data.len() != n * d {
        return None;
    }

    Some((n, d, data))
}

impl Retriever for PerTokenRetriever {
    fn query(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Vec<RetrievalResult> {
        if self.tokens.is_empty() || query_key.is_empty() {
            return Vec::new();
        }

        // Decode query: could be per-token encoded or a single vector.
        let query_tokens: Vec<QuantizedInt8Vec> =
            if let Some((n, d, data)) = decode_per_token_keys(query_key) {
                // Per-token encoded query: quantize each token.
                (0..n).map(|i| Int8Quantizer::quantize(&data[i * d..(i + 1) * d])).collect()
            } else {
                // Single vector query (e.g., mean-pooled).
                vec![Int8Quantizer::quantize(query_key)]
            };

        let dim = self.dim.unwrap_or(0);
        let inv_sqrt_dk = if dim > 0 { 1.0 / (dim as f32).sqrt() } else { 1.0 };

        // Max-sim: for each cell, take the max score across all (query_token, stored_token) pairs.
        let mut cell_max_scores: HashMap<CellId, (f32, OwnerId)> = HashMap::new();

        for token_entry in &self.tokens {
            if owner_filter.is_some_and(|o| o != token_entry.owner) {
                continue;
            }

            // Skip dimension mismatch.
            if token_entry.quantized_key.values.len() != query_tokens[0].values.len() {
                continue;
            }

            for qt in &query_tokens {
                let dot = DotProduct::int8_dot(qt, &token_entry.quantized_key) * inv_sqrt_dk;

                let entry = cell_max_scores
                    .entry(token_entry.cell_id)
                    .or_insert((f32::NEG_INFINITY, token_entry.owner));

                if dot > entry.0 {
                    *entry = (dot, token_entry.owner);
                }
            }
        }

        // Sort by max score descending, take top-k.
        let mut results: Vec<RetrievalResult> = cell_max_scores
            .into_iter()
            .map(|(cell_id, (score, owner))| RetrievalResult { cell_id, owner, score })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    fn insert(&mut self, cell_id: CellId, owner: OwnerId, key: &[f32]) {
        // Try to decode as per-token encoded.
        if let Some((n, d, data)) = decode_per_token_keys(key) {
            if self.dim.is_none() {
                self.dim = Some(d);
            }

            for i in 0..n {
                let token_key = &data[i * d..(i + 1) * d];
                self.tokens.push(TokenEntry {
                    cell_id,
                    owner,
                    quantized_key: Int8Quantizer::quantize(token_key),
                });
            }

            *self.cell_token_count.entry(cell_id).or_insert(0) += n;
        } else {
            // Fallback: treat as a single-token key (backward compatible).
            if self.dim.is_none() {
                self.dim = Some(key.len());
            }

            self.tokens.push(TokenEntry {
                cell_id,
                owner,
                quantized_key: Int8Quantizer::quantize(key),
            });

            *self.cell_token_count.entry(cell_id).or_insert(0) += 1;
        }
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }
}
