//! Per-token retriever — Inverted Multi-Key Index pattern.
//!
//! Stores individual token vectors (INT8 quantized) mapped back to their
//! parent `CellId`. Retrieval uses a pluggable scoring [`ScoringMode`]:
//! max-sim for exact token spikes, or Top-5 average for broader matches across
//! multiple query/stored token pairs. This preserves token-level discriminative
//! signal that mean-pooling destroys.
//!
//! Based on the Inverted Multi-Key Index pattern from latent retrieval systems,
//! adapted for the `Retriever` trait.
//!
//! ## Encoding Convention
//!
//! Per-token keys are packed into a flat `&[f32]` slice with a 64-float
//! Q4-safe header:
//!
//! ```text
//! [0]      sentinel = -1e9
//! [1..31]  zeros
//! [32]     token_count
//! [33]     dimension
//! [34..63] zeros
//! [64..]   N × D floats: concatenated per-token vectors
//! ```

use std::collections::HashMap;

use tdb_core::{CellId, OwnerId};

use crate::attention::RetrievalResult;
use crate::int8_quant::{Int8Quantizer, QuantizedInt8Vec};
use crate::key_view::RetrievalKeyView;
use crate::retriever::Retriever;
use crate::simd_distance::DotProduct;

/// Scoring aggregation strategy for per-token retrieval.
#[derive(Debug, Clone, Copy)]
pub enum ScoringMode {
    /// Max-sim: cell score = max dot product across all token pairs.
    /// Fast but sensitive to high-magnitude outlier tokens.
    MaxSim,
    /// Top-5 average: cell score = mean of the 5 highest dot products.
    /// More robust — rewards broad matches over single spikes.
    Top5Avg,
}

/// A single token's K vector, quantized to INT8, with its parent cell reference.
#[derive(Debug)]
struct TokenEntry {
    cell_id: CellId,
    owner: OwnerId,
    quantized_key: QuantizedInt8Vec,
}

/// Per-token retriever using Inverted Multi-Key Index.
///
/// Multiple token entries point back to the same `CellId`. On query,
/// each query token scores against all stored tokens. Aggregation
/// depends on [`ScoringMode`].
#[derive(Debug)]
pub struct PerTokenRetriever {
    tokens: Vec<TokenEntry>,
    /// Track token count per cell for diagnostics.
    cell_token_count: HashMap<CellId, usize>,
    /// Dimension of each token vector (detected on first insert).
    dim: Option<usize>,
    /// How to aggregate per-token scores into a cell score.
    scoring_mode: ScoringMode,
}

impl PerTokenRetriever {
    /// Create a new empty per-token retriever with `MaxSim` scoring.
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            cell_token_count: HashMap::new(),
            dim: None,
            scoring_mode: ScoringMode::MaxSim,
        }
    }

    /// Create a retriever with a specific scoring mode.
    pub fn with_scoring_mode(mode: ScoringMode) -> Self {
        Self { tokens: Vec::new(), cell_token_count: HashMap::new(), dim: None, scoring_mode: mode }
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

/// Header spans two Q4 groups (64 floats) to keep sentinel and metadata
/// in separate quantization groups. Q4 uses groups of 32 floats; the
/// sentinel (-1e9) would crush metadata values if they shared a group.
///
/// Layout:
///   Group 0 (indices 0-31):  [SENTINEL, 0, 0, ..., 0]  (sentinel dominates)
///   Group 1 (indices 32-63): `[n_tokens, dim, 0, ..., 0]` (metadata preserved)
///   Data (index 64+):        [token vectors]
const HEADER_SIZE: usize = 64;

/// Sentinel value at position 0 marking a per-token encoded key.
const HEADER_SENTINEL: f32 = -1.0e9;

/// Encode multiple per-token K vectors into a flat f32 slice with header.
///
/// The header spans 64 floats (two Q4 groups) so sentinel and metadata
/// are quantized independently.
pub fn encode_per_token_keys(token_keys: &[&[f32]]) -> Vec<f32> {
    if token_keys.is_empty() {
        return Vec::new();
    }

    let n = token_keys.len();
    let d = token_keys[0].len();
    let mut encoded = Vec::with_capacity(HEADER_SIZE + n * d);

    // Group 0: sentinel only (indices 0-31).
    encoded.push(HEADER_SENTINEL);
    encoded.resize(32, 0.0);

    // Group 1: metadata (indices 32-63).
    encoded.push(n as f32);
    encoded.push(d as f32);
    encoded.resize(HEADER_SIZE, 0.0);

    // Data: concatenated token vectors (index 64+).
    for tk in token_keys {
        encoded.extend_from_slice(tk);
    }

    encoded
}

/// Decode a per-token encoded key slice into token count, dim, and flat data.
///
/// Returns `None` if the slice doesn't have the sentinel or sizes don't match.
pub fn decode_per_token_keys(encoded: &[f32]) -> Option<(usize, usize, &[f32])> {
    if encoded.len() < HEADER_SIZE {
        return None;
    }

    // Check sentinel in group 0.
    if encoded[0] > -1.0e8 {
        return None;
    }

    // Metadata in group 1 (indices 32, 33).
    let encoded_n = encoded[32].round() as usize;
    let d = encoded[33].round() as usize;

    if d == 0 {
        return None;
    }

    let data = &encoded[HEADER_SIZE..];
    let n = if encoded_n == 0 {
        if !data.len().is_multiple_of(d) {
            return None;
        }
        data.len() / d
    } else {
        encoded_n
    };

    if n == 0 {
        return None;
    }

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
        let Ok(query_view) = RetrievalKeyView::parse(query_key) else {
            return Vec::new();
        };
        let query_tokens: Vec<QuantizedInt8Vec> =
            if let Some((n, d, data)) = query_view.raw_tokens() {
                // Per-token encoded query: quantize each token.
                (0..n).map(|i| Int8Quantizer::quantize(&data[i * d..(i + 1) * d])).collect()
            } else {
                // Single vector query (e.g., mean-pooled).
                vec![Int8Quantizer::quantize(query_key)]
            };

        let dim = self.dim.unwrap_or(0);
        let inv_sqrt_dk = if dim > 0 { 1.0 / (dim as f32).sqrt() } else { 1.0 };

        // Collect dot products per cell. Aggregation depends on scoring_mode.
        let mut cell_scores: HashMap<CellId, (Vec<f32>, OwnerId)> = HashMap::new();

        for token_entry in &self.tokens {
            if owner_filter.is_some_and(|o| o != token_entry.owner) {
                continue;
            }

            if token_entry.quantized_key.values.len() != query_tokens[0].values.len() {
                continue;
            }

            for qt in &query_tokens {
                let dot = DotProduct::int8_dot(qt, &token_entry.quantized_key) * inv_sqrt_dk;

                let entry = cell_scores
                    .entry(token_entry.cell_id)
                    .or_insert_with(|| (Vec::new(), token_entry.owner));

                entry.0.push(dot);
            }
        }

        // Aggregate per-cell scores based on scoring mode.
        let mut results: Vec<RetrievalResult> = cell_scores
            .into_iter()
            .map(|(cell_id, (mut dots, owner))| {
                let score = match self.scoring_mode {
                    ScoringMode::MaxSim => dots.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    ScoringMode::Top5Avg => {
                        dots.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                        let take = dots.len().min(5);
                        if take == 0 {
                            f32::NEG_INFINITY
                        } else {
                            dots[..take].iter().sum::<f32>() / take as f32
                        }
                    }
                };
                RetrievalResult { cell_id, owner, score }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    fn insert(&mut self, cell_id: CellId, owner: OwnerId, key: &[f32]) {
        // Try to decode as per-token encoded.
        let Ok(view) = RetrievalKeyView::parse(key) else {
            return;
        };
        if let Some((n, d, data)) = view.raw_tokens() {
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
