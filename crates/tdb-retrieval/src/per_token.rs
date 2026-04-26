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

use std::collections::{HashMap, HashSet};

use tdb_core::{CellId, OwnerId};

use crate::attention::RetrievalResult;
use crate::int8_quant::{Int8Quantizer, QuantizedInt8Vec};
use crate::key_view::{RetrievalKeyView, fixed_dim_key};
use crate::retriever::Retriever;
use crate::simd_distance::DotProduct;

const CANDIDATE_REDUCTION_THRESHOLD: usize = 512;
const MIN_CANDIDATES: usize = 256;
const CANDIDATE_MULTIPLIER: usize = 64;
const TOP5_AVG_MATCH_COUNT: usize = 5;

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

#[derive(Debug)]
struct CellSummary {
    owner: OwnerId,
    pooled_key: Vec<f32>,
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
    /// Fixed-dimension pooled summaries for latent candidate selection.
    cell_summaries: HashMap<CellId, CellSummary>,
    /// Dimension of each token vector (detected on first insert).
    dim: Option<usize>,
    /// How to aggregate per-token scores into a cell score.
    scoring_mode: ScoringMode,
    /// Number of distinct cells fully reranked during the previous query.
    last_scored_cell_count: usize,
}

impl PerTokenRetriever {
    /// Create a new empty per-token retriever with `MaxSim` scoring.
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            cell_token_count: HashMap::new(),
            cell_summaries: HashMap::new(),
            dim: None,
            scoring_mode: ScoringMode::MaxSim,
            last_scored_cell_count: 0,
        }
    }

    /// Create a retriever with a specific scoring mode.
    pub fn with_scoring_mode(mode: ScoringMode) -> Self {
        Self {
            tokens: Vec::new(),
            cell_token_count: HashMap::new(),
            cell_summaries: HashMap::new(),
            dim: None,
            scoring_mode: mode,
            last_scored_cell_count: 0,
        }
    }

    /// Number of individual token entries stored.
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Number of distinct cells indexed.
    pub fn cell_count(&self) -> usize {
        self.cell_token_count.len()
    }

    fn candidate_limit(&self, k: usize) -> usize {
        self.cell_count().min(MIN_CANDIDATES.max(k * CANDIDATE_MULTIPLIER))
    }

    fn candidate_cells(
        &self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Option<HashSet<CellId>> {
        if self.cell_count() <= CANDIDATE_REDUCTION_THRESHOLD {
            return None;
        }

        let query = fixed_dim_key(query_key)?;
        let dim = query.len();
        if dim == 0 {
            return None;
        }
        let inv_sqrt_dk = 1.0 / (dim as f32).sqrt();
        let mut candidates: Vec<(CellId, f32)> = self
            .cell_summaries
            .iter()
            .filter(|(_, summary)| owner_filter.is_none_or(|owner| owner == summary.owner))
            .filter(|(_, summary)| summary.pooled_key.len() == dim)
            .map(|(cell_id, summary)| {
                (*cell_id, DotProduct::f32_dot(&query, &summary.pooled_key) * inv_sqrt_dk)
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
        });
        candidates.truncate(self.candidate_limit(k));

        Some(candidates.into_iter().map(|(cell_id, _)| cell_id).collect())
    }

    fn query_inner(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
        use_candidate_reduction: bool,
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

        let candidate_cells = if use_candidate_reduction && query_view.is_encoded_per_token() {
            self.candidate_cells(query_key, k, owner_filter)
        } else {
            None
        };

        // Collect dot products per cell. Aggregation depends on scoring_mode.
        let mut cell_scores: HashMap<CellId, (Vec<f32>, OwnerId)> = HashMap::new();

        for token_entry in &self.tokens {
            if owner_filter.is_some_and(|o| o != token_entry.owner) {
                continue;
            }

            if candidate_cells.as_ref().is_some_and(|ids| !ids.contains(&token_entry.cell_id)) {
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
        self.last_scored_cell_count = cell_scores.len();

        // Aggregate per-cell scores based on scoring mode.
        let mut results: Vec<RetrievalResult> = cell_scores
            .into_iter()
            .map(|(cell_id, (mut dots, owner))| {
                let score = match self.scoring_mode {
                    ScoringMode::MaxSim => dots.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    ScoringMode::Top5Avg => {
                        dots.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                        let take = dots.len().min(TOP5_AVG_MATCH_COUNT);
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
        self.query_inner(query_key, k, owner_filter, true)
    }

    fn insert(&mut self, cell_id: CellId, owner: OwnerId, key: &[f32]) {
        // Try to decode as per-token encoded.
        let Ok(view) = RetrievalKeyView::parse(key) else {
            return;
        };
        let summary_key = view.pooled_vector();
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

        self.cell_summaries.insert(cell_id, CellSummary { owner, pooled_key: summary_key });
    }

    fn remove(&mut self, cell_id: CellId) {
        self.tokens.retain(|t| t.cell_id != cell_id);
        self.cell_token_count.remove(&cell_id);
        self.cell_summaries.remove(&cell_id);
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_DIM: usize = 128;
    const SMALL_FIXTURE_DIM: usize = 8;
    const OWNER_SPLIT_MODULUS: usize = 2;
    const TOKENS_PER_CELL: usize = 8;
    const BROAD_MATCH_TOKENS: usize = 5;
    const LARGE_CELL_COUNT: usize = 1_000;
    const DISTRACTOR_CELL_COUNT: u64 = 700;
    const OWNER_ONE: OwnerId = 1;
    const OWNER_TWO: OwnerId = 2;
    const TARGET_CELL_ID: usize = 503;
    const OWNER_FILTER_TARGET_CELL_ID: usize = 42;
    const QUERY_K: usize = 5;
    const SPLIT_OWNER_CELL_COUNT: usize = 600;
    const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
    const CELL_SEED_MULTIPLIER: u64 = 0x9E37_79B1_85EB_CA87;
    const TOKEN_SEED_MULTIPLIER: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const RANDOM_SCALE_DENOMINATOR: f32 = (1u64 << 24) as f32;
    const RANDOM_SHIFT: u32 = 40;

    fn normalized_token(dim: usize, cell_id: usize, token_id: usize) -> Vec<f32> {
        let mut state = (cell_id as u64 + 1).wrapping_mul(CELL_SEED_MULTIPLIER)
            ^ (token_id as u64 + 1).wrapping_mul(TOKEN_SEED_MULTIPLIER);
        let mut vector = Vec::with_capacity(dim);
        for _ in 0..dim {
            state = state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(1);
            vector.push(((state >> RANDOM_SHIFT) as f32 / RANDOM_SCALE_DENOMINATOR) * 2.0 - 1.0);
        }
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        for value in &mut vector {
            *value /= norm;
        }
        vector
    }

    fn encoded_cell(dim: usize, cell_id: usize, tokens_per_cell: usize) -> Vec<f32> {
        let tokens: Vec<Vec<f32>> =
            (0..tokens_per_cell).map(|token_id| normalized_token(dim, cell_id, token_id)).collect();
        let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
        encode_per_token_keys(&refs)
    }

    fn populated_top5_retriever(cell_count: usize) -> PerTokenRetriever {
        let mut retriever = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        for cell_id in 0..cell_count {
            let encoded = encoded_cell(FIXTURE_DIM, cell_id, TOKENS_PER_CELL);
            retriever.insert(cell_id as u64, OWNER_ONE, &encoded);
        }
        retriever
    }

    #[test]
    fn test_candidate_reduction_matches_exact_top5avg_on_normalized_fixture() {
        let mut retriever = populated_top5_retriever(LARGE_CELL_COUNT);
        let query = encoded_cell(FIXTURE_DIM, TARGET_CELL_ID, TOKENS_PER_CELL);

        let exact = retriever.query_inner(&query, QUERY_K, Some(OWNER_ONE), false);
        let reduced = retriever.query_inner(&query, QUERY_K, Some(OWNER_ONE), true);

        let exact_ids: Vec<CellId> = exact.iter().map(|result| result.cell_id).collect();
        let reduced_ids: Vec<CellId> = reduced.iter().map(|result| result.cell_id).collect();
        assert_eq!(reduced_ids, exact_ids);
    }

    #[test]
    fn test_candidate_reduction_prefers_broad_match_over_spike() {
        let mut retriever = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        let query_token = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let broad = [0.6f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let spike = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let orthogonal = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        retriever.insert(
            0,
            OWNER_ONE,
            &encode_per_token_keys(&[&broad, &broad, &broad, &broad, &broad]),
        );
        retriever.insert(
            1,
            OWNER_ONE,
            &encode_per_token_keys(&[&spike, &orthogonal, &orthogonal, &orthogonal, &orthogonal]),
        );
        for cell_id in 2..DISTRACTOR_CELL_COUNT {
            retriever.insert(
                cell_id,
                OWNER_ONE,
                &encoded_cell(SMALL_FIXTURE_DIM, cell_id as usize, BROAD_MATCH_TOKENS),
            );
        }

        let query = encode_per_token_keys(&[&query_token]);
        let results = retriever.query_inner(&query, 2, Some(OWNER_ONE), true);

        assert_eq!(results[0].cell_id, 0);
    }

    #[test]
    fn test_candidate_reduction_limits_scored_cells_for_encoded_queries() {
        let mut retriever = populated_top5_retriever(LARGE_CELL_COUNT);
        let query = encoded_cell(FIXTURE_DIM, TARGET_CELL_ID, TOKENS_PER_CELL);

        let _ = retriever.query_inner(&query, QUERY_K, Some(OWNER_ONE), true);

        assert_eq!(
            retriever.last_scored_cell_count,
            MIN_CANDIDATES.max(QUERY_K * CANDIDATE_MULTIPLIER)
        );
        assert!(retriever.last_scored_cell_count < retriever.cell_count());
    }

    #[test]
    fn test_plain_token_query_stays_exact_and_can_find_low_mean_target() {
        let mut retriever = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        let exact = [1.0f32, 0.0];
        let negative = [-1.0f32, 0.0];
        let distractor = [0.4f32, 0.4];

        retriever.insert(
            0,
            OWNER_ONE,
            &encode_per_token_keys(&[
                &exact, &exact, &exact, &exact, &exact, &negative, &negative, &negative, &negative,
                &negative, &negative, &negative,
            ]),
        );
        for cell_id in 1..DISTRACTOR_CELL_COUNT {
            retriever.insert(
                cell_id,
                OWNER_ONE,
                &encode_per_token_keys(&[
                    &distractor,
                    &distractor,
                    &distractor,
                    &distractor,
                    &distractor,
                ]),
            );
        }

        let results = retriever.query_inner(&exact, 1, Some(OWNER_ONE), true);

        assert_eq!(results[0].cell_id, 0);
        assert_eq!(
            retriever.last_scored_cell_count,
            retriever.cell_count(),
            "plain token queries must keep exact full scan behavior"
        );
    }

    #[test]
    fn test_candidate_selector_respects_owner_filter_before_rerank() {
        let mut retriever = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        let target = encoded_cell(FIXTURE_DIM, OWNER_FILTER_TARGET_CELL_ID, TOKENS_PER_CELL);
        for cell_id in 0..SPLIT_OWNER_CELL_COUNT {
            let owner = if cell_id % OWNER_SPLIT_MODULUS == 0 { OWNER_ONE } else { OWNER_TWO };
            let encoded = if cell_id == OWNER_FILTER_TARGET_CELL_ID {
                target.clone()
            } else {
                encoded_cell(FIXTURE_DIM, cell_id, TOKENS_PER_CELL)
            };
            retriever.insert(cell_id as u64, owner, &encoded);
        }

        let results = retriever.query_inner(&target, QUERY_K, Some(OWNER_ONE), true);

        assert!(results.iter().all(|result| result.owner == 1));
        assert_eq!(results[0].cell_id, OWNER_FILTER_TARGET_CELL_ID as CellId);
        assert!(retriever.last_scored_cell_count <= retriever.candidate_limit(QUERY_K));
    }
}
