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
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use lru::LruCache;
use tdb_core::{CellId, OwnerId};

use crate::attention::RetrievalResult;
use crate::cell_source::CellSource;
use crate::int8_quant::{Int8Quantizer, QuantizedInt8Vec};
use crate::key_view::{RetrievalKeyView, fixed_dim_key};
use crate::quantization::{Int8Group32, QuantizedToken, RetrievalQuantStrategy};
use crate::retriever::Retriever;
use crate::simd_distance::DotProduct;

/// Default capacity of the decoded-token cache, in cells. Chosen so that
/// the cache fits roughly 250 MB at Qwen3-0.6B-shaped cells
/// (~512 cells × 127 tokens × 1024 dims × 4 bytes ≈ 260 MB) — enough to
/// hold a working set of recently-scored candidates without bounding the
/// retriever's memory by corpus size.
pub const DEFAULT_TOKEN_CACHE_CAPACITY: NonZeroUsize = match NonZeroUsize::new(512) {
    Some(n) => n,
    None => unreachable!(),
};

/// Default capacity of the INT8 retrieval tier, in cells.
///
/// At Qwen3-0.6B shape (~127 tokens × 1024 dims) each INT8-quantized
/// cell occupies ~150 KB. The default of 32K cells caps the tier at
/// ~5 GB — comfortably below typical RAM budgets and large enough
/// that LoCoMo-scale corpora keep their full candidate set in the
/// hot tier through query phase. Cold-cell misses fall through to
/// the Q4 archival source (degraded precision); the LRU's job is to
/// keep the working set hot.
pub const DEFAULT_INT8_TIER_CAPACITY: NonZeroUsize = match NonZeroUsize::new(32_768) {
    Some(n) => n,
    None => unreachable!(),
};

/// Parameter Object: tuneable thresholds for per-token retrieval.
///
/// Controls the recall-vs-latency tradeoff in candidate reduction.
/// Defaults match the values validated in experiments (100% recall at 100 memories
/// with `Top5Avg` scoring on Qwen3-0.6B).
#[derive(Debug, Clone)]
pub struct PerTokenConfig {
    /// Cell count above which candidate reduction activates.
    pub candidate_reduction_threshold: usize,
    /// Floor on candidate count after reduction.
    pub min_candidates: usize,
    /// Multiplier applied to k when computing candidate limit.
    pub candidate_multiplier: usize,
    /// Number of top dot products averaged in `Top5Avg` scoring.
    pub top_n_avg_match_count: usize,
}

impl Default for PerTokenConfig {
    fn default() -> Self {
        Self {
            candidate_reduction_threshold: 512,
            min_candidates: 256,
            candidate_multiplier: 64,
            top_n_avg_match_count: 5,
        }
    }
}

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

/// Contiguous token store — Structure of Arrays (`SoA`) layout.
///
/// All token vectors are packed sequentially in one `Vec<i8>` arena.
/// Eliminates the pointer-chasing cache misses of `Vec<TokenEntry>`
/// where each `QuantizedInt8Vec.values` was a separate heap allocation.
#[derive(Debug)]
struct TokenStore {
    data: Vec<i8>,
    scales: Vec<f32>,
    cell_ids: Vec<CellId>,
    owners: Vec<OwnerId>,
    dim: usize,
}

impl TokenStore {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            scales: Vec::new(),
            cell_ids: Vec::new(),
            owners: Vec::new(),
            dim: 0,
        }
    }

    fn len(&self) -> usize {
        self.scales.len()
    }

    fn push(&mut self, cell_id: CellId, owner: OwnerId, quantized: &QuantizedInt8Vec) {
        if self.dim == 0 {
            self.dim = quantized.values.len();
        }
        self.data.extend_from_slice(&quantized.values);
        self.scales.push(quantized.scale);
        self.cell_ids.push(cell_id);
        self.owners.push(owner);
    }

    fn token_data(&self, index: usize) -> &[i8] {
        let start = index * self.dim;
        &self.data[start..start + self.dim]
    }

    fn retain_by_cell(&mut self, keep: impl Fn(CellId) -> bool) {
        let dim = self.dim;
        if dim == 0 {
            return;
        }
        let mut write = 0;
        for read in 0..self.len() {
            if keep(self.cell_ids[read]) {
                if write != read {
                    self.data.copy_within(read * dim..(read + 1) * dim, write * dim);
                    self.scales[write] = self.scales[read];
                    self.cell_ids[write] = self.cell_ids[read];
                    self.owners[write] = self.owners[read];
                }
                write += 1;
            }
        }
        self.data.truncate(write * dim);
        self.scales.truncate(write);
        self.cell_ids.truncate(write);
        self.owners.truncate(write);
    }
}

#[derive(Debug)]
struct CellSummary {
    owner: OwnerId,
    pooled_key: Vec<f32>,
}

/// Per-token retriever using Inverted Multi-Key Index with `SoA` layout.
///
/// Multiple token entries point back to the same `CellId`. On query,
/// each query token scores against all stored tokens. Aggregation
/// depends on [`ScoringMode`]. Token vectors are stored contiguously
/// in a flat arena for cache-friendly sequential access.
#[derive(Debug)]
pub struct PerTokenRetriever {
    store: TokenStore,
    /// Track token count per cell for diagnostics.
    cell_token_count: HashMap<CellId, usize>,
    /// Fixed-dimension pooled summaries for latent candidate selection.
    cell_summaries: HashMap<CellId, CellSummary>,
    /// Dimension of each token vector (detected on first insert).
    dim: Option<usize>,
    /// How to aggregate per-token scores into a cell score.
    scoring_mode: ScoringMode,
    /// Tuneable retrieval thresholds (Parameter Object pattern).
    config: PerTokenConfig,
    /// Number of distinct cells fully reranked during the previous query.
    last_scored_cell_count: usize,
    /// Running sum of stored f32 token vectors, per dimension. Used by
    /// mean-centered refinement to subtract the corpus's shared high-energy
    /// direction. Updated on insert; not decremented on remove (the bias is
    /// small in practice and is fully refreshed when the engine rebuilds the
    /// retriever by replaying inserts from durable storage).
    corpus_sum: Vec<f32>,
    /// Number of f32 token vectors that have contributed to `corpus_sum`.
    corpus_token_count: usize,
    /// Running sum of outer products (`x_j * x_k`) across all tokens, stored
    /// as a flat `dim*dim` matrix in row-major order. Used to compute corpus
    /// covariance for ZCA whitening.
    corpus_sq_sum: Vec<f32>,
    /// Cached ZCA whitening matrix (dim*dim, row-major). Invalidated when
    /// `corpus_token_count` changes.
    whitening_cache: Option<Vec<f32>>,
    /// Token count at which `whitening_cache` was last computed.
    whitening_cache_count: usize,
    /// Bounded cache of decoded f32 token matrices, keyed by `CellId`.
    /// Used by `query_with_source` to amortize Q4-decode cost over hot
    /// candidates. Cache-Aside semantics — the source remains the source
    /// of truth; the cache only fills as queries decode cells.
    token_cache: Mutex<LruCache<CellId, Arc<Vec<f32>>>>,
    /// In-memory INT8-quantized retrieval tier — the *scoring* tier.
    ///
    /// Bounded LRU of `Vec<QuantizedToken>` per cell at INT8 precision
    /// (~150 KB per Qwen-0.6B-shaped cell). Read by the lazy
    /// `query_with_source` path *before* falling back to the Q4-archival
    /// source. Populated at [`Retriever::insert`] time when lazy mode is
    /// active; LRU eviction kicks in once the cache reaches its
    /// configured capacity.
    ///
    /// **Cold-cell behavior:** when a cell is evicted (or never seen,
    /// e.g. after engine restart before the tier is re-populated), the
    /// lazy retriever falls back to reading the Q4 archival source.
    /// Q4 precision is coarser; recall on cold cells degrades. The LRU
    /// capacity should be sized to comfortably hold the active working
    /// set so this fallback is rare in steady state.
    int8_tokens: Mutex<LruCache<CellId, Arc<Vec<QuantizedToken>>>>,
    /// Whether `insert` populates the in-memory `TokenStore`.
    ///
    /// `true` (default): legacy eager mode — `query` and
    /// `dequantized_tokens_for_cell` work against the in-memory store.
    /// Memory cost is ~130 KB per cell at Qwen3-0.6B-shaped tokens.
    ///
    /// `false`: lazy mode — only metadata + `cell_summaries` are kept
    /// per cell. `query` returns empty (callers must use
    /// `query_with_source` with a [`CellSource`]). Memory cost is
    /// ~4 KB per cell.
    ///
    /// The engine constructs the retriever in lazy mode; tests that
    /// need the eager API construct it in default mode.
    populate_token_store: bool,
}

impl PerTokenRetriever {
    /// Create a new empty per-token retriever with `MaxSim` scoring and default config.
    pub fn new() -> Self {
        Self::with_config(ScoringMode::MaxSim, PerTokenConfig::default())
    }

    /// Create a retriever with a specific scoring mode and default config.
    pub fn with_scoring_mode(mode: ScoringMode) -> Self {
        Self::with_config(mode, PerTokenConfig::default())
    }

    /// Create a retriever with explicit scoring mode and config (Parameter Object).
    pub fn with_config(mode: ScoringMode, config: PerTokenConfig) -> Self {
        Self::with_config_and_cache_capacity(mode, config, DEFAULT_TOKEN_CACHE_CAPACITY)
    }

    /// Create a retriever with explicit scoring mode, config, and decoded-
    /// token cache capacity (in cells). Defaults to eager mode (in-memory
    /// `TokenStore` populated on insert).
    pub fn with_config_and_cache_capacity(
        mode: ScoringMode,
        config: PerTokenConfig,
        token_cache_capacity: NonZeroUsize,
    ) -> Self {
        Self::build(
            mode,
            config,
            token_cache_capacity,
            DEFAULT_INT8_TIER_CAPACITY,
            /*populate_token_store=*/ true,
        )
    }

    /// Create a retriever in **lazy mode**: `insert` updates only metadata
    /// and `cell_summaries`; the in-memory `TokenStore` is not populated.
    /// Used by the engine — callers must supply a [`CellSource`] at query
    /// time via [`Self::query_with_source`].
    pub fn lazy_with_config_and_cache_capacity(
        mode: ScoringMode,
        config: PerTokenConfig,
        token_cache_capacity: NonZeroUsize,
    ) -> Self {
        Self::build(
            mode,
            config,
            token_cache_capacity,
            DEFAULT_INT8_TIER_CAPACITY,
            /*populate_token_store=*/ false,
        )
    }

    /// Lazy-mode constructor with an explicit INT8 tier capacity.
    ///
    /// Use this to tune the precision/memory trade-off: a larger
    /// capacity keeps more cells at INT8 precision at the cost of
    /// RAM; a smaller capacity caps RAM at the cost of cold-cell
    /// recall (falls back to the Q4 archival source).
    pub fn lazy_with_int8_tier_capacity(
        mode: ScoringMode,
        config: PerTokenConfig,
        token_cache_capacity: NonZeroUsize,
        int8_tier_capacity: NonZeroUsize,
    ) -> Self {
        Self::build(
            mode,
            config,
            token_cache_capacity,
            int8_tier_capacity,
            /*populate_token_store=*/ false,
        )
    }

    fn build(
        mode: ScoringMode,
        config: PerTokenConfig,
        token_cache_capacity: NonZeroUsize,
        int8_tier_capacity: NonZeroUsize,
        populate_token_store: bool,
    ) -> Self {
        let cache = LruCache::new(token_cache_capacity);
        let int8_tier_cache: LruCache<CellId, Arc<Vec<QuantizedToken>>> =
            LruCache::new(int8_tier_capacity);
        Self {
            store: TokenStore::new(),
            cell_token_count: HashMap::new(),
            cell_summaries: HashMap::new(),
            dim: None,
            scoring_mode: mode,
            config,
            last_scored_cell_count: 0,
            corpus_sum: Vec::new(),
            corpus_token_count: 0,
            corpus_sq_sum: Vec::new(),
            whitening_cache: None,
            whitening_cache_count: 0,
            token_cache: Mutex::new(cache),
            int8_tokens: Mutex::new(int8_tier_cache),
            populate_token_store,
        }
    }

    /// Iterate stored f32-dequantized tokens for a single cell, paired with owner.
    ///
    /// Yields `(token_f32, owner)` for each token of `cell_id`. Returns an empty
    /// iterator if the cell is not present. Used by refinement strategies that
    /// need to re-score a small set of candidates outside the INT8 fast path.
    ///
    /// **Eager-mode only**: reads from the in-memory `TokenStore`. For the
    /// lazy path, use [`Self::tokens_for_cell_via_source`] which falls
    /// back to a [`CellSource`] when `TokenStore` is empty.
    pub fn dequantized_tokens_for_cell(
        &self,
        cell_id: CellId,
    ) -> impl Iterator<Item = (Vec<f32>, OwnerId)> + '_ {
        (0..self.store.len()).filter(move |&i| self.store.cell_ids[i] == cell_id).map(|i| {
            let scale = self.store.scales[i];
            let tokens = self.store.token_data(i);
            let f32_tokens: Vec<f32> = tokens.iter().map(|&v| v as f32 * scale).collect();
            (f32_tokens, self.store.owners[i])
        })
    }

    /// Load a cell's per-token f32 vectors via the `CellSource`, with the
    /// retriever's bounded LRU cache in front.
    ///
    /// Returns `None` if the cell is unknown to the retriever or the
    /// source has no data for it. The returned `Arc<Vec<f32>>` is the
    /// flat `n * d` data slice; callers iterate `d`-sized chunks.
    pub fn cached_tokens_for_cell(
        &self,
        cell_id: CellId,
        source: &dyn CellSource,
    ) -> Option<Arc<Vec<f32>>> {
        let expected_dim = self.dim?;
        self.load_decoded_tokens(cell_id, source, expected_dim)
    }

    /// Convenience: dequantized tokens for a cell, lazy-via-source if a
    /// source is supplied, else fall back to the eager
    /// [`Self::dequantized_tokens_for_cell`] path. Owner is taken from
    /// `cell_summaries` so the result is consistent across modes.
    pub fn tokens_for_cell_via_source(
        &self,
        cell_id: CellId,
        source: Option<&dyn CellSource>,
    ) -> Vec<(Vec<f32>, OwnerId)> {
        if let Some(src) = source
            && let Some(tokens) = self.cached_tokens_for_cell(cell_id, src)
            && let Some(d) = self.dim
            && d > 0
            && let Some(summary) = self.cell_summaries.get(&cell_id)
        {
            return (0..tokens.len() / d)
                .map(|i| (tokens[i * d..(i + 1) * d].to_vec(), summary.owner))
                .collect();
        }
        self.dequantized_tokens_for_cell(cell_id).collect()
    }

    /// Per-token vector dimension, or `None` if no tokens have been inserted.
    pub fn token_dim(&self) -> Option<usize> {
        self.dim
    }

    /// Aggregated number of dot products averaged in `Top5Avg` scoring.
    pub fn top_n(&self) -> usize {
        self.config.top_n_avg_match_count
    }

    /// Mean f32 token vector across the entire corpus, or `None` if empty.
    ///
    /// Used by mean-centered refinement to subtract the dominant shared
    /// direction from query and stored tokens before scoring — the same
    /// "skip position 0" insight applied at corpus scope.
    pub fn corpus_mean(&self) -> Option<Vec<f32>> {
        if self.corpus_token_count == 0 || self.corpus_sum.is_empty() {
            return None;
        }
        let inv = 1.0 / self.corpus_token_count as f32;
        Some(self.corpus_sum.iter().map(|s| s * inv).collect())
    }

    /// Corpus covariance matrix (dim x dim, row-major), or `None` if fewer
    /// than 2 tokens have been inserted.
    ///
    /// Computed from the running sums: `Cov[j,k] = E[x_j*x_k] - E[x_j]*E[x_k]`.
    pub fn corpus_covariance(&self) -> Option<Vec<f32>> {
        if self.corpus_token_count < 2 || self.corpus_sum.is_empty() {
            return None;
        }
        let dim = self.corpus_sum.len();
        if self.corpus_sq_sum.len() != dim * dim {
            return None;
        }
        let n = self.corpus_token_count as f32;
        let mean: Vec<f32> = self.corpus_sum.iter().map(|s| s / n).collect();
        let mut cov = vec![0.0_f32; dim * dim];
        for j in 0..dim {
            for k in 0..dim {
                cov[j * dim + k] = self.corpus_sq_sum[j * dim + k] / n - mean[j] * mean[k];
            }
        }
        Some(cov)
    }

    /// Cached ZCA whitening matrix (dim x dim, row-major).
    ///
    /// Computed via eigendecomposition of the corpus covariance matrix using
    /// `faer`. The cache is invalidated when `corpus_token_count` changes.
    /// Returns `None` if the corpus is too small or eigendecomposition fails.
    pub fn whitening_matrix(&mut self) -> Option<&[f32]> {
        if self.corpus_token_count == self.whitening_cache_count
            && let Some(ref cache) = self.whitening_cache
        {
            return Some(cache.as_slice());
        }
        let cov = self.corpus_covariance()?;
        let dim = self.corpus_sum.len();
        let w = compute_whitening_matrix(&cov, dim)?;
        self.whitening_cache = Some(w);
        self.whitening_cache_count = self.corpus_token_count;
        self.whitening_cache.as_deref()
    }

    /// Number of individual token entries stored.
    pub fn token_count(&self) -> usize {
        self.store.len()
    }

    /// Number of distinct cells indexed.
    pub fn cell_count(&self) -> usize {
        self.cell_token_count.len()
    }

    fn ensure_corpus_sum_dim(&mut self, dim: usize) {
        if self.corpus_sum.len() != dim {
            // First insert (or unexpected dim change) — size the running sum.
            // Mismatched dim resets it; corpus_token_count is reset to keep mean valid.
            self.corpus_sum = vec![0.0; dim];
            self.corpus_token_count = 0;
            self.corpus_sq_sum = vec![0.0; dim * dim];
            self.whitening_cache = None;
            self.whitening_cache_count = 0;
        }
    }

    fn accumulate_corpus(&mut self, token: &[f32]) {
        let dim = self.corpus_sum.len();
        if token.len() != dim {
            return;
        }
        for (sum, value) in self.corpus_sum.iter_mut().zip(token.iter()) {
            *sum += *value;
        }
        self.corpus_token_count += 1;

        // Accumulate outer product for covariance computation.
        if self.corpus_sq_sum.len() == dim * dim {
            for j in 0..dim {
                for k in 0..dim {
                    self.corpus_sq_sum[j * dim + k] += token[j] * token[k];
                }
            }
        }
    }

    fn candidate_limit(&self, k: usize) -> usize {
        self.cell_count().min(self.config.min_candidates.max(k * self.config.candidate_multiplier))
    }

    fn candidate_cells(
        &self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
    ) -> Option<HashSet<CellId>> {
        if self.cell_count() <= self.config.candidate_reduction_threshold {
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
        if self.store.len() == 0 || query_key.is_empty() {
            return Vec::new();
        }

        let Ok(query_view) = RetrievalKeyView::parse(query_key) else {
            return Vec::new();
        };
        let query_tokens: Vec<QuantizedInt8Vec> =
            if let Some((n, d, data)) = query_view.raw_tokens() {
                (0..n).map(|i| Int8Quantizer::quantize(&data[i * d..(i + 1) * d])).collect()
            } else {
                vec![Int8Quantizer::quantize(query_key)]
            };

        let dim = self.dim.unwrap_or(0);
        let inv_sqrt_dk = if dim > 0 { 1.0 / (dim as f32).sqrt() } else { 1.0 };

        let candidate_cells = if use_candidate_reduction && query_view.is_encoded_per_token() {
            self.candidate_cells(query_key, k, owner_filter)
        } else {
            None
        };

        let store_dim = self.store.dim;
        let qt_dim = query_tokens[0].values.len();

        if store_dim != qt_dim {
            return Vec::new();
        }

        // Score all stored tokens against all query tokens.
        // Accumulate dot products per cell using a HashMap, but with
        // inline top-K tracking to avoid growing Vecs.
        let top_n = self.config.top_n_avg_match_count;
        let n_qt = query_tokens.len();

        // Pre-compute capacity hint from cell count
        let mut cell_scores: HashMap<CellId, (Vec<f32>, OwnerId)> =
            HashMap::with_capacity(self.cell_token_count.len());

        for i in 0..self.store.len() {
            let owner = self.store.owners[i];
            if owner_filter.is_some_and(|o| o != owner) {
                continue;
            }

            let cell_id = self.store.cell_ids[i];
            if candidate_cells.as_ref().is_some_and(|ids| !ids.contains(&cell_id)) {
                continue;
            }

            let token_data = self.store.token_data(i);
            let token_scale = self.store.scales[i];

            for qt in &query_tokens {
                let raw = DotProduct::int8_dot_raw_slice(&qt.values, token_data);
                let dot = raw as f32 * qt.scale * token_scale * inv_sqrt_dk;

                cell_scores
                    .entry(cell_id)
                    .or_insert_with(|| (Vec::with_capacity(n_qt * 8), owner))
                    .0
                    .push(dot);
            }
        }
        self.last_scored_cell_count = cell_scores.len();

        let mut results: Vec<RetrievalResult> = cell_scores
            .into_iter()
            .map(|(cell_id, (mut dots, owner))| {
                let score = match self.scoring_mode {
                    ScoringMode::MaxSim => dots.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    ScoringMode::Top5Avg => {
                        let take = dots.len().min(top_n);
                        if take == 0 {
                            f32::NEG_INFINITY
                        } else {
                            dots.select_nth_unstable_by(take - 1, |a, b| {
                                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                            });
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

    /// Query against a [`CellSource`] — the lazy path, with no upstream
    /// candidate filter. Equivalent to
    /// [`Self::query_with_source_and_candidates`] called with `None`.
    pub fn query_with_source(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
        source: &dyn CellSource,
    ) -> Vec<RetrievalResult> {
        self.query_with_source_and_candidates(query_key, k, owner_filter, None, source)
    }

    /// Query against a [`CellSource`] with an optional upstream candidate
    /// filter from earlier pipeline stages.
    ///
    /// Decodes tokens for candidate cells on demand instead of relying on
    /// the in-memory `TokenStore`. The result set is observationally
    /// identical to [`Retriever::query`] modulo INT8 vs f32 dot-product
    /// rounding (the eager path quantizes both sides; the lazy path
    /// scores in f32). When `upstream_candidates` is `Some`, only cells
    /// in that set are scored — Phase-1 selection still runs and the two
    /// sets are intersected.
    pub fn query_with_source_and_candidates(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
        upstream_candidates: Option<&HashSet<CellId>>,
        source: &dyn CellSource,
    ) -> Vec<RetrievalResult> {
        if query_key.is_empty() || self.cell_summaries.is_empty() {
            return Vec::new();
        }

        let Ok(query_view) = RetrievalKeyView::parse(query_key) else {
            return Vec::new();
        };

        let query_tokens: Vec<&[f32]> =
            if let Some((n_query, d_query, data)) = query_view.raw_tokens() {
                (0..n_query).map(|i| &data[i * d_query..(i + 1) * d_query]).collect()
            } else {
                vec![query_key]
            };

        let dim = self.dim.unwrap_or(0);
        if dim == 0 {
            return Vec::new();
        }
        let inv_sqrt_dk = (dim as f32).sqrt().recip();

        let candidate_ids =
            self.lazy_candidate_ids(query_key, k, owner_filter, upstream_candidates);

        let mut results: Vec<RetrievalResult> = candidate_ids
            .iter()
            .filter_map(|cell_id| {
                self.score_candidate_from_source(
                    *cell_id,
                    source,
                    &query_tokens,
                    dim,
                    inv_sqrt_dk,
                    owner_filter,
                )
            })
            .collect();

        self.last_scored_cell_count = results.len();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Select cell IDs to score: Phase-1 reduction when above threshold,
    /// otherwise every summary the retriever knows about. When
    /// `upstream_candidates` is provided, intersect with it so we only
    /// score cells the earlier pipeline stages nominated.
    fn lazy_candidate_ids(
        &self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
        upstream_candidates: Option<&HashSet<CellId>>,
    ) -> Vec<CellId> {
        let base = self
            .candidate_cells(query_key, k, owner_filter)
            .unwrap_or_else(|| self.all_summary_ids(owner_filter).into_iter().collect());
        match upstream_candidates {
            Some(upstream) => base.intersection(upstream).copied().collect(),
            None => base.into_iter().collect(),
        }
    }

    fn all_summary_ids(&self, owner_filter: Option<OwnerId>) -> Vec<CellId> {
        self.cell_summaries
            .iter()
            .filter(|(_, summary)| owner_filter.is_none_or(|owner| owner == summary.owner))
            .map(|(cell_id, _)| *cell_id)
            .collect()
    }

    /// Load a single candidate's tokens via `source` and score them.
    fn score_candidate_from_source(
        &self,
        cell_id: CellId,
        source: &dyn CellSource,
        query_tokens: &[&[f32]],
        expected_dim: usize,
        inv_sqrt_dk: f32,
        owner_filter: Option<OwnerId>,
    ) -> Option<RetrievalResult> {
        let summary = self.cell_summaries.get(&cell_id)?;
        if owner_filter.is_some_and(|owner| owner != summary.owner) {
            return None;
        }
        let tokens = self.load_decoded_tokens(cell_id, source, expected_dim)?;
        let stored_tokens: Vec<&[f32]> = (0..tokens.len() / expected_dim)
            .map(|i| &tokens[i * expected_dim..(i + 1) * expected_dim])
            .collect();
        let score = aggregate_token_scores(
            query_tokens,
            &stored_tokens,
            inv_sqrt_dk,
            self.scoring_mode,
            self.config.top_n_avg_match_count,
        );
        Some(RetrievalResult { cell_id, owner: summary.owner, score })
    }

    /// Cache-Aside: cache hit → return; miss → load via source, decode,
    /// insert into cache, return. The cache stores decoded f32 token
    /// data wrapped in `Arc` so hits are cheap to clone.
    ///
    /// In lazy mode, the INT8 retrieval tier is checked *first* — it
    /// holds full-precision quantized tokens captured at insert time
    /// (before any Q4 round-trip). If the cell is present there, we
    /// dequantize from INT8 (precise) rather than from Q4 (coarse).
    /// The fallback to `source` (Q4 archival) remains for cells the
    /// retriever has not yet seen — for example, when the retriever is
    /// being rebuilt from durable storage on engine restart.
    fn load_decoded_tokens(
        &self,
        cell_id: CellId,
        source: &dyn CellSource,
        expected_dim: usize,
    ) -> Option<Arc<Vec<f32>>> {
        if let Ok(mut cache) = self.token_cache.lock()
            && let Some(hit) = cache.get(&cell_id)
        {
            return Some(Arc::clone(hit));
        }

        // INT8 retrieval tier — precise, captured at insert time.
        if let Some(tokens) = self.dequantize_from_int8_tier(cell_id, expected_dim) {
            let arc = Arc::new(tokens);
            if let Ok(mut cache) = self.token_cache.lock() {
                cache.put(cell_id, Arc::clone(&arc));
            }
            return Some(arc);
        }

        // Fallback: Q4 archival source. Coarse — used only when the
        // INT8 tier has not seen this cell (e.g. after engine reopen
        // before the INT8 tier has been rebuilt from durable cells).
        let encoded = source.get_encoded_key(cell_id)?;
        let (_n_stored, d_stored, data) = decode_per_token_keys(&encoded)?;
        if d_stored != expected_dim {
            return None;
        }
        let tokens = Arc::new(data.to_vec());
        if let Ok(mut cache) = self.token_cache.lock() {
            cache.put(cell_id, Arc::clone(&tokens));
        }
        Some(tokens)
    }

    fn dequantize_from_int8_tier(&self, cell_id: CellId, expected_dim: usize) -> Option<Vec<f32>> {
        // LruCache::get takes &mut self because it updates LRU order on
        // access. We clone the Arc out so the lock is released quickly
        // and the decode work happens without holding the mutex.
        let cell_tokens = {
            let mut int8_store = self.int8_tokens.lock().ok()?;
            Arc::clone(int8_store.get(&cell_id)?)
        };
        let strategy = Int8Group32::new(expected_dim);
        let mut flat = Vec::with_capacity(cell_tokens.len() * expected_dim);
        for quantized in cell_tokens.iter() {
            let decoded = strategy.dequantize_token(quantized);
            if decoded.len() != expected_dim {
                return None;
            }
            flat.extend_from_slice(&decoded);
        }
        Some(flat)
    }
}

/// Aggregate per-token dot products into a single cell score (f32 path).
///
/// Mirrors the [`ScoringMode`] semantics used by the eager INT8 path;
/// only the precision differs (f32 here, INT8 there).
fn aggregate_token_scores(
    query_tokens: &[&[f32]],
    stored_tokens: &[&[f32]],
    inv_sqrt_dk: f32,
    mode: ScoringMode,
    top_n: usize,
) -> f32 {
    const EMPTY_SCORE_SENTINEL: f32 = f32::NEG_INFINITY;
    if query_tokens.is_empty() || stored_tokens.is_empty() {
        return EMPTY_SCORE_SENTINEL;
    }

    let mut dots: Vec<f32> = Vec::with_capacity(query_tokens.len() * stored_tokens.len());
    for qt in query_tokens {
        for st in stored_tokens {
            let raw: f32 = qt.iter().zip(st.iter()).map(|(a, b)| a * b).sum();
            dots.push(raw * inv_sqrt_dk);
        }
    }

    match mode {
        ScoringMode::MaxSim => dots.iter().copied().fold(EMPTY_SCORE_SENTINEL, f32::max),
        ScoringMode::Top5Avg => {
            let take = dots.len().min(top_n);
            if take == 0 {
                return EMPTY_SCORE_SENTINEL;
            }
            dots.select_nth_unstable_by(take - 1, |a, b| {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            });
            dots[..take].iter().sum::<f32>() / take as f32
        }
    }
}

/// Compute ZCA whitening matrix W = U * diag(1/sqrt(lambda)) * U^T from
/// a covariance matrix via `faer`'s self-adjoint eigendecomposition.
///
/// Returns `None` if the input is malformed or `dim == 0`.
fn compute_whitening_matrix(cov: &[f32], dim: usize) -> Option<Vec<f32>> {
    use faer::Mat;

    if cov.len() != dim * dim || dim == 0 {
        return None;
    }

    let mat = Mat::from_fn(dim, dim, |i, j| cov[i * dim + j] as f64);
    let eigen = mat.self_adjoint_eigen(faer::Side::Lower).ok()?;
    let s_diag = eigen.S();
    let u_mat = eigen.U();

    // Extract eigenvalues into a plain Vec for indexing.
    let s_col = s_diag.column_vector();
    let mut eigenvalues = Vec::with_capacity(dim);
    for k in 0..dim {
        eigenvalues.push(*s_col.get(k));
    }

    let eps = 1e-5_f64;
    let mut result = vec![0.0_f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0_f64;
            for k in 0..dim {
                let lam = eigenvalues[k].max(eps);
                sum += u_mat[(i, k)] * (1.0 / lam.sqrt()) * u_mat[(j, k)];
            }
            result[i * dim + j] = sum as f32;
        }
    }
    Some(result)
}

impl Default for PerTokenRetriever {
    fn default() -> Self {
        Self::new()
    }
}

/// Header spans two Q4 groups (64 floats) so the sentinel can dominate its
/// own group (-1e9 `abs_max`) without crushing the data values that follow.
///
/// Layout:
///   Group 0 (indices 0-31):  [SENTINEL, 0, 0, ..., 0]  (sentinel dominates)
///   Group 1 (indices 32-63): `[0, dim, 0, ..., 0]`     (only `dim` is load-bearing)
///   Data (index 64+):        [token vectors]
///
/// # Q4 invariants
///
/// `dim` survives Q4 exactly because it is the `abs_max` of group 1: Q4 stores
/// it as the max signed level (7) and reconstructs it as `7 * (dim / 7) = dim`.
///
/// The `n_tokens` slot at [`N_TOKENS_IDX`] **does not** survive Q4 — when it
/// co-quantizes with `dim` (typically much larger), the per-group scale
/// `dim/7` crushes `n_tokens` to 0 or 1 on round-trip. The encoder therefore
/// writes `0.0` to that slot, and the decoder infers `n` from
/// `data.len() / dim`. The constant [`N_TOKENS_IDX`] is retained for layout
/// stability and on-disk inspection; do not depend on its post-Q4 value.
pub const HEADER_SIZE: usize = 64;

/// Sentinel value at position 0 marking a per-token encoded key.
pub const HEADER_SENTINEL: f32 = -1.0e9;

/// Index within the header reserved for token count. **Informational only** —
/// see [`HEADER_SIZE`] for why this field is not trusted on read.
pub const N_TOKENS_IDX: usize = 32;

/// Index within the header where the vector dimension is stored. Survives Q4.
pub const DIM_IDX: usize = 33;

/// Encode multiple per-token K vectors into a flat f32 slice with header.
///
/// The encoded buffer is designed to survive Q4 round-trip via the storage
/// layer. See [`HEADER_SIZE`] for the precise contract.
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

    // Group 1: only `dim` is stored. `n_tokens` is left as 0.0 because Q4
    // would corrupt it anyway; readers must compute `n = data.len() / dim`.
    // We retain the `n` argument for symmetry with the (n, d) pair but do
    // not write it to the header.
    let _ = n;
    encoded.push(0.0);
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
///
/// # Tolerant Reader
///
/// `n_tokens` is inferred from `data.len() / d`, not read from the header.
/// Q4 quantization corrupts the `n_tokens` slot whenever it shares a 32-float
/// group with a larger value (`dim`): `dim` becomes the group's `abs_max` and
/// drives `scale = dim / 7`, which crushes the much-smaller `n_tokens` to
/// either `0` or `1` on reconstruction. `dim` itself survives Q4 exactly
/// because it *is* the group's `abs_max`, and `data.len()` is preserved
/// because Q4 only changes per-value precision, not buffer length. Together
/// `(dim, data.len())` are sufficient to recover `n`, so the header
/// `n_tokens` field is informational only.
pub fn decode_per_token_keys(encoded: &[f32]) -> Option<(usize, usize, &[f32])> {
    if encoded.len() < HEADER_SIZE {
        return None;
    }

    // Check sentinel in group 0.
    if encoded[0] > -1.0e8 {
        return None;
    }

    // Metadata in group 1: only `dim` is trusted (it survives Q4 as group `abs_max`).
    let d = encoded[DIM_IDX].round() as usize;

    if d == 0 {
        return None;
    }

    let data = &encoded[HEADER_SIZE..];
    if !data.len().is_multiple_of(d) {
        return None;
    }
    let n = data.len() / d;

    if n == 0 {
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
        let Ok(view) = RetrievalKeyView::parse(key) else {
            return;
        };
        let summary_key = view.pooled_vector();
        if let Some((n, d, data)) = view.raw_tokens() {
            if self.dim.is_none() {
                self.dim = Some(d);
            }
            self.ensure_corpus_sum_dim(d);

            // INT8 retrieval tier — only populated in lazy mode, where
            // the on-disk Q4 path is too coarse for scoring.
            let int8_strategy = (!self.populate_token_store).then(|| Int8Group32::new(d));
            let mut int8_tokens_for_cell: Vec<QuantizedToken> =
                if int8_strategy.is_some() { Vec::with_capacity(n) } else { Vec::new() };

            for i in 0..n {
                let token_key = &data[i * d..(i + 1) * d];
                self.accumulate_corpus(token_key);
                if self.populate_token_store {
                    let quantized = Int8Quantizer::quantize(token_key);
                    self.store.push(cell_id, owner, &quantized);
                }
                if let Some(strategy) = &int8_strategy {
                    int8_tokens_for_cell.push(strategy.quantize_token(token_key));
                }
            }

            if int8_strategy.is_some()
                && let Ok(mut int8_store) = self.int8_tokens.lock()
            {
                int8_store.put(cell_id, Arc::new(int8_tokens_for_cell));
            }

            *self.cell_token_count.entry(cell_id).or_insert(0) += n;
        } else {
            if self.dim.is_none() {
                self.dim = Some(key.len());
            }
            self.ensure_corpus_sum_dim(key.len());
            self.accumulate_corpus(key);

            if self.populate_token_store {
                let quantized = Int8Quantizer::quantize(key);
                self.store.push(cell_id, owner, &quantized);
            }

            *self.cell_token_count.entry(cell_id).or_insert(0) += 1;
        }

        self.cell_summaries.insert(cell_id, CellSummary { owner, pooled_key: summary_key });
    }

    fn remove(&mut self, cell_id: CellId) {
        if self.populate_token_store {
            self.store.retain_by_cell(|id| id != cell_id);
        }
        if let Ok(mut int8_store) = self.int8_tokens.lock() {
            int8_store.pop(&cell_id);
        }
        self.cell_token_count.remove(&cell_id);
        self.cell_summaries.remove(&cell_id);
    }

    fn len(&self) -> usize {
        if self.populate_token_store {
            self.store.len()
        } else {
            self.cell_token_count.values().sum()
        }
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    /// Lazy override: when given a [`CellSource`], skip the in-memory
    /// `TokenStore` and decode candidates on demand. Upstream `candidates`,
    /// if provided, intersect with the Phase-1 selection so we only score
    /// cells that survived earlier pipeline stages (e.g. Vamana).
    fn query_with_source(
        &mut self,
        query_key: &[f32],
        k: usize,
        owner_filter: Option<OwnerId>,
        candidates: Option<&HashSet<CellId>>,
        source: Option<&dyn CellSource>,
    ) -> Vec<RetrievalResult> {
        let Some(source) = source else {
            return self.query(query_key, k, owner_filter);
        };
        self.query_with_source_and_candidates(query_key, k, owner_filter, candidates, source)
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

        let defaults = PerTokenConfig::default();
        assert_eq!(
            retriever.last_scored_cell_count,
            defaults.min_candidates.max(QUERY_K * defaults.candidate_multiplier)
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

    #[test]
    fn test_custom_config_activates_candidate_reduction_at_lower_threshold() {
        // GIVEN a PerTokenRetriever with candidate_reduction_threshold = 2
        let config = PerTokenConfig {
            candidate_reduction_threshold: 2,
            min_candidates: 3,
            candidate_multiplier: 1,
            ..PerTokenConfig::default()
        };
        let mut retriever = PerTokenRetriever::with_config(ScoringMode::Top5Avg, config);

        // AND 5 inserted cells (above threshold=2, well below default=512)
        for cell_id in 0..5u64 {
            let encoded = encoded_cell(SMALL_FIXTURE_DIM, cell_id as usize, BROAD_MATCH_TOKENS);
            retriever.insert(cell_id, OWNER_ONE, &encoded);
        }

        let query = encoded_cell(SMALL_FIXTURE_DIM, 2, BROAD_MATCH_TOKENS);
        let _ = retriever.query_inner(&query, 1, Some(OWNER_ONE), true);

        // THEN candidate reduction activates (scored fewer than total cells)
        assert!(
            retriever.last_scored_cell_count < retriever.cell_count(),
            "candidate reduction should activate: scored {} of {} cells",
            retriever.last_scored_cell_count,
            retriever.cell_count(),
        );
    }

    #[test]
    fn test_corpus_mean_is_none_when_empty() {
        let retriever = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        assert!(retriever.corpus_mean().is_none());
    }

    #[test]
    fn test_corpus_mean_averages_per_dim_across_inserted_tokens() {
        let mut retriever = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [3.0f32, 4.0, 5.0, 6.0];
        let c = [5.0f32, 6.0, 7.0, 8.0];

        retriever.insert(0, OWNER_ONE, &encode_per_token_keys(&[&a, &b, &c]));

        let mean = retriever.corpus_mean().expect("mean exists after insert");
        assert_eq!(mean.len(), 4);
        // Mean of (1,3,5)=3, (2,4,6)=4, (3,5,7)=5, (4,6,8)=6.
        let expected = [3.0f32, 4.0, 5.0, 6.0];
        for (got, want) in mean.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "expected {want}, got {got}");
        }
    }

    #[test]
    fn test_corpus_mean_accumulates_across_multiple_cells() {
        let mut retriever = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        let cell0 = [2.0f32, 0.0];
        let cell1 = [0.0f32, 4.0];

        retriever.insert(0, OWNER_ONE, &encode_per_token_keys(&[&cell0]));
        retriever.insert(1, OWNER_ONE, &encode_per_token_keys(&[&cell1]));

        let mean = retriever.corpus_mean().expect("mean exists");
        assert!((mean[0] - 1.0).abs() < 1e-5);
        assert!((mean[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_custom_top_n_changes_scoring_aggregation() {
        // GIVEN two retrievers: one with top_n=1, one with top_n=5
        let config_top1 = PerTokenConfig { top_n_avg_match_count: 1, ..PerTokenConfig::default() };
        let mut retriever_top1 = PerTokenRetriever::with_config(ScoringMode::Top5Avg, config_top1);
        let mut retriever_top5 = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);

        // AND identical cells inserted into both
        for cell_id in 0..3u64 {
            let encoded = encoded_cell(SMALL_FIXTURE_DIM, cell_id as usize, BROAD_MATCH_TOKENS);
            retriever_top1.insert(cell_id, OWNER_ONE, &encoded);
            retriever_top5.insert(cell_id, OWNER_ONE, &encoded);
        }

        let query = encoded_cell(SMALL_FIXTURE_DIM, 0, BROAD_MATCH_TOKENS);
        let results_top1 = retriever_top1.query_inner(&query, 1, None, false);
        let results_top5 = retriever_top5.query_inner(&query, 1, None, false);

        // THEN the top-1 score should differ from top-5 score (different aggregation)
        assert!(!results_top1.is_empty());
        assert!(!results_top5.is_empty());
        let score_top1 = results_top1[0].score;
        let score_top5 = results_top5[0].score;
        assert!(
            (score_top1 - score_top5).abs() > f32::EPSILON,
            "top_n=1 score ({score_top1}) should differ from top_n=5 score ({score_top5})"
        );
    }

    #[test]
    fn test_corpus_covariance_none_when_empty() {
        let r = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        assert!(r.corpus_covariance().is_none());
    }

    #[test]
    fn test_corpus_covariance_computed_correctly() {
        let mut r = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        r.insert(0, OWNER_ONE, &encode_per_token_keys(&[&[2.0_f32, 0.0]]));
        r.insert(1, OWNER_ONE, &encode_per_token_keys(&[&[0.0_f32, 2.0]]));
        let cov = r.corpus_covariance().unwrap();
        assert_eq!(cov.len(), 4);
        // Mean=[1,1]. Var(x)=1, Var(y)=1, Cov(x,y)=-1
        assert!((cov[0] - 1.0).abs() < 0.01);
        assert!((cov[3] - 1.0).abs() < 0.01);
        assert!((cov[1] - (-1.0)).abs() < 0.01);
    }

    /// Regression: decoder must recover `n` from `data.len() / d` even when the
    /// `n_tokens` header slot has been corrupted by Q4 quantization.
    ///
    /// `Q4_0` quantizes 32 floats per group with `scale = ``abs_max`` / 7`. In header
    /// group 1, `dim` (1024) dominates `n_tokens` (127), giving scale=146.29,
    /// which rounds `n_tokens` to 1 and reconstructs it as 146.28572. The
    /// decoder must treat the `n_tokens` slot as informational and trust only
    /// `dim` + the actual data length.
    #[test]
    fn test_decode_tolerates_corrupted_n_tokens_header() {
        const N: usize = 127;
        const D: usize = 1024;
        // The exact Q4 artifact for n=127 alongside dim=1024 in the same group.
        const CORRUPTED_N: f32 = 1024.0 / 7.0;

        let mut buf = vec![0.0f32; HEADER_SIZE + N * D];
        buf[0] = HEADER_SENTINEL;
        buf[N_TOKENS_IDX] = CORRUPTED_N;
        buf[DIM_IDX] = D as f32;
        // Fill data with a deterministic pattern so we can verify the slice.
        for (i, slot) in buf[HEADER_SIZE..].iter_mut().enumerate() {
            *slot = i as f32;
        }

        let (n, d, data) =
            decode_per_token_keys(&buf).expect("decoder must tolerate corrupted n_tokens");

        assert_eq!(n, N, "n must be inferred from data.len() / d, not from header");
        assert_eq!(d, D);
        assert_eq!(data.len(), N * D);
        #[allow(clippy::float_cmp)] // integer-valued payload; bit-exact round-trip
        {
            assert_eq!(data[0], 0.0);
            assert_eq!(data[data.len() - 1], (N * D - 1) as f32);
        }
    }
}
