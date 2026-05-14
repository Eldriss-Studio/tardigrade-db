//! Two-stage retrieval refinement — vague-query rescue.
//!
//! After the first-stage pipeline returns top-K candidates, an optional
//! refinement pass re-scores or re-queries to improve recall on queries
//! whose vocabulary doesn't overlap the stored content (the 48% R@5 vague
//! cliff measured in `docs/experiments/README.md`).
//!
//! ## Strategies
//!
//! - [`RefinementMode::None`] — no-op (default; preserves all existing
//!   behavior and test expectations).
//! - [`RefinementMode::MeanCentered`] — subtract the corpus-mean K vector
//!   from query and stored tokens before re-scoring the top-K candidates.
//!   Same insight as the position-0 attention-sink skip, applied at
//!   corpus scope. Single re-score pass, no extra retrieval.
//! - [`RefinementMode::LatentPrf`] — Rocchio-style pseudo-relevance
//!   feedback in latent space: form `q' = α·q + β·centroid(top_K')` and
//!   re-retrieve. Closes the vocabulary mismatch by pulling the query
//!   toward the cluster of plausible top results. One extra first-stage
//!   pass; no LLM call, no text dependency.
//!
//! ## Why not `HyDE` / cross-encoder / `BM25`?
//!
//! See `docs/refs/external-references.md` C2 — `HyDE` rejected for
//! per-query LLM latency (500–2000ms breaks the agent-loop budget);
//! cross-encoder/`BM25` rejected as primary stages because `TardigradeDB`'s
//! stored unit is `KV` tensors, not text. Latent-space `PRF` is the
//! architecturally aligned choice (Yu et al. CIKM 2021, `ColBERT-PRF`
//! Wang et al. 2022).

use tdb_core::OwnerId;

use crate::attention::RetrievalResult;
use crate::cell_source::CellSource;
use crate::per_token::{PerTokenRetriever, encode_per_token_keys};

// ── Strategy pattern: trait + concrete strategies ───────────────────────

/// Object-safe trait for post-retrieval refinement strategies.
///
/// Each implementation encapsulates the algorithm and its parameters.
/// The engine holds a `Box<dyn RefinementStrategy>` and delegates to it,
/// replacing the previous `match` on [`RefinementMode`].
pub trait RefinementStrategy: Send + Sync + std::fmt::Debug {
    /// Apply refinement to first-stage candidates.
    ///
    /// Semantics are identical to [`apply`]: re-score or re-query the
    /// candidates, returning at most `k` results.
    ///
    /// `source`: when `Some`, strategies that need per-cell tokens
    /// (mean-centered, whitened, PRF) load them lazily via the
    /// [`CellSource`] instead of relying on the retriever's in-memory
    /// token store. Engines wired for lazy mode pass `Some`; legacy
    /// callers that still use the in-memory store pass `None`.
    fn refine(
        &self,
        retriever: &mut PerTokenRetriever,
        query_key: &[f32],
        first_stage: Vec<RetrievalResult>,
        k: usize,
        owner_filter: Option<OwnerId>,
        source: Option<&dyn CellSource>,
    ) -> Vec<RetrievalResult>;

    /// Human-readable name used by Python bindings and diagnostics.
    fn name(&self) -> &'static str;

    /// Whether this strategy is a no-op (allows short-circuit in the engine).
    fn is_noop(&self) -> bool {
        false
    }
}

/// Pass-through: first-stage results returned unchanged.
#[derive(Debug)]
pub struct NoOpStrategy;

impl RefinementStrategy for NoOpStrategy {
    fn refine(
        &self,
        _retriever: &mut PerTokenRetriever,
        _query_key: &[f32],
        first_stage: Vec<RetrievalResult>,
        _k: usize,
        _owner_filter: Option<OwnerId>,
        _source: Option<&dyn CellSource>,
    ) -> Vec<RetrievalResult> {
        first_stage
    }

    fn name(&self) -> &'static str {
        "none"
    }

    fn is_noop(&self) -> bool {
        true
    }
}

/// Subtract corpus mean from query + stored tokens, re-score top candidates.
#[derive(Debug)]
pub struct MeanCenteredStrategy;

impl RefinementStrategy for MeanCenteredStrategy {
    fn refine(
        &self,
        retriever: &mut PerTokenRetriever,
        query_key: &[f32],
        first_stage: Vec<RetrievalResult>,
        k: usize,
        _owner_filter: Option<OwnerId>,
        source: Option<&dyn CellSource>,
    ) -> Vec<RetrievalResult> {
        mean_centered_rescore(retriever, query_key, first_stage, k, source)
    }

    fn name(&self) -> &'static str {
        "centered"
    }
}

/// Rocchio expansion in latent K-space, then re-retrieve.
#[derive(Debug)]
pub struct LatentPrfStrategy {
    pub alpha: f32,
    pub beta: f32,
    pub k_prime: usize,
}

impl RefinementStrategy for LatentPrfStrategy {
    fn refine(
        &self,
        retriever: &mut PerTokenRetriever,
        query_key: &[f32],
        first_stage: Vec<RetrievalResult>,
        k: usize,
        owner_filter: Option<OwnerId>,
        source: Option<&dyn CellSource>,
    ) -> Vec<RetrievalResult> {
        latent_prf(
            retriever,
            query_key,
            first_stage,
            self.alpha,
            self.beta,
            self.k_prime,
            k,
            owner_filter,
            source,
        )
    }

    fn name(&self) -> &'static str {
        "prf"
    }
}

/// ZCA whitening: transform query + stored tokens by W = U * diag(1/sqrt(lambda)) * U^T
/// before scoring. Decorrelates dimensions and equalizes variance, making the
/// dot product sensitive to semantic differences rather than dominant activation
/// directions. Falls back to mean-centered if corpus is too small for eigendecomposition.
#[derive(Debug)]
pub struct WhiteningStrategy;

impl RefinementStrategy for WhiteningStrategy {
    fn refine(
        &self,
        retriever: &mut PerTokenRetriever,
        query_key: &[f32],
        first_stage: Vec<RetrievalResult>,
        k: usize,
        _owner_filter: Option<OwnerId>,
        source: Option<&dyn CellSource>,
    ) -> Vec<RetrievalResult> {
        whitened_rescore(retriever, query_key, first_stage, k, source)
    }

    fn name(&self) -> &'static str {
        "whitened"
    }
}

/// Factory: create a boxed strategy from a name string (for Python bindings).
///
/// Accepted names: `"none"`, `"centered"` / `"mean_centered"`, `"prf"` / `"latent_prf"`.
/// Returns `None` for unrecognized names.
pub fn strategy_from_name(
    name: &str,
    alpha: Option<f32>,
    beta: Option<f32>,
    k_prime: Option<usize>,
) -> Option<Box<dyn RefinementStrategy>> {
    match name {
        "none" => Some(Box::new(NoOpStrategy)),
        "centered" | "mean_centered" => Some(Box::new(MeanCenteredStrategy)),
        "prf" | "latent_prf" => Some(Box::new(LatentPrfStrategy {
            alpha: alpha.unwrap_or(0.7),
            beta: beta.unwrap_or(0.3),
            k_prime: k_prime.unwrap_or(3),
        })),
        "whitened" | "zca" => Some(Box::new(WhiteningStrategy)),
        _ => None,
    }
}

/// Refinement strategy applied after the first-stage retrieval pipeline.
///
/// Default is [`RefinementMode::None`] — existing engine behavior is
/// preserved exactly when refinement is not opted in.
#[derive(Debug, Clone, Copy, Default)]
pub enum RefinementMode {
    /// Pass-through. First-stage results returned unchanged.
    #[default]
    None,
    /// Subtract corpus mean from query + stored tokens, re-score top candidates.
    MeanCentered,
    /// Rocchio expansion in latent K-space, then re-retrieve.
    ///
    /// `alpha` weights the original query, `beta` weights the centroid of
    /// the top `k_prime` first-stage results. Conservative defaults
    /// (α=0.7, β=0.3, `k_prime`=3) follow the dense-PRF survey (Li et al.
    /// TOIS 2023) which warns of drift if α is too low.
    LatentPrf { alpha: f32, beta: f32, k_prime: usize },
}

/// Apply a refinement strategy to first-stage results.
///
/// `query_key` is the original encoded per-token query (`encode_per_token_keys`
/// format). `first_stage` is the output of the existing pipeline. The
/// retriever is taken `&mut` because PRF re-retrieves through it.
pub fn apply(
    mode: RefinementMode,
    retriever: &mut PerTokenRetriever,
    query_key: &[f32],
    first_stage: Vec<RetrievalResult>,
    k: usize,
    owner_filter: Option<OwnerId>,
) -> Vec<RetrievalResult> {
    match mode {
        RefinementMode::None => first_stage,
        RefinementMode::MeanCentered => {
            mean_centered_rescore(retriever, query_key, first_stage, k, None)
        }
        RefinementMode::LatentPrf { alpha, beta, k_prime } => latent_prf(
            retriever,
            query_key,
            first_stage,
            alpha,
            beta,
            k_prime,
            k,
            owner_filter,
            None,
        ),
    }
}

/// Re-score top candidates after subtracting the corpus mean from both sides
/// of the dot product.
fn mean_centered_rescore(
    retriever: &PerTokenRetriever,
    query_key: &[f32],
    first_stage: Vec<RetrievalResult>,
    k: usize,
    source: Option<&dyn CellSource>,
) -> Vec<RetrievalResult> {
    if first_stage.is_empty() {
        return first_stage;
    }
    let Some(mean) = retriever.corpus_mean() else {
        return first_stage;
    };
    let Some(query_tokens) = parse_query_tokens(query_key, mean.len()) else {
        return first_stage;
    };

    let centered_query: Vec<Vec<f32>> =
        query_tokens.iter().map(|tok| subtract_in_place(tok.clone(), &mean)).collect();

    let dim = mean.len();
    let inv_sqrt_dk = 1.0 / (dim as f32).sqrt();
    let top_n = retriever.top_n();

    let mut rescored: Vec<RetrievalResult> = first_stage
        .into_iter()
        .map(|res| {
            let mut dots: Vec<f32> = Vec::new();
            for (stored_token, _owner) in retriever.tokens_for_cell_via_source(res.cell_id, source)
            {
                if stored_token.len() != dim {
                    continue;
                }
                let centered_stored = subtract_in_place(stored_token, &mean);
                for cq in &centered_query {
                    dots.push(dot(cq, &centered_stored) * inv_sqrt_dk);
                }
            }
            let score = top_n_avg(&mut dots, top_n);
            RetrievalResult { cell_id: res.cell_id, owner: res.owner, score }
        })
        .collect();

    rescored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    rescored.truncate(k);
    rescored
}

/// Re-score top candidates after ZCA whitening both query and stored tokens.
///
/// Falls back to [`mean_centered_rescore`] when the corpus is too small for
/// eigendecomposition (fewer than 2 tokens).
fn whitened_rescore(
    retriever: &mut PerTokenRetriever,
    query_key: &[f32],
    first_stage: Vec<RetrievalResult>,
    k: usize,
    source: Option<&dyn CellSource>,
) -> Vec<RetrievalResult> {
    if first_stage.is_empty() {
        return first_stage;
    }
    let Some(w_matrix) = retriever.whitening_matrix() else {
        // Corpus too small — fall back to mean-centered.
        return mean_centered_rescore(retriever, query_key, first_stage, k, source);
    };
    // Copy to avoid borrow conflict with retriever methods below.
    let w_matrix = w_matrix.to_vec();
    let Some(mean) = retriever.corpus_mean() else {
        return first_stage;
    };
    let dim = mean.len();
    let Some(query_tokens) = parse_query_tokens(query_key, dim) else {
        return first_stage;
    };

    let whitened_query: Vec<Vec<f32>> = query_tokens
        .iter()
        .map(|tok| mat_vec_mul(&w_matrix, &subtract_in_place(tok.clone(), &mean), dim))
        .collect();

    let inv_sqrt_dk = 1.0 / (dim as f32).sqrt();
    let top_n = retriever.top_n();

    let mut rescored: Vec<RetrievalResult> = first_stage
        .into_iter()
        .map(|res| {
            let mut dots: Vec<f32> = Vec::new();
            for (stored_token, _owner) in retriever.tokens_for_cell_via_source(res.cell_id, source)
            {
                if stored_token.len() != dim {
                    continue;
                }
                let ws = mat_vec_mul(&w_matrix, &subtract_in_place(stored_token, &mean), dim);
                for wq in &whitened_query {
                    dots.push(dot(wq, &ws) * inv_sqrt_dk);
                }
            }
            let score = top_n_avg(&mut dots, top_n);
            RetrievalResult { cell_id: res.cell_id, owner: res.owner, score }
        })
        .collect();

    rescored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    rescored.truncate(k);
    rescored
}

/// Multiply a dim x dim row-major matrix by a dim-length vector.
fn mat_vec_mul(matrix: &[f32], vec: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0_f32; dim];
    for i in 0..dim {
        let row = &matrix[i * dim..(i + 1) * dim];
        result[i] = row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
    }
    result
}

/// Rocchio-in-K-space: expand query toward centroid of top-K' results, re-retrieve.
fn latent_prf(
    retriever: &mut PerTokenRetriever,
    query_key: &[f32],
    first_stage: Vec<RetrievalResult>,
    alpha: f32,
    beta: f32,
    k_prime: usize,
    k: usize,
    owner_filter: Option<OwnerId>,
    source: Option<&dyn CellSource>,
) -> Vec<RetrievalResult> {
    if first_stage.is_empty() || k_prime == 0 {
        return first_stage;
    }
    let Some(dim) = retriever.token_dim() else {
        return first_stage;
    };
    let Some(query_tokens) = parse_query_tokens(query_key, dim) else {
        return first_stage;
    };

    // Aggregate centroid across all tokens of the top k_prime cells.
    let mut centroid = vec![0.0_f32; dim];
    let mut count = 0_usize;
    for res in first_stage.iter().take(k_prime) {
        for (token, _owner) in retriever.tokens_for_cell_via_source(res.cell_id, source) {
            if token.len() != dim {
                continue;
            }
            for (acc, value) in centroid.iter_mut().zip(token.iter()) {
                *acc += *value;
            }
            count += 1;
        }
    }
    if count == 0 {
        return first_stage;
    }
    let inv = 1.0 / count as f32;
    for value in &mut centroid {
        *value *= inv;
    }

    // Form expanded query tokens: q' = alpha * q + beta * centroid.
    let expanded: Vec<Vec<f32>> = query_tokens
        .iter()
        .map(|tok| {
            tok.iter().zip(centroid.iter()).map(|(q, c)| alpha * q + beta * c).collect::<Vec<f32>>()
        })
        .collect();
    let refs: Vec<&[f32]> = expanded.iter().map(Vec::as_slice).collect();
    let expanded_key = encode_per_token_keys(&refs);

    let prf_results = match source {
        Some(src) => {
            <PerTokenRetriever>::query_with_source(retriever, &expanded_key, k, owner_filter, src)
        }
        None => <PerTokenRetriever as crate::retriever::Retriever>::query(
            retriever,
            &expanded_key,
            k,
            owner_filter,
        ),
    };

    if prf_results.is_empty() {
        return truncate(first_stage, k);
    }
    prf_results
}

fn parse_query_tokens(query_key: &[f32], expected_dim: usize) -> Option<Vec<Vec<f32>>> {
    use crate::per_token::decode_per_token_keys;
    let (n, d, data) = decode_per_token_keys(query_key)?;
    if d != expected_dim {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(data[i * d..(i + 1) * d].to_vec());
    }
    Some(out)
}

fn subtract_in_place(mut v: Vec<f32>, mean: &[f32]) -> Vec<f32> {
    for (x, m) in v.iter_mut().zip(mean.iter()) {
        *x -= *m;
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn top_n_avg(dots: &mut [f32], top_n: usize) -> f32 {
    if dots.is_empty() {
        return f32::NEG_INFINITY;
    }
    let take = dots.len().min(top_n);
    dots.select_nth_unstable_by(take - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    dots[..take].iter().sum::<f32>() / take as f32
}

fn truncate(mut v: Vec<RetrievalResult>, k: usize) -> Vec<RetrievalResult> {
    v.truncate(k);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::per_token::{PerTokenRetriever, ScoringMode, encode_per_token_keys};
    use crate::retriever::Retriever;

    const OWNER: OwnerId = 1;

    fn populated_retriever() -> PerTokenRetriever {
        let mut r = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        // 3 cells with deliberately clear separation in dim=4.
        let cell0 = [1.0_f32, 0.0, 0.0, 0.0];
        let cell1 = [0.0_f32, 1.0, 0.0, 0.0];
        let cell2 = [0.0_f32, 0.0, 1.0, 0.0];
        r.insert(0, OWNER, &encode_per_token_keys(&[&cell0, &cell0]));
        r.insert(1, OWNER, &encode_per_token_keys(&[&cell1, &cell1]));
        r.insert(2, OWNER, &encode_per_token_keys(&[&cell2, &cell2]));
        r
    }

    #[test]
    fn none_mode_is_pass_through() {
        let mut r = populated_retriever();
        let q = encode_per_token_keys(&[&[1.0_f32, 0.0, 0.0, 0.0][..]]);
        let first_stage = r.query(&q, 3, Some(OWNER));
        let refined = apply(RefinementMode::None, &mut r, &q, first_stage.clone(), 3, Some(OWNER));
        assert_eq!(refined.len(), first_stage.len());
        for (a, b) in refined.iter().zip(first_stage.iter()) {
            assert_eq!(a.cell_id, b.cell_id);
        }
    }

    #[test]
    fn mean_centered_returns_results_when_corpus_has_data() {
        let mut r = populated_retriever();
        let q = encode_per_token_keys(&[&[1.0_f32, 0.0, 0.0, 0.0][..]]);
        let first_stage = r.query(&q, 3, Some(OWNER));
        let refined = apply(RefinementMode::MeanCentered, &mut r, &q, first_stage, 3, Some(OWNER));
        // Top-1 should still be cell 0 (the one aligned with the query direction).
        assert!(!refined.is_empty());
        assert_eq!(refined[0].cell_id, 0);
    }

    #[test]
    fn mean_centered_falls_back_when_first_stage_empty() {
        let mut r = populated_retriever();
        let q = encode_per_token_keys(&[&[1.0_f32, 0.0, 0.0, 0.0][..]]);
        let refined = apply(RefinementMode::MeanCentered, &mut r, &q, Vec::new(), 3, Some(OWNER));
        assert!(refined.is_empty());
    }

    #[test]
    fn latent_prf_returns_results_for_normal_query() {
        let mut r = populated_retriever();
        let q = encode_per_token_keys(&[&[1.0_f32, 0.0, 0.0, 0.0][..]]);
        let first_stage = r.query(&q, 3, Some(OWNER));
        let refined = apply(
            RefinementMode::LatentPrf { alpha: 0.7, beta: 0.3, k_prime: 1 },
            &mut r,
            &q,
            first_stage,
            3,
            Some(OWNER),
        );
        assert!(!refined.is_empty());
        // With k_prime=1 the centroid pulls toward cell 0 — top result must be cell 0.
        assert_eq!(refined[0].cell_id, 0);
    }

    #[test]
    fn latent_prf_handles_empty_first_stage() {
        let mut r = populated_retriever();
        let q = encode_per_token_keys(&[&[1.0_f32, 0.0, 0.0, 0.0][..]]);
        let refined = apply(
            RefinementMode::LatentPrf { alpha: 0.7, beta: 0.3, k_prime: 3 },
            &mut r,
            &q,
            Vec::new(),
            3,
            Some(OWNER),
        );
        assert!(refined.is_empty());
    }

    /// Strategy trait objects produce the same results as the enum dispatch.
    #[test]
    fn strategy_trait_produces_same_results_as_enum() {
        let mut r = populated_retriever();
        let q = encode_per_token_keys(&[&[1.0_f32, 0.0, 0.0, 0.0][..]]);
        let first_stage = r.query(&q, 3, Some(OWNER));

        // Enum dispatch (existing).
        let enum_result =
            apply(RefinementMode::None, &mut r, &q, first_stage.clone(), 3, Some(OWNER));

        // Trait dispatch (new).
        let strategy = strategy_from_name("none", None, None, None).unwrap();
        let trait_result = strategy.refine(&mut r, &q, first_stage.clone(), 3, Some(OWNER), None);

        let enum_ids: Vec<_> = enum_result.iter().map(|r| r.cell_id).collect();
        let trait_ids: Vec<_> = trait_result.iter().map(|r| r.cell_id).collect();
        assert_eq!(enum_ids, trait_ids);

        // MeanCentered via trait.
        let first_stage2 = r.query(&q, 3, Some(OWNER));
        let enum_mc =
            apply(RefinementMode::MeanCentered, &mut r, &q, first_stage2.clone(), 3, Some(OWNER));
        let strategy_mc = strategy_from_name("centered", None, None, None).unwrap();
        let trait_mc = strategy_mc.refine(&mut r, &q, first_stage2, 3, Some(OWNER), None);
        let enum_mc_ids: Vec<_> = enum_mc.iter().map(|r| r.cell_id).collect();
        let trait_mc_ids: Vec<_> = trait_mc.iter().map(|r| r.cell_id).collect();
        assert_eq!(enum_mc_ids, trait_mc_ids);
    }

    #[test]
    fn whitened_produces_different_scores_than_mean_centered() {
        let mut r = populated_retriever();
        let q = encode_per_token_keys(&[&[0.7_f32, 0.3, 0.1, 0.0][..]]);
        let first = r.query(&q, 3, Some(OWNER));
        let mc = MeanCenteredStrategy.refine(&mut r, &q, first.clone(), 3, Some(OWNER), None);
        let wh = WhiteningStrategy.refine(&mut r, &q, first, 3, Some(OWNER), None);
        let mc_scores: Vec<f32> = mc.iter().map(|r| r.score).collect();
        let wh_scores: Vec<f32> = wh.iter().map(|r| r.score).collect();
        assert_ne!(mc_scores, wh_scores);
    }

    #[test]
    fn whitened_falls_back_when_corpus_too_small() {
        let mut r = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        r.insert(0, OWNER, &encode_per_token_keys(&[&[1.0_f32, 0.0, 0.0, 0.0]]));
        let q = encode_per_token_keys(&[&[1.0_f32, 0.0, 0.0, 0.0][..]]);
        let first = r.query(&q, 1, Some(OWNER));
        let refined = WhiteningStrategy.refine(&mut r, &q, first, 1, Some(OWNER), None);
        assert!(!refined.is_empty());
    }

    #[test]
    fn latent_prf_with_zero_kprime_is_pass_through() {
        let mut r = populated_retriever();
        let q = encode_per_token_keys(&[&[1.0_f32, 0.0, 0.0, 0.0][..]]);
        let first_stage = r.query(&q, 3, Some(OWNER));
        let refined = apply(
            RefinementMode::LatentPrf { alpha: 0.7, beta: 0.3, k_prime: 0 },
            &mut r,
            &q,
            first_stage.clone(),
            3,
            Some(OWNER),
        );
        let ids_a: Vec<_> = refined.iter().map(|r| r.cell_id).collect();
        let ids_b: Vec<_> = first_stage.iter().map(|r| r.cell_id).collect();
        assert_eq!(ids_a, ids_b);
    }
}
