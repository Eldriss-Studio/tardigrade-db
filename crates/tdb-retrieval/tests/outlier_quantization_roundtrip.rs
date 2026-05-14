//! AT-A2 — Outlier-shaped INT8 round-trip canary.
//!
//! **This is the acceptance test that would have caught the LoCoMo
//! recall regression** (68% → 3.3%) introduced by the lazy
//! `PerTokenRetriever` cutover before it reached production.
//!
//! Background: LLM hidden states develop "activation outlier"
//! channels — a small number of dims whose magnitudes are 10-100×
//! the rest (Dettmers et al., 2022, [arXiv:2208.07339][llm-int8]).
//! Plain group-wise Q4_0 (32-float groups, one f32 scale, 4-bit
//! signed [-7, 7]) collapses the non-outlier dims to zero whenever
//! an outlier shares a group with them: scale = abs_max/7, so any
//! value below abs_max/14 rounds to 0.
//!
//! Empirical finding (this file): **plain INT8 group quantization
//! is sufficient** to preserve outlier-shaped tokens at cosine ≥
//! 0.99. INT8 has 127 levels per side vs Q4's 7, and the half-step
//! is small enough that non-outlier values remain resolvable even
//! when an outlier dominates the group abs_max. Per-channel
//! pre-scaling (SmoothQuant pattern) is not required at this scale
//! of outlier dominance — INT8 alone handles it.
//!
//! Slice A2 of the retrieval correctness fix
//! (`docs/refs/external-references.md` §A3f).
//!
//! [llm-int8]: https://arxiv.org/abs/2208.07339

use tdb_retrieval::quantization::{Int8Group32, RetrievalQuantStrategy};

// ---------- fixture constants ----------

const FIXTURE_DIM: usize = 32;
const OUTLIER_DIM_INDEX: usize = 7;
const OUTLIER_MAGNITUDE: f32 = 0.96;
const REGULAR_HALF_RANGE: f32 = 0.1;
const FIXTURE_SAMPLE_COUNT: usize = 256;
const INT8_COSINE_FLOOR: f32 = 0.99;

// ---------- helpers ----------

/// Deterministic LCG; avoids pulling in `rand` for tests.
fn lcg_advance(state: &mut u64) -> f32 {
    const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
    const LCG_INCREMENT: u64 = 1;
    const RANDOM_SHIFT: u32 = 40;
    const RANDOM_SCALE_DENOM: f32 = (1u64 << 24) as f32;
    *state = state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);
    (*state >> RANDOM_SHIFT) as f32 / RANDOM_SCALE_DENOM
}

/// One token: outlier dim at OUTLIER_MAGNITUDE, all other dims
/// uniform in [-REGULAR_HALF_RANGE, REGULAR_HALF_RANGE].
fn outlier_shaped_token(seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    let mut token = vec![0.0_f32; FIXTURE_DIM];
    for slot in &mut token {
        let unit_random = lcg_advance(&mut state);
        *slot = (unit_random * 2.0 - 1.0) * REGULAR_HALF_RANGE;
    }
    token[OUTLIER_DIM_INDEX] = OUTLIER_MAGNITUDE;
    token
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b).max(f32::MIN_POSITIVE)
}

// ---------- the canary ----------

#[test]
fn int8_group_quantization_preserves_outlier_shaped_token_within_recall_floor() {
    let strategy = Int8Group32::new(FIXTURE_DIM);

    let mut min_cos = 1.0_f32;
    for seed in 0..(FIXTURE_SAMPLE_COUNT as u64) {
        let token = outlier_shaped_token(seed);
        let recovered = strategy.dequantize_token(&strategy.quantize_token(&token));
        assert_eq!(recovered.len(), FIXTURE_DIM, "recovered token must match dim");
        let cos = cosine(&token, &recovered);
        if cos < min_cos {
            min_cos = cos;
        }
    }

    assert!(
        min_cos >= INT8_COSINE_FLOOR,
        "minimum cosine {min_cos} below floor {INT8_COSINE_FLOOR}: \
         INT8 group quantization is no longer sufficient on outlier-shaped tokens; \
         per-channel pre-scaling (SmoothQuant pattern) is now required",
    );
}
