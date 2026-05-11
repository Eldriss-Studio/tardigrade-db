//! Per-token importance weighting — Decorator pattern.
//!
//! Weight = `1 - cosine(token, corpus_mean)`. Distinctive tokens
//! (orthogonal to mean) get high weight; common tokens get low weight.

/// Compute the importance weight for a single token vector relative to the corpus mean.
///
/// Returns `1 - cosine(token, corpus_mean)`, clamped to `[0.0, 1.0]`.
///
/// - A token aligned with the mean (cosine ≈ 1) gets weight ≈ 0 — it is common/generic.
/// - A token orthogonal to the mean (cosine ≈ 0) gets weight ≈ 1 — it is distinctive.
/// - Zero-norm inputs return `1.0` (treat unknowns as fully weighted).
pub fn token_weight(token: &[f32], corpus_mean: &[f32]) -> f32 {
    let dot: f32 = token.iter().zip(corpus_mean).map(|(a, b)| a * b).sum();
    let norm_t = token.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_m = corpus_mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_t < 1e-9 || norm_m < 1e-9 {
        return 1.0;
    }
    (1.0 - dot / (norm_t * norm_m)).clamp(0.0, 1.0)
}

/// Return the token weight when `enabled`, or `1.0` when disabled.
///
/// The unit return allows callers to multiply scores by this value
/// without branching in the hot path — the identity element `1.0` makes
/// the multiplication a no-op when reweighting is off.
pub fn weighted_or_unit(enabled: bool, token: &[f32], corpus_mean: &[f32]) -> f32 {
    if enabled { token_weight(token, corpus_mean) } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_near_mean_is_low() {
        let mean = [1.0_f32, 0.0, 0.0];
        let token = [0.9_f32, 0.1, 0.0];
        assert!(token_weight(&token, &mean) < 0.3);
    }

    #[test]
    fn weight_orthogonal_is_high() {
        let mean = [1.0_f32, 0.0, 0.0];
        let token = [0.0_f32, 1.0, 0.0];
        assert!(token_weight(&token, &mean) > 0.8);
    }

    #[test]
    fn disabled_returns_one() {
        assert!((weighted_or_unit(false, &[0.5, 0.5], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weight_clamped_non_negative() {
        let mean = [1.0_f32, 0.0];
        let token = [1.0_f32, 0.0];
        assert!(token_weight(&token, &mean) >= 0.0);
    }
}
