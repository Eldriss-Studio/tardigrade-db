//! `PerChannelScaleCalibrator` — accumulates the per-dimension scale
//! vector σ used to neutralize activation outliers before INT8
//! quantization.
//!
//! # The problem σ solves
//!
//! Group-wise INT8 quantization (group of 32 floats, one f32 scale per
//! group, signed 8-bit values) is sensitive to per-group `abs_max`.
//! When the source data is LLM hidden states, a small number of
//! channels carry magnitudes 10-100× the rest ("activation outlier
//! channels", Dettmers et al., 2022, [arXiv:2208.07339][llm-int8]).
//! In a 32-float group where one channel value is 5.0 and the other
//! 31 are in [-0.5, 0.5], plain group quantization sets the scale
//! to `5.0/127 ≈ 0.039`. The smaller values still resolve at this
//! scale — but at *coarser* widths (`Q4_0`, scale = 5.0/7 = 0.714)
//! everything below ~0.36 collapses to zero.
//!
//! Per-channel pre-scaling (`SmoothQuant` pattern, [Xiao et al.,
//! 2022][smoothquant]) sidesteps this by dividing each dim by its
//! own σ[d] *before* group quantization runs. After scaling, every
//! dim has comparable range across cells; no single channel
//! dominates its group.
//!
//! # API contract
//!
//! Calibration is *one-shot* in this implementation: feed a bounded
//! window of hidden states via [`Self::observe`], then call
//! [`Self::finalize`] to obtain the σ vector. Further `observe`
//! calls after finalize are no-ops (already pinned).
//!
//! σ is `max(|x[d]|)` over the observed window, floored at
//! [`PER_CHANNEL_SCALE_FLOOR`] to prevent divide-by-zero on dead
//! channels.
//!
//! [llm-int8]: https://arxiv.org/abs/2208.07339
//! [smoothquant]: https://arxiv.org/abs/2211.10438

/// Default number of *token vectors* (not cells) the calibrator
/// accumulates before finalizing. 4096 tokens at 127 tokens/cell ≈
/// 32 cells — a small fraction of any realistic corpus and enough
/// to capture stable activation statistics for a fixed model.
pub const DEFAULT_CALIBRATION_WINDOW_TOKENS: usize = 4096;

/// Floor applied to per-channel scales to guard against
/// divide-by-zero on dead (always-zero) channels. Set tight enough
/// that legitimate scales are never floored, loose enough that
/// dead-channel divisions stay numerically stable.
pub const PER_CHANNEL_SCALE_FLOOR: f32 = 1e-6;

/// Accumulates `σ[d] = max(|x[d]|)` across a bounded window of
/// observed hidden-state vectors, then freezes.
///
/// Single Responsibility: this struct does *only* calibration —
/// it does not quantize, does not store tokens, does not know about
/// retrievers. The σ it produces is consumed by the
/// per-channel-scaled quantization strategy.
#[derive(Debug)]
pub struct PerChannelScaleCalibrator {
    window_tokens: usize,
    observed_tokens: usize,
    running_abs_max: Vec<f32>,
    finalized: Option<Vec<f32>>,
}

impl PerChannelScaleCalibrator {
    /// Create a calibrator with an explicit token window.
    ///
    /// `dim` is the per-token vector dimension (e.g. 1024 for
    /// Qwen3-0.6B hidden states). `window_tokens` caps how many
    /// individual token vectors contribute to σ before finalization.
    pub fn new(dim: usize, window_tokens: usize) -> Self {
        Self {
            window_tokens,
            observed_tokens: 0,
            running_abs_max: vec![0.0_f32; dim],
            finalized: None,
        }
    }

    /// Create with the default window of
    /// [`DEFAULT_CALIBRATION_WINDOW_TOKENS`].
    pub fn with_default_window(dim: usize) -> Self {
        Self::new(dim, DEFAULT_CALIBRATION_WINDOW_TOKENS)
    }

    /// Per-token vector dimension this calibrator was built for.
    pub fn dim(&self) -> usize {
        self.running_abs_max.len()
    }

    /// True after [`Self::finalize`] has been called.
    pub fn is_finalized(&self) -> bool {
        self.finalized.is_some()
    }

    /// True when enough tokens have been observed to be ready to
    /// finalize. Always returns `true` after finalization regardless
    /// of count, so callers can poll with a single predicate.
    pub fn is_ready(&self) -> bool {
        self.finalized.is_some() || self.observed_tokens >= self.window_tokens
    }

    /// Feed one token vector into the calibration window.
    ///
    /// No-op if already finalized, if the vector's length does not
    /// match this calibrator's dim, or if the window is already
    /// full. Returns the number of tokens consumed *by this call*
    /// (0 or 1) so callers can detect saturation without polling
    /// internal counters.
    pub fn observe(&mut self, token: &[f32]) -> usize {
        if self.finalized.is_some() || token.len() != self.running_abs_max.len() {
            return 0;
        }
        if self.observed_tokens >= self.window_tokens {
            return 0;
        }
        for (max, value) in self.running_abs_max.iter_mut().zip(token.iter()) {
            let abs_value = value.abs();
            if abs_value > *max {
                *max = abs_value;
            }
        }
        self.observed_tokens += 1;
        1
    }

    /// Freeze and return the σ vector.
    ///
    /// After finalization, the calibrator is read-only — subsequent
    /// [`Self::observe`] calls are no-ops, and [`Self::finalize`]
    /// returns a clone of the previously computed σ. Each element
    /// is `max(running_abs_max[d], PER_CHANNEL_SCALE_FLOOR)`.
    pub fn finalize(&mut self) -> Vec<f32> {
        if let Some(existing) = &self.finalized {
            return existing.clone();
        }
        let sigma: Vec<f32> =
            self.running_abs_max.iter().map(|max| max.max(PER_CHANNEL_SCALE_FLOOR)).collect();
        self.finalized = Some(sigma.clone());
        sigma
    }

    /// Number of tokens observed so far (saturating at the window).
    /// Exposed for diagnostics; not load-bearing for behavior.
    pub fn observed_tokens(&self) -> usize {
        self.observed_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_DIM: usize = 8;
    const FIXTURE_WINDOW: usize = 256;
    const OUTLIER_DIM_INDEX: usize = 3;
    const OUTLIER_MAGNITUDE: f32 = 5.0;
    const REGULAR_MAGNITUDE: f32 = 0.5;
    const SIGMA_TOLERANCE_FRACTION: f32 = 0.1;

    fn make_outlier_token(seed: u64) -> Vec<f32> {
        // Deterministic LCG for reproducibility without rand dep.
        const LCG: u64 = 6_364_136_223_846_793_005;
        const INC: u64 = 1;
        const RANDOM_SHIFT: u32 = 40;
        const RANDOM_SCALE_DENOM: f32 = (1u64 << 24) as f32;

        let mut state = seed.wrapping_mul(LCG).wrapping_add(INC);
        let mut token = vec![0.0_f32; FIXTURE_DIM];
        for slot in &mut token {
            state = state.wrapping_mul(LCG).wrapping_add(INC);
            let unit_random = (state >> RANDOM_SHIFT) as f32 / RANDOM_SCALE_DENOM;
            *slot = (unit_random * 2.0 - 1.0) * REGULAR_MAGNITUDE;
        }
        token[OUTLIER_DIM_INDEX] = OUTLIER_MAGNITUDE;
        token
    }

    #[test]
    fn captures_outlier_dim_above_observed_magnitude() {
        let mut calibrator = PerChannelScaleCalibrator::new(FIXTURE_DIM, FIXTURE_WINDOW);
        for seed in 0..(FIXTURE_WINDOW as u64 + 50) {
            calibrator.observe(&make_outlier_token(seed));
        }
        let sigma = calibrator.finalize();

        assert_eq!(sigma.len(), FIXTURE_DIM);
        assert!(
            sigma[OUTLIER_DIM_INDEX] >= OUTLIER_MAGNITUDE,
            "outlier σ must reach the observed magnitude (≥{OUTLIER_MAGNITUDE}); \
             got {}",
            sigma[OUTLIER_DIM_INDEX],
        );
    }

    #[test]
    fn other_dims_settle_near_regular_magnitude() {
        let mut calibrator = PerChannelScaleCalibrator::new(FIXTURE_DIM, FIXTURE_WINDOW);
        for seed in 0..(FIXTURE_WINDOW as u64) {
            calibrator.observe(&make_outlier_token(seed));
        }
        let sigma = calibrator.finalize();

        let upper_bound = REGULAR_MAGNITUDE * (1.0 + SIGMA_TOLERANCE_FRACTION);
        let lower_bound = REGULAR_MAGNITUDE * (1.0 - SIGMA_TOLERANCE_FRACTION);
        for (dim_index, sigma_value) in sigma.iter().enumerate() {
            if dim_index == OUTLIER_DIM_INDEX {
                continue;
            }
            assert!(
                *sigma_value <= upper_bound,
                "σ[{dim_index}] = {sigma_value} exceeds upper bound {upper_bound}",
            );
            assert!(
                *sigma_value >= lower_bound,
                "σ[{dim_index}] = {sigma_value} below lower bound {lower_bound}; \
                 calibration under-saturated",
            );
        }
    }

    #[test]
    fn observe_stops_after_window_saturates() {
        let mut calibrator = PerChannelScaleCalibrator::new(FIXTURE_DIM, FIXTURE_WINDOW);
        let mut consumed = 0;
        for seed in 0..(FIXTURE_WINDOW as u64 * 2) {
            consumed += calibrator.observe(&make_outlier_token(seed));
        }
        assert_eq!(
            consumed, FIXTURE_WINDOW,
            "calibrator must stop accepting tokens once window saturates",
        );
        assert!(calibrator.is_ready());
    }

    #[test]
    fn finalize_is_idempotent() {
        let mut calibrator = PerChannelScaleCalibrator::new(FIXTURE_DIM, FIXTURE_WINDOW);
        for seed in 0..(FIXTURE_WINDOW as u64) {
            calibrator.observe(&make_outlier_token(seed));
        }
        let first = calibrator.finalize();
        let second = calibrator.finalize();
        assert_eq!(first, second, "finalize must be deterministic across calls");
    }

    #[test]
    fn observe_floor_protects_dead_channels() {
        let mut calibrator = PerChannelScaleCalibrator::new(FIXTURE_DIM, FIXTURE_WINDOW);
        let dead_token = vec![0.0_f32; FIXTURE_DIM];
        for _ in 0..FIXTURE_WINDOW {
            calibrator.observe(&dead_token);
        }
        let sigma = calibrator.finalize();
        for sigma_value in &sigma {
            assert!(
                *sigma_value >= PER_CHANNEL_SCALE_FLOOR,
                "dead-channel σ must be floored to {PER_CHANNEL_SCALE_FLOOR}, \
                 got {sigma_value}",
            );
        }
    }
}
