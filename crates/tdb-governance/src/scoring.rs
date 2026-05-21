//! Importance scoring (ι) for the Adaptive Knowledge Lifecycle.
//!
//! ι ∈ [0, 100]. Access: +3, Update: +5, Daily decay: ×0.995.
//! Bounded to prevent unbounded accumulation.

/// Daily decay factor applied to importance scores.
const DAILY_DECAY_FACTOR: f32 = 0.995;

/// Maximum importance score.
const MAX_IMPORTANCE: f32 = 100.0;

/// Minimum importance score.
const MIN_IMPORTANCE: f32 = 0.0;

/// Boost applied on each read access.
const ACCESS_BOOST: f32 = 3.0;

/// Boost applied on each update/write.
const UPDATE_BOOST: f32 = 5.0;

/// Tracks and manages the importance score for a single memory cell.
///
/// Uses the Observer pattern: callers notify the scorer of access/update events,
/// and the scorer maintains the bounded ι value.
#[derive(Debug, Clone)]
pub struct ImportanceScorer {
    importance: f32,
}

impl ImportanceScorer {
    /// Create a new scorer with an initial importance value.
    #[must_use]
    pub fn new(initial: f32) -> Self {
        Self { importance: initial.clamp(MIN_IMPORTANCE, MAX_IMPORTANCE) }
    }

    /// Current importance score.
    #[must_use]
    pub fn importance(&self) -> f32 {
        self.importance
    }

    /// Record a read access event (+3).
    pub fn on_access(&mut self) {
        self.importance = (self.importance + ACCESS_BOOST).min(MAX_IMPORTANCE);
    }

    /// Record an update/write event (+5).
    pub fn on_update(&mut self) {
        self.importance = (self.importance + UPDATE_BOOST).min(MAX_IMPORTANCE);
    }

    /// Apply `n` days of decay (ι × 0.995^n).
    ///
    /// Days beyond 10,000 are clamped (the result is effectively 0 after ~4600 days).
    ///
    /// # Panics
    /// Cannot panic in practice: the `i32::try_from` is on the clamped value
    /// (≤ 10,000), which always fits in `i32`. The `.expect()` is a defensive
    /// guard against future refactors that change the clamp constant.
    pub fn apply_daily_decay(&mut self, days: u32) {
        let clamped = days.min(10_000);
        // Safe: clamped ≤ 10_000, which always fits in i32.
        let exponent = i32::try_from(clamped).expect("clamped to 10_000");
        self.importance = (self.importance * DAILY_DECAY_FACTOR.powi(exponent)).max(MIN_IMPORTANCE);
    }

    /// Check if this cell is below the eviction threshold.
    #[must_use]
    pub fn is_evictable(&self, threshold: f32) -> bool {
        self.importance < threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_clamping() {
        assert!((ImportanceScorer::new(150.0).importance() - 100.0).abs() < f32::EPSILON);
        assert!((ImportanceScorer::new(-10.0).importance() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_access_boost() {
        let mut s = ImportanceScorer::new(0.0);
        s.on_access();
        assert!((s.importance() - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_update_boost() {
        let mut s = ImportanceScorer::new(0.0);
        s.on_update();
        assert!((s.importance() - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_max_cap() {
        let mut s = ImportanceScorer::new(99.0);
        s.on_update(); // 99 + 5 = 104 → capped at 100
        assert!((s.importance() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_daily_decay() {
        let mut s = ImportanceScorer::new(100.0);
        s.apply_daily_decay(1);
        assert!((s.importance() - 99.5).abs() < 0.01);
    }

    #[test]
    fn test_evictable() {
        let s = ImportanceScorer::new(3.0);
        assert!(s.is_evictable(5.0));
        assert!(!s.is_evictable(2.0));
    }

    /// Numerical-precision AT for [`apply_daily_decay`].
    ///
    /// Guards the policy decision (workspace `Cargo.toml` `[lints]`) to allow
    /// `cast_precision_loss` workspace-wide: the f32 decay accumulator must
    /// track the analytical formula `ι₀ × 0.995^n` to within tier-relevant
    /// precision (the smallest hysteresis band is 5 importance points). If a
    /// future refactor switches to a numerically-unstable formulation, this
    /// AT fails before any production deployment regression.
    ///
    /// Sampled across the full lifetime of an importance score (1 day to the
    /// 10 000-day clamp). At 10 000 days the score is effectively zero, so
    /// the absolute-tolerance bar relaxes but the relative bar still holds.
    #[test]
    fn daily_decay_tracks_analytical_formula() {
        // (days, analytical_value). Reference values computed independently
        // in Python: `100.0 * (0.995 ** days)`.
        let cases: &[(u32, f64)] = &[
            (0, 1.000_000e2),
            (1, 9.950_000e1),
            (7, 9.655_206e1),
            (30, 8.603_842e1),
            (90, 6.369_088e1),
            (365, 1.604_813e1),
            (1_000, 6.653_969e-1),
            (5_000, 1.304_379e-9),
        ];

        for &(days, expected) in cases {
            let mut s = ImportanceScorer::new(100.0);
            s.apply_daily_decay(days);
            let got = f64::from(s.importance());
            // Tier hysteresis bands are ≥ 5 points; we hold a tighter
            // bar so the test fails on real numerical drift, not noise.
            let tolerance = (expected.abs() * 1e-3).max(1e-6);
            assert!(
                (got - expected).abs() < tolerance,
                "decay at day {days}: expected {expected:.6}, got {got:.6} \
                 (tolerance {tolerance:.6})",
            );
        }

        // Saturation at the 10 000-day clamp: anything past that floor
        // must stay equal (the clamp is the contract).
        let mut clamped = ImportanceScorer::new(100.0);
        clamped.apply_daily_decay(10_000);
        let mut past_clamp = ImportanceScorer::new(100.0);
        past_clamp.apply_daily_decay(50_000);
        assert!(
            (clamped.importance() - past_clamp.importance()).abs() < f32::EPSILON,
            "decay must saturate at the 10 000-day clamp; got {} vs {}",
            clamped.importance(),
            past_clamp.importance(),
        );
    }
}
