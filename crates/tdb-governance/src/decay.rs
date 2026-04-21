//! Recency decay function for the AKL.
//!
//! `r = exp(-Δt / τ)`, where `τ = 30` days (~21-day half-life).
//! Applied as a multiplier during retrieval scoring to down-weight stale cells.

/// Time constant τ in days. Controls decay rate.
/// Half-life = τ × ln(2) ≈ 20.8 days.
const TAU_DAYS: f32 = 30.0;

/// Compute the recency decay factor for a cell.
///
/// Returns a value in (0.0, 1.0]:
/// - `r = 1.0` when `days_since_update = 0` (just updated)
/// - `r ≈ 0.368` when `days_since_update = τ` (30 days)
/// - `r ≈ 0.5` when `days_since_update ≈ 20.8` (half-life)
///
/// Negative inputs are clamped to 0 (returns 1.0).
pub fn recency_decay(days_since_update: f32) -> f32 {
    let dt = days_since_update.max(0.0);
    (-dt / TAU_DAYS).exp()
}

/// Compute a retrieval score adjusted for recency.
///
/// `adjusted_score = raw_score × recency_decay(days_since_update)`
pub fn decay_adjusted_score(raw_score: f32, days_since_update: f32) -> f32 {
    raw_score * recency_decay(days_since_update)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_days() {
        assert!((recency_decay(0.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tau_days() {
        let r = recency_decay(30.0);
        let expected = (-1.0f32).exp();
        assert!((r - expected).abs() < 0.001);
    }

    #[test]
    fn test_negative_clamped() {
        assert!((recency_decay(-5.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_monotonically_decreasing() {
        let r1 = recency_decay(10.0);
        let r2 = recency_decay(20.0);
        let r3 = recency_decay(30.0);
        assert!(r1 > r2);
        assert!(r2 > r3);
    }

    #[test]
    fn test_decay_adjusted_score() {
        let raw = 10.0;
        let adjusted = decay_adjusted_score(raw, 30.0);
        let expected = raw * (-1.0f32).exp();
        assert!((adjusted - expected).abs() < 0.01);
    }

    #[test]
    fn test_very_old_approaches_zero() {
        let r = recency_decay(365.0); // 1 year
        assert!(r < 0.001, "Decay at 365 days should be near zero, got {r}");
    }
}
