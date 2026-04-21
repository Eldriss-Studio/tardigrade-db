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
    pub fn new(initial: f32) -> Self {
        Self { importance: initial.clamp(MIN_IMPORTANCE, MAX_IMPORTANCE) }
    }

    /// Current importance score.
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
    pub fn apply_daily_decay(&mut self, days: u32) {
        let clamped = days.min(10_000);
        // Safe: clamped ≤ 10_000, which always fits in i32.
        let exponent = i32::try_from(clamped).expect("clamped to 10_000");
        self.importance = (self.importance * DAILY_DECAY_FACTOR.powi(exponent)).max(MIN_IMPORTANCE);
    }

    /// Check if this cell is below the eviction threshold.
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
}
