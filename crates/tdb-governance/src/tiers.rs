//! Maturity tier state machine with hysteresis.
//!
//! Three tiers: Draft → Validated → Core.
//! Transitions use hysteresis gaps to prevent rapid oscillation:
//! - Draft → Validated: ι ≥ 65 (demote back at ι < 35, gap = 30)
//! - Validated → Core: ι ≥ 85 (demote back at ι < 60, gap = 25)

use tdb_core::Tier;

// Promotion thresholds.
const PROMOTE_TO_VALIDATED: f32 = 65.0;
const PROMOTE_TO_CORE: f32 = 85.0;

// Demotion thresholds (with hysteresis gap).
const DEMOTE_FROM_VALIDATED: f32 = 35.0;
const DEMOTE_FROM_CORE: f32 = 60.0;

/// State machine managing maturity tier transitions.
///
/// Implements the State pattern: the current tier determines which
/// transitions are valid, and hysteresis gaps prevent oscillation.
#[derive(Debug, Clone)]
pub struct TierStateMachine {
    current: Tier,
}

impl TierStateMachine {
    pub fn new() -> Self {
        Self { current: Tier::Draft }
    }

    /// Create a state machine starting at a specific tier.
    pub fn with_tier(tier: Tier) -> Self {
        Self { current: tier }
    }

    /// Current tier.
    pub fn current(&self) -> Tier {
        self.current
    }

    /// Evaluate the importance score and transition tiers if thresholds are crossed.
    ///
    /// Promotion checks go upward (Draft→Validated→Core).
    /// Demotion checks go downward (Core→Validated→Draft).
    pub fn evaluate(&mut self, importance: f32) {
        match self.current {
            Tier::Draft => {
                if importance >= PROMOTE_TO_VALIDATED {
                    self.current = Tier::Validated;
                    // Check if we should promote further in the same evaluation.
                    if importance >= PROMOTE_TO_CORE {
                        self.current = Tier::Core;
                    }
                }
            }
            Tier::Validated => {
                if importance >= PROMOTE_TO_CORE {
                    self.current = Tier::Core;
                } else if importance < DEMOTE_FROM_VALIDATED {
                    self.current = Tier::Draft;
                }
            }
            Tier::Core => {
                if importance < DEMOTE_FROM_CORE {
                    self.current = Tier::Validated;
                    // Check if we should demote further.
                    if importance < DEMOTE_FROM_VALIDATED {
                        self.current = Tier::Draft;
                    }
                }
            }
        }
    }
}

impl Default for TierStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starts_at_draft() {
        let sm = TierStateMachine::new();
        assert_eq!(sm.current(), Tier::Draft);
    }

    #[test]
    fn test_promote_draft_to_validated() {
        let mut sm = TierStateMachine::new();
        sm.evaluate(64.9);
        assert_eq!(sm.current(), Tier::Draft, "Should not promote at 64.9");
        sm.evaluate(65.0);
        assert_eq!(sm.current(), Tier::Validated);
    }

    #[test]
    fn test_promote_validated_to_core() {
        let mut sm = TierStateMachine::with_tier(Tier::Validated);
        sm.evaluate(84.9);
        assert_eq!(sm.current(), Tier::Validated, "Should not promote at 84.9");
        sm.evaluate(85.0);
        assert_eq!(sm.current(), Tier::Core);
    }

    #[test]
    fn test_demote_core_to_validated() {
        let mut sm = TierStateMachine::with_tier(Tier::Core);
        sm.evaluate(60.0);
        assert_eq!(sm.current(), Tier::Core, "Should not demote at 60.0 (threshold is <60)");
        sm.evaluate(59.9);
        assert_eq!(sm.current(), Tier::Validated);
    }

    #[test]
    fn test_demote_validated_to_draft() {
        let mut sm = TierStateMachine::with_tier(Tier::Validated);
        sm.evaluate(35.0);
        assert_eq!(sm.current(), Tier::Validated, "Should not demote at 35.0 (threshold is <35)");
        sm.evaluate(34.9);
        assert_eq!(sm.current(), Tier::Draft);
    }

    #[test]
    fn test_hysteresis_prevents_oscillation() {
        let mut sm = TierStateMachine::new();
        sm.evaluate(65.0); // Draft → Validated
        assert_eq!(sm.current(), Tier::Validated);

        // Hovering just above demotion threshold.
        for _ in 0..10 {
            sm.evaluate(36.0);
            assert_eq!(sm.current(), Tier::Validated, "Should remain Validated at 36.0");
        }

        // Drop below demotion threshold.
        sm.evaluate(34.0);
        assert_eq!(sm.current(), Tier::Draft);

        // Hovering just below promotion threshold — should stay Draft.
        sm.evaluate(64.0);
        assert_eq!(sm.current(), Tier::Draft);
    }

    #[test]
    fn test_skip_tier_promotion() {
        // If importance is high enough, should promote directly Draft → Core.
        let mut sm = TierStateMachine::new();
        sm.evaluate(90.0);
        assert_eq!(sm.current(), Tier::Core);
    }

    #[test]
    fn test_skip_tier_demotion() {
        // If importance drops very low, should demote Core → Draft.
        let mut sm = TierStateMachine::with_tier(Tier::Core);
        sm.evaluate(10.0);
        assert_eq!(sm.current(), Tier::Draft);
    }
}
