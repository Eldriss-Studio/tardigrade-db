//! Maturity tier state machine with hysteresis.
//!
//! Draft → Validated at ι ≥ 65 (demote < 35, gap = 30)
//! Validated → Core at ι ≥ 85 (demote < 60, gap = 25)
