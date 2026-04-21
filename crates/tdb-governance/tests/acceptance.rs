use tdb_core::Tier;
use tdb_governance::decay::recency_decay;
use tdb_governance::scoring::ImportanceScorer;
use tdb_governance::tiers::TierStateMachine;

/// ATDD Test 1: Create a cell with ι=50. Read it 5 times. Assert ι = 50 + (5 × 3) = 65.
#[test]
fn test_importance_access_boost() {
    let mut scorer = ImportanceScorer::new(50.0);

    for _ in 0..5 {
        scorer.on_access();
    }

    let expected = 65.0;
    assert!(
        (scorer.importance() - expected).abs() < f32::EPSILON,
        "Expected ι={expected}, got ι={}",
        scorer.importance()
    );
}

/// ATDD Test 2: Draft→Validated at ι=65. Stays Validated at ι=36 (above demotion threshold 35).
/// Demotes back to Draft at ι=34.
#[test]
fn test_tier_hysteresis() {
    let mut tier_sm = TierStateMachine::new();
    assert_eq!(tier_sm.current(), Tier::Draft);

    // Promote to Validated at ι=65.
    tier_sm.evaluate(65.0);
    assert_eq!(tier_sm.current(), Tier::Validated, "Should promote to Validated at ι=65");

    // Stay Validated at ι=36 (above demotion threshold of 35).
    tier_sm.evaluate(36.0);
    assert_eq!(tier_sm.current(), Tier::Validated, "Should stay Validated at ι=36 (above 35)");

    // Demote to Draft at ι=34 (below demotion threshold of 35).
    tier_sm.evaluate(34.0);
    assert_eq!(tier_sm.current(), Tier::Draft, "Should demote to Draft at ι=34 (below 35)");
}

/// ATDD Test 3: Cell updated 30 days ago. Assert recency r ≈ exp(-1) ≈ 0.368.
/// Cell updated today. Assert r ≈ 1.0.
#[test]
fn test_recency_decay_30_day() {
    // τ = 30 days. At Δt = 30 days: r = exp(-30/30) = exp(-1) ≈ 0.3679.
    let r_30_days = recency_decay(30.0);
    let expected = (-1.0f64).exp() as f32;
    assert!(
        (r_30_days - expected).abs() < 0.001,
        "At 30 days: expected r≈{expected:.4}, got r={r_30_days:.4}"
    );

    // At Δt = 0 days: r = exp(0) = 1.0.
    let r_now = recency_decay(0.0);
    assert!((r_now - 1.0).abs() < f32::EPSILON, "At 0 days: expected r=1.0, got r={r_now:.4}");

    // Half-life check: at Δt ≈ 20.79 days, r ≈ 0.5.
    let half_life = 30.0 * (2.0f32).ln();
    let r_half = recency_decay(half_life);
    assert!(
        (r_half - 0.5).abs() < 0.01,
        "At half-life ({half_life:.1} days): expected r≈0.5, got r={r_half:.4}"
    );
}

/// ATDD Test 4: Create 100 Draft cells with ι=10. Simulate 200 days of daily decay
/// (ι × 0.995^200 ≈ 3.6). Assert all cells are below the eviction threshold (ι<5).
#[test]
fn test_sweep_evicts_stale() {
    let mut scorers: Vec<ImportanceScorer> =
        (0..100).map(|_| ImportanceScorer::new(10.0)).collect();

    // Apply 200 days of decay.
    for scorer in &mut scorers {
        scorer.apply_daily_decay(200);
    }

    // After 200 days: 10.0 × 0.995^200 ≈ 3.66.
    let expected_approx = 10.0f32 * 0.995f32.powi(200);
    for (i, scorer) in scorers.iter().enumerate() {
        assert!(
            scorer.importance() < 5.0,
            "Cell {i}: ι={:.2} should be below eviction threshold 5.0 (expected ≈{expected_approx:.2})",
            scorer.importance()
        );
    }
}

/// ATDD Test 5: Core cell (ι=90) survives 50 days of decay (ι≈69.8, above Core demotion at 60).
/// Draft cell (ι=10) after 200 days decays to ι≈3.7, below eviction threshold (5).
#[test]
fn test_core_survives_pressure() {
    // --- Core cell: ι=90, 50 days of decay → ι ≈ 69.8 (still Core) ---
    let mut core_scorer = ImportanceScorer::new(90.0);
    let mut core_tier = TierStateMachine::new();
    core_tier.evaluate(90.0); // Draft → Core (skip-promotion at ι≥85)
    assert_eq!(core_tier.current(), Tier::Core);

    core_scorer.apply_daily_decay(50);
    // 90.0 × 0.995^50 ≈ 69.8
    let core_importance = core_scorer.importance();
    assert!(
        core_importance > 60.0,
        "Core ι={core_importance:.1} should be above demotion threshold 60"
    );
    core_tier.evaluate(core_importance);
    assert_eq!(
        core_tier.current(),
        Tier::Core,
        "Core cell at ι={core_importance:.1} should remain Core"
    );

    // --- Draft cell: ι=10, 200 days of decay → ι ≈ 3.7 (evictable) ---
    let mut draft_scorer = ImportanceScorer::new(10.0);
    let mut draft_tier = TierStateMachine::new();
    draft_tier.evaluate(10.0);
    assert_eq!(draft_tier.current(), Tier::Draft);

    draft_scorer.apply_daily_decay(200);
    // 10.0 × 0.995^200 ≈ 3.66
    let draft_importance = draft_scorer.importance();
    assert!(
        draft_scorer.is_evictable(5.0),
        "Draft ι={draft_importance:.2} should be below eviction threshold 5.0"
    );
    draft_tier.evaluate(draft_importance);
    assert_eq!(draft_tier.current(), Tier::Draft);
}
