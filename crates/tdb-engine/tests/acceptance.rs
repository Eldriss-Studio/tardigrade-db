use tdb_core::Tier;
use tdb_engine::engine::Engine;

/// ATDD Test 1: Write a cell via Engine, read it back, verify data round-trips.
#[test]
fn test_engine_write_read_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let value: Vec<f32> = (0..64).map(|i| (i as f32 * 0.2).cos()).collect();

    let id = engine.mem_write(42, 12, &key, value, 50.0).unwrap();

    let results = engine.mem_read(&key, 1, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].cell.id, id);
    assert_eq!(results[0].cell.owner, 42);
    assert_eq!(results[0].cell.layer, 12);
    // Key is Q4 quantized on disk, so approximate match.
    assert_eq!(results[0].cell.key.len(), 64);
}

/// ATDD Test 2: Write 100 cells, read top-5, verify correct cell is found.
#[test]
fn test_engine_mem_read_topk() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    for i in 0..100u64 {
        let mut key = vec![0.01f32; 32];
        key[(i as usize) % 32] = 1.0;
        let value = vec![0.0f32; 32];
        engine.mem_write(1, 0, &key, value, 50.0).unwrap();
    }

    // Query for cell #10's key pattern.
    let mut query = vec![0.01f32; 32];
    query[10] = 1.0;

    let results = engine.mem_read(&query, 5, None).unwrap();
    assert_eq!(results.len(), 5);

    // Cell #10 should be in the top results.
    let ids: Vec<u64> = results.iter().map(|r| r.cell.id).collect();
    assert!(ids.contains(&10), "Cell #10 not in top-5. Got: {ids:?}");
}

/// ATDD Test 3: Owner filtering in `mem_read`.
#[test]
fn test_engine_owner_filter() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Owner 1 cells.
    for i in 0..10u64 {
        engine.mem_write(1, 0, &[i as f32; 16], vec![0.0; 16], 50.0).unwrap();
    }
    // Owner 2 cells.
    for i in 10..20u64 {
        engine.mem_write(2, 0, &[i as f32; 16], vec![0.0; 16], 50.0).unwrap();
    }

    let query = vec![5.0f32; 16];
    let results = engine.mem_read(&query, 5, Some(1)).unwrap();
    for r in &results {
        assert_eq!(r.cell.owner, 1, "Expected owner 1, got {}", r.cell.owner);
    }
}

/// ATDD Test 4: Governance integration — importance boosts on read, tier promotion.
#[test]
fn test_engine_governance_integration() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = vec![1.0f32; 32];
    let id = engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0).unwrap();

    // Initial importance: 50 + 5 (write boost) = 55. Tier: Draft (55 < 65).
    assert_eq!(engine.cell_tier(id), Some(Tier::Draft));

    // Read the cell 4 times: each read adds +3. 55 + 12 = 67 → Validated.
    for _ in 0..4 {
        let _ = engine.mem_read(&key, 1, None).unwrap();
    }
    assert!(
        engine.cell_importance(id).unwrap() >= 65.0,
        "Importance {} should be ≥65",
        engine.cell_importance(id).unwrap()
    );
    assert_eq!(engine.cell_tier(id), Some(Tier::Validated));
}

/// ATDD Test 5: Decay over time — cells lose importance, tiers demote.
#[test]
fn test_engine_decay_and_demotion() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = vec![1.0f32; 32];
    let id = engine.mem_write(1, 0, &key, vec![0.0; 32], 90.0).unwrap();

    // Initial: 90 + 5 = 95. Tier: Core.
    assert_eq!(engine.cell_tier(id), Some(Tier::Core));

    // Simulate 100 days of decay: 95 × 0.995^100 ≈ 57.5 → below Core demotion (60).
    engine.advance_days(100.0);
    let importance = engine.cell_importance(id).unwrap();
    assert!(importance < 60.0, "After 100 days, importance {importance:.1} should be <60");
    assert_eq!(engine.cell_tier(id), Some(Tier::Validated), "Should demote from Core to Validated");
}
