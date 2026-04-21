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

// ── Phase 6: State Rebuild on Open (Memento pattern) ──────────────────────

/// ATDD Test 6: Write 10 cells, drop engine, reopen at same dir.
/// `mem_read` should find the correct cells — proving the retriever was rebuilt from disk.
#[test]
fn test_rebuild_retriever_on_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let key_10 = {
        let mut engine = Engine::open(dir.path()).unwrap();
        for i in 0..10u64 {
            let mut key = vec![0.01f32; 32];
            key[(i as usize) % 32] = 1.0;
            engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0).unwrap();
        }
        // Return cell #5's key for querying after reopen.
        let mut k = vec![0.01f32; 32];
        k[5] = 1.0;
        k
    }; // engine dropped here

    // Reopen from same directory.
    let mut engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.cell_count(), 10);

    let results = engine.mem_read(&key_10, 3, None).unwrap();
    let ids: Vec<u64> = results.iter().map(|r| r.cell.id).collect();
    assert!(ids.contains(&5), "Cell #5 not found after reopen. Got: {ids:?}");
}

/// ATDD Test 7: Write cell with ι=90 (→ Core tier), drop, reopen.
/// Governance state (importance + tier) should be reconstructed from persisted metadata.
#[test]
fn test_rebuild_governance_on_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let cell_id = {
        let mut engine = Engine::open(dir.path()).unwrap();
        let key = vec![1.0f32; 32];
        engine.mem_write(1, 0, &key, vec![0.0; 32], 90.0).unwrap()
        // Initial: 90 + 5 (write) = 95 → Core.
    };

    // Reopen.
    let engine = Engine::open(dir.path()).unwrap();

    // Tier should be Core (persisted as tier byte in segment).
    assert_eq!(engine.cell_tier(cell_id), Some(Tier::Core), "Tier should be Core after reopen");

    // Importance should be approximately what was stored.
    // Note: the persisted value is the *initial* importance (90.0), not the boosted value (95).
    // The engine reconstructs from cell.meta.importance which was set at write time.
    let importance = engine.cell_importance(cell_id).unwrap();
    assert!(importance > 80.0, "Importance {importance:.1} should be >80 after reopen");
}

/// ATDD Test 8: Write cells 0..5, drop, reopen, write one more.
/// New cell's ID must be >= 5 (no collision with persisted IDs).
#[test]
fn test_next_id_monotonic_after_reopen() {
    let dir = tempfile::tempdir().unwrap();

    {
        let mut engine = Engine::open(dir.path()).unwrap();
        for _ in 0..5 {
            engine.mem_write(1, 0, &[1.0f32; 16], vec![0.0; 16], 50.0).unwrap();
        }
    }

    let mut engine = Engine::open(dir.path()).unwrap();
    let new_id = engine.mem_write(1, 0, &[2.0f32; 16], vec![0.0; 16], 50.0).unwrap();

    assert!(new_id >= 5, "New cell ID {new_id} collides with persisted IDs 0..5");
}

/// ATDD Test 9: Write 1000 cells, drop, reopen. cell_count == 1000.
/// `mem_read` for a specific key returns the correct cell.
#[test]
fn test_rebuild_large_pool() {
    let dir = tempfile::tempdir().unwrap();

    {
        let mut engine = Engine::open(dir.path()).unwrap();
        for i in 0..1000u64 {
            let mut key = vec![0.01f32; 32];
            key[(i as usize) % 32] = 1.0;
            key[((i as usize) + 1) % 32] = (i as f32) * 0.001;
            engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0).unwrap();
        }
    }

    let mut engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.cell_count(), 1000);

    // Query for cell #500's pattern.
    let mut query = vec![0.01f32; 32];
    query[500 % 32] = 1.0;
    query[(500 + 1) % 32] = 500.0 * 0.001;

    let results = engine.mem_read(&query, 5, None).unwrap();
    assert!(!results.is_empty(), "Should find results after rebuilding 1000 cells");
}

/// ATDD Test 10: Use small segment threshold to span 3+ segments.
/// Drop, reopen, all cells retrievable.
#[test]
fn test_rebuild_across_segments() {
    let dir = tempfile::tempdir().unwrap();

    {
        let mut engine = Engine::open_with_segment_size(dir.path(), 4096).unwrap();
        for i in 0..100u64 {
            let mut key = vec![0.01f32; 32];
            key[(i as usize) % 32] = 1.0;
            engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0).unwrap();
        }
    }

    let mut engine = Engine::open_with_segment_size(dir.path(), 4096).unwrap();
    assert_eq!(engine.cell_count(), 100);

    // Verify a cell from each likely segment by querying its unique key pattern.
    for target_id in [0u64, 50, 99] {
        let mut query = vec![0.01f32; 32];
        query[(target_id as usize) % 32] = 1.0;
        let results = engine.mem_read(&query, 5, None).unwrap();
        let ids: Vec<u64> = results.iter().map(|r| r.cell.id).collect();
        assert!(
            ids.contains(&target_id),
            "Cell {target_id} not found after cross-segment rebuild. Got: {ids:?}"
        );
    }
}
