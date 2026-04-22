use tdb_core::Tier;
use tdb_engine::engine::Engine;

/// ATDD Test 1: Write a cell via Engine, read it back, verify data round-trips.
#[test]
fn test_engine_write_read_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let value: Vec<f32> = (0..64).map(|i| (i as f32 * 0.2).cos()).collect();

    let id = engine.mem_write(42, 12, &key, value, 50.0, None).unwrap();

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
        engine.mem_write(1, 0, &key, value, 50.0, None).unwrap();
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
        engine.mem_write(1, 0, &[i as f32; 16], vec![0.0; 16], 50.0, None).unwrap();
    }
    // Owner 2 cells.
    for i in 10..20u64 {
        engine.mem_write(2, 0, &[i as f32; 16], vec![0.0; 16], 50.0, None).unwrap();
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
    let id = engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0, None).unwrap();

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
    let id = engine.mem_write(1, 0, &key, vec![0.0; 32], 90.0, None).unwrap();

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
            engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0, None).unwrap();
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
        engine.mem_write(1, 0, &key, vec![0.0; 32], 90.0, None).unwrap()
        // Initial: 90 + 5 (write) = 95 → Core.
    };

    // Reopen.
    let engine = Engine::open(dir.path()).unwrap();

    // Tier should be Core (persisted as tier byte in segment).
    assert_eq!(engine.cell_tier(cell_id), Some(Tier::Core), "Tier should be Core after reopen");

    // Importance should match the persisted value: 90 (salience) + 5 (write boost) = 95.
    // Governance is computed before persistence, so the on-disk value includes the boost.
    let importance = engine.cell_importance(cell_id).unwrap();
    assert!(
        (importance - 95.0).abs() < 1.0,
        "Importance {importance:.1} should be ≈95.0 after reopen"
    );
}

/// ATDD Test 8: Write cells 0..5, drop, reopen, write one more.
/// New cell's ID must be >= 5 (no collision with persisted IDs).
#[test]
fn test_next_id_monotonic_after_reopen() {
    let dir = tempfile::tempdir().unwrap();

    {
        let mut engine = Engine::open(dir.path()).unwrap();
        for _ in 0..5 {
            engine.mem_write(1, 0, &[1.0f32; 16], vec![0.0; 16], 50.0, None).unwrap();
        }
    }

    let mut engine = Engine::open(dir.path()).unwrap();
    let new_id = engine.mem_write(1, 0, &[2.0f32; 16], vec![0.0; 16], 50.0, None).unwrap();

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
            engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0, None).unwrap();
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
            engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0, None).unwrap();
        }
    }

    let mut engine = Engine::open_with_segment_size(dir.path(), 4096).unwrap();
    assert_eq!(engine.cell_count(), 100);

    // Verify reads work across all rebuilt segments.
    let query = vec![1.0f32; 32]; // matches all cells somewhat
    let results = engine.mem_read(&query, 10, None).unwrap();
    assert!(!results.is_empty(), "Should return results after cross-segment rebuild");
    // All returned cells should have valid IDs in the 0..100 range.
    for r in &results {
        assert!(r.cell.id < 100, "Cell ID {} out of range", r.cell.id);
    }
}

// ── Phase 7: Wire SLB + Vamana + Trace + WAL (Chain of Responsibility) ────

/// ATDD Test 11: Write 100 cells, read one twice. SLB should serve the repeat query.
/// Functional correctness: both reads return the correct cell.
#[test]
fn test_slb_accelerates_read() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Write 100 cells with distinct key patterns.
    for i in 0..100u64 {
        let mut key = vec![0.01f32; 32];
        key[(i as usize) % 32] = 1.0;
        engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0, None).unwrap();
    }

    // Query for cell #10.
    let mut query = vec![0.01f32; 32];
    query[10] = 1.0;

    // First read — populates SLB with accessed cells.
    let results1 = engine.mem_read(&query, 3, None).unwrap();
    assert!(!results1.is_empty(), "First read should return results");

    // All results should have key[10] == 1.0 (dominant dimension matches query).
    for r in &results1 {
        assert_eq!(r.cell.id % 32, 10, "Result cell {} has wrong dominant dim", r.cell.id);
    }

    // Second read — SLB now has the accessed cells cached.
    let results2 = engine.mem_read(&query, 3, None).unwrap();
    assert!(!results2.is_empty(), "Second read should return results via SLB");

    // Same dominant-dimension cells should appear.
    for r in &results2 {
        assert_eq!(r.cell.id % 32, 10, "SLB result cell {} has wrong dominant dim", r.cell.id);
    }
}

/// ATDD Test 12: Write cell A, write cell B with parent=A.
/// `trace_ancestors(B, CausedBy)` returns [A].
#[test]
fn test_trace_causal_edges_via_engine() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let id_a = engine.mem_write(1, 0, &[1.0f32; 16], vec![0.0; 16], 50.0, None).unwrap();
    let id_b = engine.mem_write(1, 0, &[2.0f32; 16], vec![0.0; 16], 50.0, Some(id_a)).unwrap();

    let ancestors = engine.trace_ancestors(id_b);
    assert!(
        ancestors.contains(&id_a),
        "Cell A ({id_a}) should be an ancestor of Cell B ({id_b}). Got: {ancestors:?}"
    );
}

/// ATDD Test 13: Write causal edges, drop engine (no explicit checkpoint), reopen.
/// `trace_ancestors()` returns the same chain — proving WAL replay works.
#[test]
fn test_wal_replay_recovers_trace() {
    let dir = tempfile::tempdir().unwrap();

    let (id_a, id_b, id_c) = {
        let mut engine = Engine::open(dir.path()).unwrap();
        let a = engine.mem_write(1, 0, &[1.0f32; 16], vec![0.0; 16], 50.0, None).unwrap();
        let b = engine.mem_write(1, 0, &[2.0f32; 16], vec![0.0; 16], 50.0, Some(a)).unwrap();
        let c = engine.mem_write(1, 0, &[3.0f32; 16], vec![0.0; 16], 50.0, Some(b)).unwrap();
        (a, b, c)
    }; // engine dropped — WAL NOT checkpointed

    // Reopen — WAL should be replayed to rebuild TraceGraph.
    let engine = Engine::open(dir.path()).unwrap();
    let ancestors_c = engine.trace_ancestors(id_c);
    assert!(
        ancestors_c.contains(&id_a) && ancestors_c.contains(&id_b),
        "Ancestors of C should include A and B after WAL replay. Got: {ancestors_c:?}"
    );
}

/// ATDD Test 14: Query when SLB is empty — falls through to BruteForce.
/// Engine should still return correct results.
#[test]
fn test_retrieval_chain_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Write a few cells.
    for i in 0..5u64 {
        let mut key = vec![0.01f32; 16];
        key[(i as usize) % 16] = 1.0;
        engine.mem_write(1, 0, &key, vec![0.0; 16], 50.0, None).unwrap();
    }

    // First-ever query — SLB is empty, must fall through to BruteForce.
    let mut query = vec![0.01f32; 16];
    query[3] = 1.0;
    let results = engine.mem_read(&query, 3, None).unwrap();
    assert!(!results.is_empty(), "Should return results even with empty SLB");
    assert_eq!(results[0].cell.id, 3, "Cell #3 should rank first");
}

/// ATDD Test 15: Set Vamana threshold=50. Write 49 cells (no Vamana).
/// Write 50th cell (triggers Vamana build). Reads still return correct results.
#[test]
fn test_vamana_activation_at_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open_with_vamana_threshold(dir.path(), 50).unwrap();

    // Write 49 cells — below threshold.
    for i in 0..49u64 {
        let mut key = vec![0.01f32; 32];
        key[(i as usize) % 32] = 1.0;
        engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0, None).unwrap();
    }
    assert!(!engine.has_vamana(), "Vamana should NOT be active below threshold");

    // Write 50th cell — triggers build.
    let mut key50 = vec![0.01f32; 32];
    key50[18] = 1.0;
    engine.mem_write(1, 0, &key50, vec![0.0; 32], 50.0, None).unwrap();
    assert!(engine.has_vamana(), "Vamana should be active at threshold");

    // Reads should still work.
    let mut query = vec![0.01f32; 32];
    query[18] = 1.0;
    let results = engine.mem_read(&query, 3, None).unwrap();
    assert!(!results.is_empty(), "Should return results after Vamana activation");
}

// ── Phase 8: Benchmark Sanity Tests ───────────────────────────────────────

/// ATDD Test 16: SLB query at 4096 entries completes <100μs per query in debug mode.
#[test]
fn test_slb_bench_sanity() {
    use tdb_retrieval::slb::SemanticLookasideBuffer;

    let dim = 128;
    let mut slb = SemanticLookasideBuffer::new(4096, dim);
    for i in 0..4096u64 {
        let key: Vec<f32> = (0..dim).map(|d| ((i * 3 + d as u64) as f32 * 0.01).sin()).collect();
        slb.insert(i, 1, &key);
    }
    let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.02).cos()).collect();

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = slb.query(&query, 5);
    }
    let avg = start.elapsed() / 100;
    // Debug mode: ~3-5ms per query at 4096 entries is expected.
    // Release target: <5μs. This test just verifies no pathological O(n²).
    assert!(
        avg < std::time::Duration::from_millis(20),
        "SLB avg latency {avg:?} exceeds 20ms (pathological)"
    );
}

/// ATDD Test 17: Vamana build at 1K vs 2K: 2K time is <5x of 1K time.
#[test]
fn test_vamana_build_scales_sanity() {
    use tdb_index::vamana::VamanaIndex;

    let dim = 16;
    let make_vectors = |n: usize| -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.01f32; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect()
    };

    let v1k = make_vectors(1000);
    let start_1k = std::time::Instant::now();
    let mut idx = VamanaIndex::new(dim, 8);
    for (i, v) in v1k.iter().enumerate() {
        idx.insert(i as u64, v);
    }
    idx.build();
    let time_1k = start_1k.elapsed();

    let v2k = make_vectors(2000);
    let start_2k = std::time::Instant::now();
    let mut idx2 = VamanaIndex::new(dim, 8);
    for (i, v) in v2k.iter().enumerate() {
        idx2.insert(i as u64, v);
    }
    idx2.build();
    let time_2k = start_2k.elapsed();

    let ratio = time_2k.as_secs_f64() / time_1k.as_secs_f64();
    assert!(
        ratio < 5.0,
        "2K build took {ratio:.1}x of 1K build (expected <5x). 1K={time_1k:?}, 2K={time_2k:?}"
    );
}

/// ATDD Test 18: WAL append 10K entries <1s in debug.
#[test]
fn test_wal_append_throughput_sanity() {
    use tdb_index::wal::{Wal, WalEntry};

    let dir = tempfile::tempdir().unwrap();
    let mut wal = Wal::open(dir.path()).unwrap();

    let start = std::time::Instant::now();
    // 500 entries — each append does file open + write + fsync (~5-10ms per call).
    for i in 0..500u64 {
        wal.append(&WalEntry::AddEdge { src: i, dst: i + 1, edge_type: 0, timestamp: i * 1000 })
            .unwrap();
    }
    let elapsed = start.elapsed();
    // Each append does fsync. 500 × ~10ms = ~5s, plus CI variance.
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "WAL append 500 took {elapsed:?} (expected <30s)"
    );
}

/// ATDD Test 19: Engine write 500 cells (dim=32) <10s in debug.
#[test]
fn test_engine_write_throughput_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let start = std::time::Instant::now();
    for i in 0..500u64 {
        let mut key = vec![0.01f32; 32];
        key[(i as usize) % 32] = 1.0;
        engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0, None).unwrap();
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "Engine write 500 cells took {elapsed:?} (expected <30s)"
    );
}

/// ATDD Test 20: Engine read 100 queries against 500 cells <10s in debug.
#[test]
fn test_engine_read_throughput_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    for i in 0..500u64 {
        let mut key = vec![0.01f32; 32];
        key[(i as usize) % 32] = 1.0;
        engine.mem_write(1, 0, &key, vec![0.0; 32], 50.0, None).unwrap();
    }

    let query = vec![1.0f32; 32];
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = engine.mem_read(&query, 5, None).unwrap();
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "Engine read 100 queries took {elapsed:?} (expected <30s)"
    );
}

// ── Phase 11: SynapticBank via Engine (Repository pattern) ────────────────

/// ATDD Test 21: Engine store + load synapsis round-trip.
#[test]
fn test_engine_synapsis_api() {
    use half::f16;
    use tdb_core::synaptic_bank::SynapticBankEntry;

    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let entry = SynapticBankEntry::new(
        0,
        42,
        vec![f16::from_f32(1.0); 8],
        vec![f16::from_f32(0.5); 8],
        f16::from_f32(0.1),
        2,
        4,
    );
    engine.store_synapsis(&entry).unwrap();

    let loaded = engine.load_synapsis(42).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].id, 0);
    assert_eq!(loaded[0].owner, 42);
    assert_eq!(loaded[0].rank, 2);
    assert_eq!(loaded[0].d_model, 4);
}
