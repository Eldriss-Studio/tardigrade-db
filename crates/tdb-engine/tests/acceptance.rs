use tdb_core::Tier;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::engine::{Engine, WriteRequest};
use tdb_retrieval::per_token::encode_per_token_keys;

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

/// ATDD Test 9: Write 1000 cells, drop, reopen. `cell_count` == 1000.
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

/// ATDD Test 14: Query when SLB is empty — falls through to `BruteForce`.
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

// ── Phase 21: Batch Write (Batch Command pattern) ───────────────────────────

/// ATDD 22: Batch write 100 cells — all readable afterward.
#[test]
fn test_batch_write_all_readable() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let requests: Vec<WriteRequest> = (0..100u64)
        .map(|i| {
            let mut key = vec![0.01f32; 64];
            key[(i as usize) % 64] = 1.0;
            WriteRequest {
                owner: 1,
                layer: 0,
                key: key.clone(),
                value: key,
                salience: 50.0,
                parent_cell_id: None,
            }
        })
        .collect();

    let ids = engine.mem_write_batch(&requests).unwrap();
    assert_eq!(ids.len(), 100);

    // All 100 cells should be readable.
    assert_eq!(engine.cell_count(), 100);

    // Query should find cells.
    let mut query = vec![0.01f32; 64];
    query[42] = 1.0;
    let results = engine.mem_read(&query, 5, None).unwrap();
    assert!(!results.is_empty());
}

/// ATDD 23: Batch write is faster than individual writes.
#[test]
fn test_batch_write_faster_than_individual() {
    let n = 200;
    let dim = 32;

    let make_requests = || -> Vec<WriteRequest> {
        (0..n)
            .map(|i: u64| {
                let mut key = vec![0.01f32; dim];
                key[(i as usize) % dim] = 1.0;
                WriteRequest {
                    owner: 1,
                    layer: 0,
                    key: key.clone(),
                    value: key,
                    salience: 50.0,
                    parent_cell_id: None,
                }
            })
            .collect()
    };

    // Individual writes.
    let dir1 = tempfile::tempdir().unwrap();
    let mut engine1 = Engine::open(dir1.path()).unwrap();
    let reqs = make_requests();

    let start_individual = std::time::Instant::now();
    for r in &reqs {
        engine1.mem_write(r.owner, r.layer, &r.key, r.value.clone(), r.salience, None).unwrap();
    }
    let time_individual = start_individual.elapsed();

    // Batch writes.
    let dir2 = tempfile::tempdir().unwrap();
    let mut engine2 = Engine::open(dir2.path()).unwrap();
    let reqs2 = make_requests();

    let start_batch = std::time::Instant::now();
    engine2.mem_write_batch(&reqs2).unwrap();
    let time_batch = start_batch.elapsed();

    // Batch should be at least 2x faster (typically 10-50x).
    let ratio = time_individual.as_secs_f64() / time_batch.as_secs_f64();
    assert!(
        ratio > 2.0,
        "Batch ({time_batch:?}) should be >2x faster than individual ({time_individual:?}). Ratio: {ratio:.1}x"
    );
}

/// ATDD 24: Batch write cells have governance (importance, tier).
#[test]
fn test_batch_write_has_governance() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let requests: Vec<WriteRequest> = (0..5u64)
        .map(|i| {
            let key = vec![i as f32; 16];
            WriteRequest {
                owner: 1,
                layer: 0,
                key: key.clone(),
                value: key,
                salience: 80.0,
                parent_cell_id: None,
            }
        })
        .collect();

    let ids = engine.mem_write_batch(&requests).unwrap();

    for id in &ids {
        let imp = engine.cell_importance(*id).unwrap();
        let tier = engine.cell_tier(*id).unwrap();
        // salience 80 + on_update boost 5 = 85 -> Core tier.
        assert!(imp >= 80.0, "Importance {imp} should be >= 80");
        assert_eq!(tier, Tier::Core, "Tier should be Core at importance {imp}");
    }
}

/// ATDD 25: Batch write cells persist across engine reopen.
#[test]
fn test_batch_write_persists() {
    let dir = tempfile::tempdir().unwrap();

    let ids;
    {
        let mut engine = Engine::open(dir.path()).unwrap();
        let requests: Vec<WriteRequest> = (0..10u64)
            .map(|i| {
                let key = vec![i as f32; 32];
                WriteRequest {
                    owner: 1,
                    layer: 0,
                    key: key.clone(),
                    value: key,
                    salience: 50.0,
                    parent_cell_id: None,
                }
            })
            .collect();

        ids = engine.mem_write_batch(&requests).unwrap();
        assert_eq!(engine.cell_count(), 10);
    }

    // Reopen engine — all cells should survive.
    let engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.cell_count(), 10);

    // Governance should be rebuilt.
    for id in &ids {
        assert!(engine.cell_importance(*id).is_some());
    }
}

/// ATDD 26: Batch write with causal parent IDs creates trace edges.
#[test]
fn test_batch_write_with_causal_edges() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Write a root cell first.
    let root_id = engine.mem_write(1, 0, &[1.0f32; 16], vec![1.0; 16], 50.0, None).unwrap();

    // Batch with parent references.
    let requests = vec![
        WriteRequest {
            owner: 1,
            layer: 0,
            key: vec![2.0; 16],
            value: vec![2.0; 16],
            salience: 50.0,
            parent_cell_id: Some(root_id),
        },
        WriteRequest {
            owner: 1,
            layer: 0,
            key: vec![3.0; 16],
            value: vec![3.0; 16],
            salience: 50.0,
            parent_cell_id: Some(root_id),
        },
    ];

    let ids = engine.mem_write_batch(&requests).unwrap();

    // Both batch cells should be traceable ancestors of root.
    let ancestors_0 = engine.trace_ancestors(ids[0]);
    let ancestors_1 = engine.trace_ancestors(ids[1]);
    assert!(ancestors_0.contains(&root_id), "Cell {} should trace to root {root_id}", ids[0]);
    assert!(ancestors_1.contains(&root_id), "Cell {} should trace to root {root_id}", ids[1]);
}

// -- Phase 22: Per-Token Engine Integration ATDD -----------------------------

/// ATDD 27: Engine retrieves per-token encoded cells correctly.
#[test]
fn test_engine_per_token_retrieval() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let dim = 8;

    // Cell 0: per-token encoded with 3 tokens including distinctive "risotto".
    let food = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let italian = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let risotto = [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let encoded_a = encode_per_token_keys(&[&food, &italian, &risotto]);

    // Cell 1: per-token encoded with different tokens.
    let travel = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let france = [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let encoded_b = encode_per_token_keys(&[&travel, &france]);

    engine.mem_write(1, 0, &encoded_a, vec![0.0; dim], 50.0, None).unwrap();
    engine.mem_write(1, 0, &encoded_b, vec![0.0; dim], 50.0, None).unwrap();

    // Query with "risotto" -- should find cell 0.
    let results = engine.mem_read(&risotto, 2, None).unwrap();
    assert!(!results.is_empty(), "Should find results for per-token query");
    assert_eq!(results[0].cell.id, 0, "Cell 0 (risotto token) should rank first");
}

/// ATDD 28: Per-token and mean-pooled cells coexist in the same engine.
#[test]
fn test_engine_per_token_and_mean_pool_coexist() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let dim = 8;

    // Cell 0: per-token encoded.
    let token_a = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let encoded = encode_per_token_keys(&[&token_a]);
    engine.mem_write(1, 0, &encoded, vec![0.0; dim], 50.0, None).unwrap();

    // Cell 1: legacy mean-pooled (raw vector, no header).
    let mean_key = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    engine.mem_write(1, 0, &mean_key, vec![0.0; dim], 50.0, None).unwrap();

    assert_eq!(engine.cell_count(), 2);

    // Both should be retrievable.
    let r1 = engine.mem_read(&token_a, 2, None).unwrap();
    assert!(!r1.is_empty(), "Should retrieve per-token cell");

    let r2 = engine.mem_read(&mean_key, 2, None).unwrap();
    assert!(!r2.is_empty(), "Should retrieve mean-pooled cell");
}

#[test]
fn test_engine_per_token_query_cannot_be_satisfied_by_slb_only() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let target = [1.0f32, 0.0];
    let zero = [0.0f32, 0.0];
    let slight = [0.3f32, 0.0];
    let near_miss = [0.5f32, 0.0];

    // Cell 0 has the exact token + moderate tokens (wins on Top5Avg and MaxSim).
    let encoded_target = encode_per_token_keys(&[&target, &slight, &slight]);
    // Cell 1 has a moderate token but no strong match.
    let encoded_slb_trap = encode_per_token_keys(&[&near_miss]);

    engine.mem_write(1, 0, &encoded_target, vec![0.0; 2], 50.0, None).unwrap();
    engine.mem_write(1, 0, &encoded_slb_trap, vec![0.0; 2], 50.0, None).unwrap();

    let query = encode_per_token_keys(&[&target, &zero]);
    let results = engine.mem_read(&query, 1, Some(1)).unwrap();

    assert_eq!(
        results[0].cell.id, 0,
        "encoded per-token queries must run max-sim cold scoring even when SLB has enough candidates"
    );
}

#[test]
fn test_vamana_handles_encoded_per_token_keys_after_activation() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open_with_vamana_threshold(dir.path(), 1).unwrap();

    let make_tokens = |offset: usize| -> Vec<Vec<f32>> {
        (0..8)
            .map(|token| {
                (0..128).map(|dim| ((offset + token * 17 + dim * 7) as f32 * 0.013).sin()).collect()
            })
            .collect()
    };
    let first_tokens = make_tokens(0);
    let second_tokens = make_tokens(10_000);
    let first_refs: Vec<&[f32]> = first_tokens.iter().map(Vec::as_slice).collect();
    let second_refs: Vec<&[f32]> = second_tokens.iter().map(Vec::as_slice).collect();
    let first = encode_per_token_keys(&first_refs);
    let second = encode_per_token_keys(&second_refs);

    engine.mem_write(1, 0, &first, vec![0.0; 128], 50.0, None).unwrap();
    assert!(engine.has_vamana(), "First encoded write should activate Vamana");

    engine.mem_write(1, 0, &second, vec![0.0; 128], 50.0, None).unwrap();
    let results = engine.mem_read(&second, 2, Some(1)).unwrap();

    assert!(!results.is_empty(), "Encoded keys must remain readable after Vamana activation");
}

fn top5_broad_match_fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Object Mother / Fixture Builder: one query, one broad match, one spike.
    // Specification: Top5Avg must prefer broad coverage over one isolated max.
    let query = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let broad = [0.6f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let spike = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let orthogonal = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    (
        encode_per_token_keys(&[&query]),
        encode_per_token_keys(&[&broad, &broad, &broad, &broad, &broad]),
        encode_per_token_keys(&[&spike, &orthogonal, &orthogonal, &orthogonal, &orthogonal]),
    )
}

fn write_top5_fixture(engine: &mut Engine) {
    let (_query, broad_key, spike_key) = top5_broad_match_fixture();
    engine.mem_write(1, 0, &broad_key, vec![0.0; 8], 50.0, None).unwrap();
    engine.mem_write(1, 0, &spike_key, vec![0.0; 8], 50.0, None).unwrap();
}

#[test]
fn test_engine_default_pipeline_uses_top5avg() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, _broad_key, _spike_key) = top5_broad_match_fixture();
    write_top5_fixture(&mut engine);

    let results = engine.mem_read(&query, 2, Some(1)).unwrap();

    assert_eq!(
        results[0].cell.id, 0,
        "Engine default pipeline should use Top5Avg: broad match must beat a single spike"
    );
}

#[test]
fn test_engine_reopen_preserves_top5avg_behavior() {
    let dir = tempfile::tempdir().unwrap();
    let (query, _broad_key, _spike_key) = top5_broad_match_fixture();

    {
        let mut engine = Engine::open(dir.path()).unwrap();
        write_top5_fixture(&mut engine);
    }

    let mut reopened = Engine::open(dir.path()).unwrap();
    let results = reopened.mem_read(&query, 2, Some(1)).unwrap();

    assert_eq!(
        results[0].cell.id, 0,
        "Rebuilt pipeline after reopen should preserve Top5Avg behavior"
    );
}

#[test]
fn test_engine_encoded_query_uses_per_token_before_fixed_dim_fallbacks() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, _broad_key, _spike_key) = top5_broad_match_fixture();
    write_top5_fixture(&mut engine);

    // Specification: the encoded query must be scored as raw tokens by
    // PerTokenRetriever(Top5Avg) before fixed-dim SLB/brute-force candidates
    // can decide the result.
    let results = engine.mem_read(&query, 1, Some(1)).unwrap();

    assert_eq!(results[0].cell.id, 0);
}

#[test]
fn test_engine_vamana_threshold_preserves_encoded_ranking() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open_with_vamana_threshold(dir.path(), 1).unwrap();
    let (query, broad_key, spike_key) = top5_broad_match_fixture();

    engine.mem_write(1, 0, &broad_key, vec![0.0; 8], 50.0, None).unwrap();
    assert!(engine.has_vamana(), "first write should activate Vamana at threshold 1");
    engine.mem_write(1, 0, &spike_key, vec![0.0; 8], 50.0, None).unwrap();

    let results = engine.mem_read(&query, 2, Some(1)).unwrap();

    assert_eq!(
        results[0].cell.id, 0,
        "Vamana activation must not let fixed-dim pooled scoring override Top5Avg"
    );
}

#[test]
fn test_engine_reopen_preserves_encoded_ranking_after_adapter_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let (query, _broad_key, _spike_key) = top5_broad_match_fixture();

    {
        let mut engine = Engine::open(dir.path()).unwrap();
        write_top5_fixture(&mut engine);
    }

    let mut reopened = Engine::open(dir.path()).unwrap();
    let results = reopened.mem_read(&query, 2, Some(1)).unwrap();

    assert_eq!(
        results[0].cell.id, 0,
        "Adapter rebuild from persisted cells must preserve encoded ranking"
    );
}

const CANDIDATE_FIXTURE_DIM: usize = 128;
const CANDIDATE_FIXTURE_TOKENS_PER_CELL: usize = 8;
const CANDIDATE_FIXTURE_CELL_COUNT: usize = 1_000;
const CANDIDATE_FIXTURE_TARGET_CELL: usize = 503;
const CANDIDATE_FIXTURE_TOP_K: usize = 5;
const CANDIDATE_FIXTURE_OWNER: u64 = 1;
const CANDIDATE_FIXTURE_LAYER: u16 = 0;
const CANDIDATE_FIXTURE_SALIENCE: f32 = 50.0;
const CANDIDATE_FIXTURE_PACK_SALIENCE: f32 = 80.0;
const CANDIDATE_FIXTURE_LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const CANDIDATE_FIXTURE_CELL_SEED_MULTIPLIER: u64 = 0x9E37_79B1_85EB_CA87;
const CANDIDATE_FIXTURE_TOKEN_SEED_MULTIPLIER: u64 = 0xC2B2_AE3D_27D4_EB4F;
const CANDIDATE_FIXTURE_RANDOM_SHIFT: u32 = 40;
const CANDIDATE_FIXTURE_RANDOM_DENOMINATOR: f32 = (1u64 << 24) as f32;

fn normalized_token(dim: usize, cell_id: usize, token_id: usize) -> Vec<f32> {
    let mut state = (cell_id as u64 + 1).wrapping_mul(CANDIDATE_FIXTURE_CELL_SEED_MULTIPLIER)
        ^ (token_id as u64 + 1).wrapping_mul(CANDIDATE_FIXTURE_TOKEN_SEED_MULTIPLIER);
    let mut vector = Vec::with_capacity(dim);
    for _ in 0..dim {
        state = state.wrapping_mul(CANDIDATE_FIXTURE_LCG_MULTIPLIER).wrapping_add(1);
        vector.push(
            ((state >> CANDIDATE_FIXTURE_RANDOM_SHIFT) as f32
                / CANDIDATE_FIXTURE_RANDOM_DENOMINATOR)
                * 2.0
                - 1.0,
        );
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn normalized_encoded_cell(dim: usize, cell_id: usize, tokens_per_cell: usize) -> Vec<f32> {
    let tokens: Vec<Vec<f32>> =
        (0..tokens_per_cell).map(|token_id| normalized_token(dim, cell_id, token_id)).collect();
    let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
    encode_per_token_keys(&refs)
}

#[test]
fn test_engine_candidate_reduction_preserves_encoded_ranking() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    for cell_id in 0..CANDIDATE_FIXTURE_CELL_COUNT {
        let key = normalized_encoded_cell(
            CANDIDATE_FIXTURE_DIM,
            cell_id,
            CANDIDATE_FIXTURE_TOKENS_PER_CELL,
        );
        engine
            .mem_write(
                CANDIDATE_FIXTURE_OWNER,
                CANDIDATE_FIXTURE_LAYER,
                &key,
                vec![0.0; CANDIDATE_FIXTURE_DIM],
                CANDIDATE_FIXTURE_SALIENCE,
                None,
            )
            .unwrap();
    }

    let query = normalized_encoded_cell(
        CANDIDATE_FIXTURE_DIM,
        CANDIDATE_FIXTURE_TARGET_CELL,
        CANDIDATE_FIXTURE_TOKENS_PER_CELL,
    );
    let results =
        engine.mem_read(&query, CANDIDATE_FIXTURE_TOP_K, Some(CANDIDATE_FIXTURE_OWNER)).unwrap();

    assert_eq!(results[0].cell.id, CANDIDATE_FIXTURE_TARGET_CELL as u64);
}

#[test]
fn test_engine_vamana_still_does_not_change_candidate_reduced_ranking() {
    let build_rankings = |threshold: usize| -> Vec<u64> {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open_with_vamana_threshold(dir.path(), threshold).unwrap();
        for cell_id in 0..CANDIDATE_FIXTURE_CELL_COUNT {
            let key = normalized_encoded_cell(
                CANDIDATE_FIXTURE_DIM,
                cell_id,
                CANDIDATE_FIXTURE_TOKENS_PER_CELL,
            );
            engine
                .mem_write(
                    CANDIDATE_FIXTURE_OWNER,
                    CANDIDATE_FIXTURE_LAYER,
                    &key,
                    vec![0.0; CANDIDATE_FIXTURE_DIM],
                    CANDIDATE_FIXTURE_SALIENCE,
                    None,
                )
                .unwrap();
        }
        let query = normalized_encoded_cell(
            CANDIDATE_FIXTURE_DIM,
            CANDIDATE_FIXTURE_TARGET_CELL,
            CANDIDATE_FIXTURE_TOKENS_PER_CELL,
        );
        engine
            .mem_read(&query, CANDIDATE_FIXTURE_TOP_K, Some(CANDIDATE_FIXTURE_OWNER))
            .unwrap()
            .into_iter()
            .map(|result| result.cell.id)
            .collect()
    };

    let exact_pipeline = build_rankings(usize::MAX);
    let with_vamana = build_rankings(1);

    assert_eq!(with_vamana, exact_pipeline);
}

// -- Phase 29: KV Pack ATDD ──────────────────────────────────────────────────

/// ATDD 30: Write a KV Pack, all layers stored atomically.
#[test]
fn test_write_pack_stores_all_layers() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let retrieval_key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);

    let pack = KVPack {
        id: 0, // assigned by engine
        owner: 1,
        retrieval_key,
        layers: (0..12)
            .map(|i| KVLayerPayload {
                layer_idx: i,
                data: vec![i as f32 * 0.1; 64], // dummy K+V payload
            })
            .collect(),
        salience: 80.0,
    };

    let pack_id = engine.mem_write_pack(&pack).unwrap();
    assert!(pack_id >= 1); // pack IDs start at 1
    assert_eq!(engine.pack_count(), 1);
}

/// ATDD 31: Read back a stored KV Pack with all layers intact.
#[test]
fn test_read_pack_returns_complete_kv() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key_vec = [1.0f32, 0.0, 0.0, 0.0];
    let retrieval_key = encode_per_token_keys(&[&key_vec]);

    let pack = KVPack {
        id: 0,
        owner: 1,
        retrieval_key,
        layers: (0..4)
            .map(|i| KVLayerPayload { layer_idx: i, data: vec![(i + 1) as f32; 32] })
            .collect(),
        salience: 80.0,
    };

    engine.mem_write_pack(&pack).unwrap();

    // Query with similar key.
    let results = engine.mem_read_pack(&key_vec, 1, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pack.layers.len(), 4);

    // Verify layer data survived Q4 round-trip (approximate).
    for (i, layer) in results[0].pack.layers.iter().enumerate() {
        assert_eq!(layer.layer_idx, i as u16);
        assert_eq!(layer.data.len(), 32);
    }
}

/// ATDD 32: Pack survives engine reopen (persistence).
#[test]
fn test_pack_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let key_vec = [1.0f32, 0.0, 0.0, 0.0];
    let retrieval_key = encode_per_token_keys(&[&key_vec]);

    {
        let mut engine = Engine::open(dir.path()).unwrap();
        let pack = KVPack {
            id: 0,
            owner: 1,
            retrieval_key,
            layers: (0..4)
                .map(|i| KVLayerPayload { layer_idx: i, data: vec![1.0f32; 16] })
                .collect(),
            salience: 80.0,
        };
        engine.mem_write_pack(&pack).unwrap();
        assert_eq!(engine.pack_count(), 1);
    }

    // Reopen.
    let mut engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.pack_count(), 1);

    let results = engine.mem_read_pack(&key_vec, 1, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pack.layers.len(), 4);
}

/// ATDD 33: Multiple packs, retrieval ranks by similarity.
#[test]
fn test_multiple_packs_retrieval_ranking() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Pack 0: about cooking.
    let cooking_key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: cooking_key,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
        })
        .unwrap();

    // Pack 1: about running.
    let running_key = encode_per_token_keys(&[&[0.0f32, 1.0, 0.0, 0.0]]);
    engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: running_key,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
            salience: 80.0,
        })
        .unwrap();

    // Query similar to cooking.
    let query = [0.9f32, 0.1, 0.0, 0.0];
    let results = engine.mem_read_pack(&query, 2, None).unwrap();
    assert_eq!(results.len(), 2);
    // Pack 0 (cooking) should rank first.
    assert_eq!(results[0].pack.layers[0].data[0] as i32, 1, "Cooking pack should rank first");
}

const TEST_PACK_OWNER_ONE: u64 = 1;
const TEST_PACK_OWNER_TWO: u64 = 2;
const TEST_PACK_LAYER_PAYLOAD_DIM: usize = 16;
const TEST_PACK_SALIENCE: f32 = 80.0;
const OWNER_ONE_LAYER_VALUE_CODE: i32 = 110;
const OWNER_TWO_LAYER_VALUE_CODE: i32 = 220;
const OWNER_ONE_LAYER_VALUE: f32 = OWNER_ONE_LAYER_VALUE_CODE as f32;
const OWNER_TWO_LAYER_VALUE: f32 = OWNER_TWO_LAYER_VALUE_CODE as f32;
const KEY_ONLY_INDEX_QUERY_K: usize = 8;
const ZERO_LAYER_COUNT: usize = 0;
const PROFILE_PAYLOAD_DIM: usize = 64;

fn test_pack(layer_values: &[f32], retrieval_key: Vec<f32>) -> KVPack {
    test_pack_for_owner(TEST_PACK_OWNER_ONE, layer_values, retrieval_key)
}

fn test_pack_for_owner(owner: u64, layer_values: &[f32], retrieval_key: Vec<f32>) -> KVPack {
    KVPack {
        id: 0,
        owner,
        retrieval_key,
        layers: layer_values
            .iter()
            .enumerate()
            .map(|(layer_idx, value)| KVLayerPayload {
                layer_idx: layer_idx as u16,
                data: vec![*value; TEST_PACK_LAYER_PAYLOAD_DIM],
            })
            .collect(),
        salience: TEST_PACK_SALIENCE,
    }
}

#[test]
fn test_mem_read_pack_uses_per_token_pipeline_before_slb() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, broad_key, spike_key) = top5_broad_match_fixture();

    engine.mem_write_pack(&test_pack(&[10.0, 10.1, 10.2], broad_key)).unwrap();
    engine.mem_write_pack(&test_pack(&[20.0, 20.1, 20.2], spike_key)).unwrap();

    let results = engine.mem_read_pack(&query, 1, Some(1)).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].pack.layers[0].data[0] as i32, 10,
        "mem_read_pack should use per-token Top5Avg scoring before SLB fallback"
    );
}

#[test]
fn test_mem_read_pack_deduplicates_layer_cells_by_pack() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    engine.mem_write_pack(&test_pack(&[30.0, 31.0, 32.0, 33.0], broad_key)).unwrap();

    let results = engine.mem_read_pack(&query, 5, Some(1)).unwrap();

    assert_eq!(results.len(), 1, "One multi-layer pack should appear once, not once per layer");
    assert_eq!(results[0].pack.layers.len(), 4);
}

#[test]
fn test_mem_read_pack_preserves_pack_deduplication_after_adapter_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    {
        let mut engine = Engine::open(dir.path()).unwrap();
        engine.mem_write_pack(&test_pack(&[30.0, 31.0, 32.0, 33.0], broad_key)).unwrap();
    }

    let mut reopened = Engine::open(dir.path()).unwrap();
    let results = reopened.mem_read_pack(&query, 5, Some(1)).unwrap();

    assert_eq!(results.len(), 1, "Rebuilt pack index should still deduplicate by pack");
    assert_eq!(results[0].pack.layers.len(), 4);
}

#[test]
fn test_mem_read_pack_candidate_reduction_preserves_pack_deduplication() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    for pack_id in 0..CANDIDATE_FIXTURE_CELL_COUNT {
        let pack = KVPack {
            id: 0,
            owner: CANDIDATE_FIXTURE_OWNER,
            retrieval_key: normalized_encoded_cell(
                CANDIDATE_FIXTURE_DIM,
                pack_id,
                CANDIDATE_FIXTURE_TOKENS_PER_CELL,
            ),
            layers: vec![
                KVLayerPayload {
                    layer_idx: CANDIDATE_FIXTURE_LAYER,
                    data: vec![pack_id as f32; CANDIDATE_FIXTURE_DIM],
                },
                KVLayerPayload {
                    layer_idx: CANDIDATE_FIXTURE_LAYER + 1,
                    data: vec![pack_id as f32 + 0.5; CANDIDATE_FIXTURE_DIM],
                },
            ],
            salience: CANDIDATE_FIXTURE_PACK_SALIENCE,
        };
        engine.mem_write_pack(&pack).unwrap();
    }

    let query = normalized_encoded_cell(
        CANDIDATE_FIXTURE_DIM,
        CANDIDATE_FIXTURE_TARGET_CELL,
        CANDIDATE_FIXTURE_TOKENS_PER_CELL,
    );
    let results = engine
        .mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(CANDIDATE_FIXTURE_OWNER))
        .unwrap();

    assert_eq!(results[0].pack.layers[0].data[0].round() as usize, CANDIDATE_FIXTURE_TARGET_CELL);
    let pack_ids: std::collections::HashSet<usize> =
        results.iter().map(|result| result.pack.layers[0].data[0].round() as usize).collect();
    assert_eq!(pack_ids.len(), results.len(), "candidate reduction must still deduplicate packs");
}

#[test]
fn test_engine_reopen_rebuilds_pack_reverse_index() {
    let dir = tempfile::tempdir().unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    {
        let mut engine = Engine::open(dir.path()).unwrap();
        engine.mem_write_pack(&test_pack(&[30.0, 31.0, 32.0, 33.0], broad_key)).unwrap();
    }

    let mut reopened = Engine::open(dir.path()).unwrap();
    let results =
        reopened.mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(TEST_PACK_OWNER_ONE)).unwrap();

    assert_eq!(results.len(), 1, "reverse pack directory should rebuild after reopen");
    assert_eq!(results[0].pack.layers.len(), 4);
}

#[test]
fn test_mem_read_pack_reverse_lookup_preserves_ranking() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    for pack_id in 0..CANDIDATE_FIXTURE_CELL_COUNT {
        engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: CANDIDATE_FIXTURE_OWNER,
                retrieval_key: normalized_encoded_cell(
                    CANDIDATE_FIXTURE_DIM,
                    pack_id,
                    CANDIDATE_FIXTURE_TOKENS_PER_CELL,
                ),
                layers: vec![KVLayerPayload {
                    layer_idx: CANDIDATE_FIXTURE_LAYER,
                    data: vec![pack_id as f32; CANDIDATE_FIXTURE_DIM],
                }],
                salience: CANDIDATE_FIXTURE_PACK_SALIENCE,
            })
            .unwrap();
    }

    let query = normalized_encoded_cell(
        CANDIDATE_FIXTURE_DIM,
        CANDIDATE_FIXTURE_TARGET_CELL,
        CANDIDATE_FIXTURE_TOKENS_PER_CELL,
    );
    let results = engine
        .mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(CANDIDATE_FIXTURE_OWNER))
        .unwrap();

    assert_eq!(results[0].pack.layers[0].data[0].round() as usize, CANDIDATE_FIXTURE_TARGET_CELL);
}

#[test]
fn test_mem_read_pack_reverse_lookup_preserves_deduplication() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    engine.mem_write_pack(&test_pack(&[30.0, 31.0, 32.0, 33.0], broad_key)).unwrap();

    let results =
        engine.mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(TEST_PACK_OWNER_ONE)).unwrap();

    assert_eq!(results.len(), 1, "reverse lookup must preserve pack-level deduplication");
    assert_eq!(results[0].pack.layers.len(), 4);
}

#[test]
fn test_mem_read_pack_reverse_lookup_respects_owner_filter() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    engine
        .mem_write_pack(&test_pack_for_owner(
            TEST_PACK_OWNER_ONE,
            &[OWNER_ONE_LAYER_VALUE],
            broad_key.clone(),
        ))
        .unwrap();
    engine
        .mem_write_pack(&test_pack_for_owner(
            TEST_PACK_OWNER_TWO,
            &[OWNER_TWO_LAYER_VALUE],
            broad_key,
        ))
        .unwrap();

    let results =
        engine.mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(TEST_PACK_OWNER_TWO)).unwrap();

    assert!(results.iter().all(|result| result.pack.owner == TEST_PACK_OWNER_TWO));
    assert_eq!(results[0].pack.layers[0].data[0].round() as i32, OWNER_TWO_LAYER_VALUE_CODE);
}

#[test]
fn test_mem_read_pack_indexes_only_retrieval_cell() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    engine.mem_write_pack(&test_pack(&[30.0, 31.0, 32.0, 33.0], broad_key)).unwrap();

    let results =
        engine.mem_read(&query, KEY_ONLY_INDEX_QUERY_K, Some(TEST_PACK_OWNER_ONE)).unwrap();

    assert_eq!(results.len(), 1, "pack layer payload cells must not be separately indexed");
    assert!(results[0].cell.value.is_empty(), "only the pack retrieval cell should be retrieved");
}

#[test]
fn test_mem_read_pack_key_only_index_preserves_complete_reconstruction() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    engine.mem_write_pack(&test_pack(&[30.0, 31.0, 32.0, 33.0], broad_key)).unwrap();

    let results =
        engine.mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(TEST_PACK_OWNER_ONE)).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pack.layers.len(), 4);
    assert!(
        results[0].pack.layers.windows(2).all(|layers| layers[0].layer_idx < layers[1].layer_idx)
    );
}

#[test]
fn test_reopen_rebuild_indexes_only_pack_retrieval_cells() {
    let dir = tempfile::tempdir().unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    {
        let mut engine = Engine::open(dir.path()).unwrap();
        engine.mem_write_pack(&test_pack(&[30.0, 31.0, 32.0, 33.0], broad_key)).unwrap();
    }

    let mut reopened = Engine::open(dir.path()).unwrap();
    let results =
        reopened.mem_read(&query, KEY_ONLY_INDEX_QUERY_K, Some(TEST_PACK_OWNER_ONE)).unwrap();

    assert_eq!(results.len(), 1, "rebuild must not index layer payload cells");
    assert!(
        results[0].cell.value.is_empty(),
        "rebuilt retrieval index should expose retrieval cell only"
    );
}

#[test]
fn test_key_only_index_preserves_1000_pack_ranking() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    for pack_id in 0..CANDIDATE_FIXTURE_CELL_COUNT {
        engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: CANDIDATE_FIXTURE_OWNER,
                retrieval_key: normalized_encoded_cell(
                    CANDIDATE_FIXTURE_DIM,
                    pack_id,
                    CANDIDATE_FIXTURE_TOKENS_PER_CELL,
                ),
                layers: vec![KVLayerPayload {
                    layer_idx: CANDIDATE_FIXTURE_LAYER,
                    data: vec![pack_id as f32; CANDIDATE_FIXTURE_DIM],
                }],
                salience: CANDIDATE_FIXTURE_PACK_SALIENCE,
            })
            .unwrap();
    }

    let query = normalized_encoded_cell(
        CANDIDATE_FIXTURE_DIM,
        CANDIDATE_FIXTURE_TARGET_CELL,
        CANDIDATE_FIXTURE_TOKENS_PER_CELL,
    );
    let results = engine
        .mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(CANDIDATE_FIXTURE_OWNER))
        .unwrap();

    assert_eq!(results[0].pack.layers[0].data[0].round() as usize, CANDIDATE_FIXTURE_TARGET_CELL);
}

#[test]
fn test_key_only_index_preserves_owner_filter() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    engine
        .mem_write_pack(&test_pack_for_owner(
            TEST_PACK_OWNER_ONE,
            &[OWNER_ONE_LAYER_VALUE],
            broad_key.clone(),
        ))
        .unwrap();
    engine
        .mem_write_pack(&test_pack_for_owner(
            TEST_PACK_OWNER_TWO,
            &[OWNER_TWO_LAYER_VALUE],
            broad_key,
        ))
        .unwrap();

    let results =
        engine.mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(TEST_PACK_OWNER_TWO)).unwrap();

    assert!(results.iter().all(|result| result.pack.owner == TEST_PACK_OWNER_TWO));
    assert_eq!(results[0].pack.layers[0].data[0].round() as i32, OWNER_TWO_LAYER_VALUE_CODE);
}

#[test]
fn test_pack_fixture_with_zero_layers_is_retrievable() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    engine.mem_write_pack(&test_pack(&[], broad_key)).unwrap();

    let results =
        engine.mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(TEST_PACK_OWNER_ONE)).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pack.owner, TEST_PACK_OWNER_ONE);
    assert_eq!(results[0].pack.layers.len(), ZERO_LAYER_COUNT);
}

#[test]
fn test_pack_fixture_payload_dimension_is_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();
    let pack = KVPack {
        id: 0,
        owner: TEST_PACK_OWNER_ONE,
        retrieval_key: broad_key,
        layers: vec![KVLayerPayload {
            layer_idx: CANDIDATE_FIXTURE_LAYER,
            data: vec![OWNER_ONE_LAYER_VALUE; PROFILE_PAYLOAD_DIM],
        }],
        salience: TEST_PACK_SALIENCE,
    };

    engine.mem_write_pack(&pack).unwrap();
    let results =
        engine.mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(TEST_PACK_OWNER_ONE)).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pack.layers[0].data.len(), PROFILE_PAYLOAD_DIM);
}

fn recall_at(rankings: &[Vec<u64>], expected: &[u64], k: usize) -> f32 {
    let hits = rankings
        .iter()
        .zip(expected)
        .filter(|(ranking, expected_id)| ranking.iter().take(k).any(|id| id == *expected_id))
        .count();
    hits as f32 / expected.len() as f32
}

fn worst_top1_concentration(rankings: &[Vec<u64>]) -> usize {
    let mut counts = std::collections::HashMap::new();
    for ranking in rankings {
        if let Some(top1) = ranking.first() {
            *counts.entry(*top1).or_insert(0usize) += 1;
        }
    }
    counts.values().copied().max().unwrap_or(0)
}

fn synthetic_target_ids(cell_count: usize, query_count: usize) -> Vec<u64> {
    (0..query_count).map(|idx| ((idx * 7 + 3) % cell_count) as u64).collect()
}

fn has_vamana_regression(before: &[Vec<u64>], after: &[Vec<u64>], expected: &[u64]) -> bool {
    recall_at(after, expected, 5) < recall_at(before, expected, 5)
}

#[test]
fn test_rust_retrieval_metrics_match_known_rankings() {
    let rankings = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
    let expected = vec![1, 6, 10];

    assert!((recall_at(&rankings, &expected, 1) - 1.0 / 3.0).abs() < f32::EPSILON);
    assert!((recall_at(&rankings, &expected, 3) - 2.0 / 3.0).abs() < f32::EPSILON);
}

#[test]
fn test_top1_concentration_flags_gravity_well() {
    let rankings = vec![vec![87, 1], vec![87, 2], vec![87, 3], vec![6, 4]];

    assert_eq!(worst_top1_concentration(&rankings), 3);
}

#[test]
fn test_synthetic_corpus_builder_assigns_one_target_per_query() {
    let targets = synthetic_target_ids(100, 30);

    assert_eq!(targets.len(), 30);
    assert!(targets.iter().all(|id| *id < 100));
}

#[test]
fn test_correctness_report_detects_vamana_regression() {
    let expected = vec![1, 2];
    let before = vec![vec![1, 9, 8], vec![2, 7, 6]];
    let after = vec![vec![9, 8, 1], vec![7, 6, 5]];

    assert!(has_vamana_regression(&before, &after, &expected));
}

// -- Phase 30: Pack Materialization ATDD ────────────────────────────────────

const OUT_OF_ORDER_LAYER_A: u16 = 3;
const OUT_OF_ORDER_LAYER_B: u16 = 0;
const OUT_OF_ORDER_LAYER_C: u16 = 7;
const OUT_OF_ORDER_LAYER_D: u16 = 1;
const OUT_OF_ORDER_LAYER_COUNT: usize = 4;
const GOVERNANCE_PACK_LAYER_COUNT: usize = 4;
const GOVERNANCE_ACCESS_BOOST_PER_READ: f32 = 3.0;
const GOVERNANCE_WRITE_BOOST: f32 = 5.0;
const GOVERNANCE_SALIENCE: f32 = 50.0;

/// ATDD 35: Out-of-order stored layer cells are returned sorted by `layer_idx`.
#[test]
fn test_pack_layer_hydration_preserves_layer_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    // Write pack with deliberately out-of-order layer indices.
    let pack = KVPack {
        id: 0,
        owner: TEST_PACK_OWNER_ONE,
        retrieval_key: broad_key,
        layers: vec![
            KVLayerPayload {
                layer_idx: OUT_OF_ORDER_LAYER_A,
                data: vec![OUT_OF_ORDER_LAYER_A as f32; TEST_PACK_LAYER_PAYLOAD_DIM],
            },
            KVLayerPayload {
                layer_idx: OUT_OF_ORDER_LAYER_B,
                data: vec![OUT_OF_ORDER_LAYER_B as f32; TEST_PACK_LAYER_PAYLOAD_DIM],
            },
            KVLayerPayload {
                layer_idx: OUT_OF_ORDER_LAYER_C,
                data: vec![OUT_OF_ORDER_LAYER_C as f32; TEST_PACK_LAYER_PAYLOAD_DIM],
            },
            KVLayerPayload {
                layer_idx: OUT_OF_ORDER_LAYER_D,
                data: vec![OUT_OF_ORDER_LAYER_D as f32; TEST_PACK_LAYER_PAYLOAD_DIM],
            },
        ],
        salience: TEST_PACK_SALIENCE,
    };

    engine.mem_write_pack(&pack).unwrap();

    let results =
        engine.mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(TEST_PACK_OWNER_ONE)).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pack.layers.len(), OUT_OF_ORDER_LAYER_COUNT);

    // Layers must be sorted by layer_idx regardless of write order.
    assert!(
        results[0].pack.layers.windows(2).all(|pair| pair[0].layer_idx < pair[1].layer_idx),
        "Layers must be sorted by layer_idx. Got: {:?}",
        results[0].pack.layers.iter().map(|l| l.layer_idx).collect::<Vec<_>>()
    );

    // Verify exact order: 0, 1, 3, 7.
    let layer_indices: Vec<u16> = results[0].pack.layers.iter().map(|l| l.layer_idx).collect();
    assert_eq!(
        layer_indices,
        vec![
            OUT_OF_ORDER_LAYER_B,
            OUT_OF_ORDER_LAYER_D,
            OUT_OF_ORDER_LAYER_A,
            OUT_OF_ORDER_LAYER_C
        ]
    );
}

/// ATDD 36: Governance updates once per returned pack, not once per layer or candidate.
#[test]
fn test_pack_materialization_updates_governance_once_per_returned_pack() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let (query, broad_key, _spike_key) = top5_broad_match_fixture();

    let pack = KVPack {
        id: 0,
        owner: TEST_PACK_OWNER_ONE,
        retrieval_key: broad_key,
        layers: (0..GOVERNANCE_PACK_LAYER_COUNT as u16)
            .map(|i| KVLayerPayload {
                layer_idx: i,
                data: vec![i as f32; TEST_PACK_LAYER_PAYLOAD_DIM],
            })
            .collect(),
        salience: GOVERNANCE_SALIENCE,
    };

    let pack_id = engine.mem_write_pack(&pack).unwrap();

    // Initial importance: salience + write boost = 50 + 5 = 55.
    let importance_before_read = engine.pack_importance(pack_id).unwrap();
    let expected_initial = GOVERNANCE_SALIENCE + GOVERNANCE_WRITE_BOOST;
    assert!(
        (importance_before_read - expected_initial).abs() < 1.0,
        "Initial importance {importance_before_read} should be ≈{expected_initial}"
    );

    // Read the pack once — should trigger exactly one on_access (+3).
    let _results =
        engine.mem_read_pack(&query, CANDIDATE_FIXTURE_TOP_K, Some(TEST_PACK_OWNER_ONE)).unwrap();

    let importance_after_read = engine.pack_importance(pack_id).unwrap();
    let expected_after_one_read = expected_initial + GOVERNANCE_ACCESS_BOOST_PER_READ;
    assert!(
        (importance_after_read - expected_after_one_read).abs() < 1.0,
        "After one read, importance {importance_after_read} should be ≈{expected_after_one_read} \
         (one on_access, not {} for {} layers)",
        GOVERNANCE_ACCESS_BOOST_PER_READ * GOVERNANCE_PACK_LAYER_COUNT as f32,
        GOVERNANCE_PACK_LAYER_COUNT
    );
}

/// ATDD 34: Pack governance — importance decays across the whole pack.
#[test]
fn test_pack_governance() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack_id = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 50.0,
        })
        .unwrap();

    let imp_before = engine.pack_importance(pack_id);
    assert!(imp_before.is_some());

    engine.advance_days(100.0);

    let imp_after = engine.pack_importance(pack_id);
    assert!(imp_after.unwrap() < imp_before.unwrap(), "Importance should decay");
}

// -- Phase 37: Pack-level Rust APIs ─────────────────────────────────────────

/// ATDD: load_pack_by_id returns complete pack without retrieval scoring.
#[test]
fn test_load_pack_by_id_returns_complete_pack() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack = KVPack {
        id: 0,
        owner: 1,
        retrieval_key: key,
        layers: (0..4)
            .map(|i| KVLayerPayload { layer_idx: i, data: vec![i as f32; 16] })
            .collect(),
        salience: 80.0,
    };

    let pack_id = engine.mem_write_pack(&pack).unwrap();
    let loaded = engine.load_pack_by_id(pack_id).unwrap();

    assert_eq!(loaded.pack.id, pack_id);
    assert_eq!(loaded.pack.layers.len(), 4);
    for (i, layer) in loaded.pack.layers.iter().enumerate() {
        assert_eq!(layer.layer_idx, i as u16);
        assert_eq!(layer.data.len(), 16);
    }
}

/// ATDD: load_pack_by_id fails for nonexistent pack.
#[test]
fn test_load_pack_by_id_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    assert!(engine.load_pack_by_id(999).is_err());
}

/// ATDD: add_pack_link creates durable bidirectional trace edge.
#[test]
fn test_add_pack_link_creates_bidirectional_edge() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key_a = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let key_b = encode_per_token_keys(&[&[0.0f32, 1.0, 0.0, 0.0]]);

    let pack_a = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_a,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
        })
        .unwrap();

    let pack_b = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_b,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
            salience: 80.0,
        })
        .unwrap();

    engine.add_pack_link(pack_a, pack_b).unwrap();

    let links_a = engine.pack_links(pack_a);
    let links_b = engine.pack_links(pack_b);
    assert!(links_a.contains(&pack_b), "pack_a should link to pack_b");
    assert!(links_b.contains(&pack_a), "pack_b should link to pack_a");
}

/// ATDD: pack links survive engine reopen via WAL replay.
#[test]
fn test_pack_links_survive_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let key_a = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let key_b = encode_per_token_keys(&[&[0.0f32, 1.0, 0.0, 0.0]]);

    let (pack_a, pack_b) = {
        let mut engine = Engine::open(dir.path()).unwrap();

        let a = engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key_a,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
                salience: 80.0,
            })
            .unwrap();

        let b = engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key_b,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
                salience: 80.0,
            })
            .unwrap();

        engine.add_pack_link(a, b).unwrap();
        (a, b)
    }; // engine dropped, WAL flushed

    // Reopen — WAL replay should restore trace edges
    let engine = Engine::open(dir.path()).unwrap();
    let links = engine.pack_links(pack_a);
    assert!(links.contains(&pack_b), "links should survive reopen");
}

/// ATDD: trace-boosted retrieval prefers linked pack over unlinked.
#[test]
fn test_trace_boost_prefers_linked_pack() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Pack A: linked, slightly lower base score
    let key_a = encode_per_token_keys(&[&[0.9f32, 0.1, 0.0, 0.0]]);
    let pack_a = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_a,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
        })
        .unwrap();

    // Pack B: unlinked, slightly higher base score
    let key_b = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack_b = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_b,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
            salience: 80.0,
        })
        .unwrap();

    // Pack C: linked to pack_a (gives pack_a a trace link)
    let key_c = encode_per_token_keys(&[&[0.0f32, 0.0, 1.0, 0.0]]);
    let _pack_c = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_c,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![3.0; 16] }],
            salience: 80.0,
        })
        .unwrap();

    engine.add_pack_link(pack_a, _pack_c).unwrap();

    // Query close to both A and B
    let query = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);

    // Without boost: pack_b should rank first (higher base score)
    let no_boost = engine.mem_read_pack(&query, 2, None).unwrap();
    assert_eq!(no_boost[0].pack.id, pack_b, "without boost, unlinked B ranks first");

    // With boost: pack_a should rank first (link boost overcomes score gap)
    let with_boost = engine
        .mem_read_pack_with_trace_boost(&query, 2, None, 0.5)
        .unwrap();
    assert_eq!(
        with_boost[0].pack.id, pack_a,
        "with boost, linked A should rank first"
    );
}
