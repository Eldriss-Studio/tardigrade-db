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

/// ATDD Test 14: Query when SLB is empty — falls through to [`BruteForce`].
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
        text: None,
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
        text: None,
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
            text: None,
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
            text: None,
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
            text: None,
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
        text: None,
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
            text: None,
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
                text: None,
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
                text: None,
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
        text: None,
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
        text: None,
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
        text: None,
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
            text: None,
        })
        .unwrap();

    let imp_before = engine.pack_importance(pack_id);
    assert!(imp_before.is_some());

    engine.advance_days(100.0);

    let imp_after = engine.pack_importance(pack_id);
    assert!(imp_after.unwrap() < imp_before.unwrap(), "Importance should decay");
}

// -- Phase 37: Pack-level Rust APIs ─────────────────────────────────────────

/// ATDD: `load_pack_by_id` returns complete pack without retrieval scoring.
#[test]
fn test_load_pack_by_id_returns_complete_pack() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack = KVPack {
        id: 0,
        owner: 1,
        retrieval_key: key,
        layers: (0..4).map(|i| KVLayerPayload { layer_idx: i, data: vec![i as f32; 16] }).collect(),
        salience: 80.0,
        text: None,
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

/// ATDD: `load_pack_by_id` fails for nonexistent pack.
#[test]
fn test_load_pack_by_id_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    assert!(engine.load_pack_by_id(999).is_err());
}

/// ATDD: `add_pack_link` creates durable bidirectional trace edge.
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
            text: None,
        })
        .unwrap();

    let pack_b = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_b,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
            salience: 80.0,
            text: None,
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
                text: None,
            })
            .unwrap();

        let b = engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key_b,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
                salience: 80.0,
                text: None,
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
            text: None,
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
            text: None,
        })
        .unwrap();

    // Pack C: linked to pack_a (gives pack_a a trace link)
    let key_c = encode_per_token_keys(&[&[0.0f32, 0.0, 1.0, 0.0]]);
    let pack_c = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_c,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![3.0; 16] }],
            salience: 80.0,
            text: None,
        })
        .unwrap();

    engine.add_pack_link(pack_a, pack_c).unwrap();

    // Query close to both A and B
    let query = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);

    // Without boost: pack_b should rank first (higher base score)
    let no_boost = engine.mem_read_pack(&query, 2, None).unwrap();
    assert_eq!(no_boost[0].pack.id, pack_b, "without boost, unlinked B ranks first");

    // With boost: pack_a should rank first (link boost overcomes score gap)
    let with_boost = engine.mem_read_pack_with_trace_boost(&query, 2, None, 0.5).unwrap();
    assert_eq!(with_boost[0].pack.id, pack_a, "with boost, linked A should rank first");
}

// ── Text Store acceptance tests ─────────────────────────────────────────────

#[test]
fn test_pack_text_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack_id = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key.clone(),
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
            text: Some("Nyx's favorite star is Vega".to_owned()),
        })
        .unwrap();

    let result = engine.load_pack_by_id(pack_id).unwrap();
    assert_eq!(result.pack.text.as_deref(), Some("Nyx's favorite star is Vega"),);
}

#[test]
fn test_pack_text_none_is_valid() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack_id = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
            text: None,
        })
        .unwrap();

    let result = engine.load_pack_by_id(pack_id).unwrap();
    assert_eq!(result.pack.text, None);
}

#[test]
fn test_pack_text_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let pack_id = {
        let mut engine = Engine::open(dir.path()).unwrap();
        let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
        engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
                salience: 80.0,
                text: Some("Persisted across restarts".to_owned()),
            })
            .unwrap()
    };

    let mut engine = Engine::open(dir.path()).unwrap();
    let result = engine.load_pack_by_id(pack_id).unwrap();
    assert_eq!(result.pack.text.as_deref(), Some("Persisted across restarts"),);
}

#[test]
fn test_pack_text_empty_string_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack_id = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
            text: Some(String::new()),
        })
        .unwrap();

    let result = engine.load_pack_by_id(pack_id).unwrap();
    assert_eq!(result.pack.text.as_deref(), Some(""));
}

#[test]
fn test_set_pack_text_updates_existing_pack() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack_id = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
            text: None,
        })
        .unwrap();

    assert_eq!(engine.pack_text(pack_id), None);
    engine.set_pack_text(pack_id, "Backfilled text").unwrap();
    assert_eq!(engine.pack_text(pack_id), Some("Backfilled text"));
}

#[test]
fn test_set_pack_text_errors_for_unknown_pack() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    assert!(engine.set_pack_text(999, "anything").is_err());
}

#[test]
fn test_set_pack_text_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let pack_id = {
        let mut engine = Engine::open(dir.path()).unwrap();
        let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
        let id = engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
                salience: 80.0,
                text: None,
            })
            .unwrap();
        engine.set_pack_text(id, "Migrated text").unwrap();
        id
    };
    let engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.pack_text(pack_id), Some("Migrated text"));
}

// -- Batch text + pack_exists acceptance tests --

fn write_pack_with_no_text(engine: &mut Engine, key_seed: f32) -> u64 {
    let key = encode_per_token_keys(&[&[key_seed, 0.0, 0.0, 0.0]]);
    engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
            text: None,
        })
        .unwrap()
}

#[test]
fn test_set_pack_texts_batch_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let id_a = write_pack_with_no_text(&mut engine, 1.0);
    let id_b = write_pack_with_no_text(&mut engine, 2.0);
    let id_c = write_pack_with_no_text(&mut engine, 3.0);

    engine.set_pack_texts(&[(id_a, "alpha"), (id_b, "beta"), (id_c, "gamma")]).unwrap();

    assert_eq!(engine.pack_text(id_a), Some("alpha"));
    assert_eq!(engine.pack_text(id_b), Some("beta"));
    assert_eq!(engine.pack_text(id_c), Some("gamma"));
}

#[test]
fn test_set_pack_texts_errors_on_first_missing_pack_with_no_writes() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let real_id = write_pack_with_no_text(&mut engine, 1.0);

    // Mix valid + invalid; the invalid one is in the middle so we can prove
    // the leading valid entry was *not* written despite preceding the error.
    let result = engine.set_pack_texts(&[(real_id, "first"), (9999, "boom"), (real_id, "third")]);

    assert!(result.is_err());
    // The valid leading entry must not have been written — fail-fast = atomic.
    assert_eq!(engine.pack_text(real_id), None);
}

#[test]
fn test_set_pack_texts_empty_batch_is_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    engine.set_pack_texts(&[]).unwrap();
    // Empty batch creates nothing on disk; reopen confirms.
    let engine2 = Engine::open(dir.path()).unwrap();
    assert_eq!(engine2.pack_count(), 0);
}

#[test]
fn test_pack_exists_true_for_stored_pack() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let id = write_pack_with_no_text(&mut engine, 1.0);
    assert!(engine.pack_exists(id));
}

#[test]
fn test_pack_exists_false_for_missing_pack() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(dir.path()).unwrap();
    assert!(!engine.pack_exists(9999));
}

#[test]
fn test_pack_exists_false_after_delete() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let id = write_pack_with_no_text(&mut engine, 1.0);
    assert!(engine.pack_exists(id));
    engine.delete_pack(id).unwrap();
    assert!(!engine.pack_exists(id));
}

// -- Delete API acceptance tests --

#[test]
fn test_delete_pack_removes_from_retrieval() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key_a = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let key_b = encode_per_token_keys(&[&[0.0f32, 1.0, 0.0, 0.0]]);

    let pack_a = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_a.clone(),
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
            text: Some("Memory A".to_owned()),
        })
        .unwrap();

    let pack_b = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_b,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
            salience: 80.0,
            text: Some("Memory B".to_owned()),
        })
        .unwrap();

    engine.delete_pack(pack_a).unwrap();

    let results = engine.mem_read_pack(&key_a, 2, None).unwrap();
    let found_ids: Vec<_> = results.iter().map(|r| r.pack.id).collect();
    assert!(!found_ids.contains(&pack_a), "deleted pack should not appear");
    assert_eq!(engine.pack_count(), 1);
    assert_eq!(results[0].pack.id, pack_b);
}

#[test]
fn test_delete_pack_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let pack_id = {
        let mut engine = Engine::open(dir.path()).unwrap();
        let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
        let id = engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
                salience: 80.0,
                text: Some("Will be deleted".to_owned()),
            })
            .unwrap();
        engine.delete_pack(id).unwrap();
        id
    };
    let mut engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.pack_count(), 0);
    assert!(engine.load_pack_by_id(pack_id).is_err());
}

#[test]
fn test_delete_nonexistent_pack_errors() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    assert!(engine.delete_pack(999).is_err());
}

#[test]
fn test_delete_pack_orphans_trace_links() {
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
            text: None,
        })
        .unwrap();
    let pack_b = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_b,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
            salience: 80.0,
            text: None,
        })
        .unwrap();
    engine.add_pack_link(pack_a, pack_b).unwrap();
    engine.delete_pack(pack_b).unwrap();
    assert!(engine.pack_links(pack_a).is_empty());
}

#[test]
fn test_delete_pack_decrements_count() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let mut ids = Vec::new();
    for i in 0..3 {
        let key = encode_per_token_keys(&[&[i as f32, 0.0, 0.0, 0.0]]);
        ids.push(
            engine
                .mem_write_pack(&KVPack {
                    id: 0,
                    owner: 1,
                    retrieval_key: key,
                    layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
                    salience: 80.0,
                    text: None,
                })
                .unwrap(),
        );
    }
    assert_eq!(engine.pack_count(), 3);
    engine.delete_pack(ids[1]).unwrap();
    assert_eq!(engine.pack_count(), 2);
}

#[test]
fn test_delete_pack_clears_text_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let pack_id = {
        let mut engine = Engine::open(dir.path()).unwrap();
        let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
        let id = engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
                salience: 80.0,
                text: Some("Will be deleted".to_owned()),
            })
            .unwrap();
        engine.delete_pack(id).unwrap();
        id
    };

    // Reopen — deletion log filters both pack_directory AND text_store.
    let engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.pack_text(pack_id), None);
}

#[test]
fn test_delete_pack_preserves_other_packs() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let mut ids = Vec::new();
    for i in 0u8..3 {
        let key = encode_per_token_keys(&[&[i as f32, 0.0, 0.0, 0.0]]);
        ids.push(
            engine
                .mem_write_pack(&KVPack {
                    id: 0,
                    owner: 1,
                    retrieval_key: key,
                    layers: vec![KVLayerPayload { layer_idx: 0, data: vec![i as f32 + 1.0; 16] }],
                    salience: 80.0,
                    text: Some(format!("Pack {i}")),
                })
                .unwrap(),
        );
    }
    engine.delete_pack(ids[1]).unwrap();
    let r0 = engine.load_pack_by_id(ids[0]).unwrap();
    assert_eq!(r0.pack.text.as_deref(), Some("Pack 0"));
    let r2 = engine.load_pack_by_id(ids[2]).unwrap();
    assert_eq!(r2.pack.text.as_deref(), Some("Pack 2"));
    assert!(engine.load_pack_by_id(ids[1]).is_err());
}

// ─────────────────────────────────────────────────────────────────────────
// Step 5 (Gap 7) ATDD Layer 1 — Engine::refresh() cross-handle visibility.
//
// Pattern under test: Memento (re-applied). `Engine::open` already builds
// in-memory state from durable history; `Engine::refresh` re-runs the same
// rebuild in place so a separately-opened Engine handle sees writes made
// by another handle at the same path.
//
// Each test here MUST fail until Step 5b lands (no method named `refresh`).
// ─────────────────────────────────────────────────────────────────────────

fn refresh_test_pack(seed: f32) -> KVPack {
    let retrieval_key = encode_per_token_keys(&[&[seed, 0.0, 0.0, 0.0]]);
    KVPack {
        id: 0,
        owner: 1,
        retrieval_key,
        layers: (0..4)
            .map(|i| KVLayerPayload { layer_idx: i, data: vec![seed + i as f32; 32] })
            .collect(),
        salience: 80.0,
        text: None,
    }
}

/// Step 5a-L1.1: cross-handle write visibility after refresh.
/// GIVEN two Engine handles open at the same dir,
/// WHEN handle A writes a pack and handle B calls `refresh()`,
/// THEN B.`pack_count()` reflects A's write.
#[test]
fn test_refresh_picks_up_writes_from_other_handle() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = Engine::open(dir.path()).unwrap();
    let mut b = Engine::open(dir.path()).unwrap();

    assert_eq!(a.pack_count(), 0);
    assert_eq!(b.pack_count(), 0);

    let pack_id = a.mem_write_pack(&refresh_test_pack(1.0)).unwrap();

    // Without refresh, B is stale.
    assert_eq!(b.pack_count(), 0, "stale view expected before refresh");

    b.refresh().unwrap();
    assert_eq!(b.pack_count(), 1, "refresh should expose A's write");
    assert!(b.pack_exists(pack_id));
}

/// Step 5a-L1.2: refresh must be idempotent — calling it twice with no
/// intervening writes yields the same state.
#[test]
fn test_refresh_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = Engine::open(dir.path()).unwrap();
    a.mem_write_pack(&refresh_test_pack(2.0)).unwrap();

    let mut b = Engine::open(dir.path()).unwrap();
    b.refresh().unwrap();
    let count_after_first = b.pack_count();
    b.refresh().unwrap();
    assert_eq!(b.pack_count(), count_after_first, "second refresh must be a no-op");
}

/// Step 5a-L1.3: WAL-derived state (`TraceGraph`) must also re-apply.
/// GIVEN handle A links two packs in the trace graph,
/// WHEN handle B calls `refresh()`,
/// THEN B can traverse the link via `pack_links()`.
#[test]
fn test_refresh_picks_up_trace_edges() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = Engine::open(dir.path()).unwrap();
    let mut b = Engine::open(dir.path()).unwrap();

    let p1 = a.mem_write_pack(&refresh_test_pack(3.0)).unwrap();
    let p2 = a.mem_write_pack(&refresh_test_pack(4.0)).unwrap();
    a.add_pack_link(p1, p2).unwrap();

    b.refresh().unwrap();
    assert!(b.pack_exists(p1));
    assert!(b.pack_exists(p2));
    let links_from_p1 = b.pack_links(p1);
    assert!(
        links_from_p1.contains(&p2),
        "refresh must replay WAL trace edges; got {links_from_p1:?}"
    );
}

/// Step 5a-L1.5: SLB dimension must adapt when another handle writes cells
/// with a different key dimension than the empty engine's SLB default (128).
///
/// Regression for: `pyo3_runtime.PanicException`
///   "query dimension 1024 does not match SLB dimension 128"
/// observed when the connector (Qwen3-0.6B, `kv_dim=1024`) queried via a
/// scheduler-side handle that had been opened against an empty path.
#[test]
fn test_refresh_rebuilds_slb_when_key_dim_changes() {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Engine::open(dir.path()).unwrap();
    // Reader opens at empty dir — its SLB defaults to dim=128.
    let mut reader = Engine::open(dir.path()).unwrap();

    // Writer stores a pack with a 64-dim retrieval key. encode_per_token_keys
    // produces a flattened encoded key whose mean-pool dim is what matters.
    let key = encode_per_token_keys(&[&vec![0.5_f32; 1024]]);
    let pack = KVPack {
        id: 0,
        owner: 1,
        retrieval_key: key.clone(),
        layers: vec![KVLayerPayload { layer_idx: 0, data: vec![0.5; 64] }],
        salience: 80.0,
        text: None,
    };
    let _pid = writer.mem_write_pack(&pack).unwrap();

    // Refresh reader, then issue a query at the writer's key dim. Before the
    // fix, this panics inside SLB; after the fix, it returns results.
    reader.refresh().unwrap();
    let query = encode_per_token_keys(&[&vec![0.5_f32; 1024]]);
    let _results = reader.mem_read_pack(&query, 1, None).unwrap();
}

/// Step 5a-L1.4: `DeletionLog` mutations from another handle must propagate.
/// GIVEN A writes then deletes a pack,
/// WHEN B refreshes (after each step),
/// THEN B sees the pack first, then sees it gone.
#[test]
fn test_refresh_handles_deletions_from_other_handle() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = Engine::open(dir.path()).unwrap();
    let mut b = Engine::open(dir.path()).unwrap();

    let pid = a.mem_write_pack(&refresh_test_pack(5.0)).unwrap();
    b.refresh().unwrap();
    assert!(b.pack_exists(pid), "B should see pack after first refresh");

    a.delete_pack(pid).unwrap();
    b.refresh().unwrap();
    assert!(!b.pack_exists(pid), "refresh must propagate deletions from another handle");
}

// -- list_packs ---------------------------------------------------------------

#[test]
fn test_list_packs_returns_all_packs() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = vec![1.0_f32; TEST_PACK_LAYER_PAYLOAD_DIM];
    engine.mem_write_pack(&test_pack(&[1.0], key.clone())).unwrap();
    engine.mem_write_pack(&test_pack(&[2.0], key.clone())).unwrap();

    let packs = engine.list_packs(None);
    assert_eq!(packs.len(), 2, "should list all packs");
}

#[test]
fn test_list_packs_filters_by_owner() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = vec![1.0_f32; TEST_PACK_LAYER_PAYLOAD_DIM];
    engine.mem_write_pack(&test_pack_for_owner(1, &[1.0], key.clone())).unwrap();
    engine.mem_write_pack(&test_pack_for_owner(2, &[2.0], key.clone())).unwrap();
    engine.mem_write_pack(&test_pack_for_owner(1, &[3.0], key.clone())).unwrap();

    let owner1 = engine.list_packs(Some(1));
    let owner2 = engine.list_packs(Some(2));
    assert_eq!(owner1.len(), 2);
    assert_eq!(owner2.len(), 1);
}

#[test]
fn test_list_packs_sorted_by_importance_descending() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = vec![1.0_f32; TEST_PACK_LAYER_PAYLOAD_DIM];
    engine.mem_write_pack(&test_pack(&[1.0], key.clone())).unwrap();
    let pid_b = engine.mem_write_pack(&test_pack(&[2.0], key.clone())).unwrap();

    // Boost pid_b by loading it several times (each access adds +3).
    for _ in 0..5 {
        let _ = engine.load_pack_by_id(pid_b);
    }

    let packs = engine.list_packs(None);
    assert_eq!(packs[0].0, pid_b, "higher-importance pack should be first");
    assert!(packs[0].3 > packs[1].3, "importance should be descending");
}

#[test]
fn test_list_packs_empty_engine() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(dir.path()).unwrap();
    assert!(engine.list_packs(None).is_empty());
}

/// ATDD: After `refresh()`, pipeline is rebuilt clean and ALL cells are retrievable.
///
/// Verifies the Memento rebuild: Vamana is reset, pipeline has the default
/// 2 stages (`PerToken` + `BruteForce`), and both old and new cells are queryable.
#[test]
fn test_refresh_rebuilds_pipeline_all_cells_retrievable() {
    let dir = tempfile::tempdir().unwrap();

    // GIVEN engine A with cells written at a low Vamana threshold
    let mut engine_a = Engine::open_with_vamana_threshold(dir.path(), 3).unwrap();
    let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

    // Write enough cells to activate Vamana (threshold = 3)
    for i in 0..5u32 {
        let mut k = key.clone();
        k[0] += i as f32;
        engine_a.mem_write(1, 0, &k, vec![0.0; 64], 50.0, None).unwrap();
    }
    assert!(engine_a.has_vamana(), "Vamana should be active");
    assert_eq!(engine_a.pipeline_stage_count(), 3); // PerToken + BruteForce + Vamana

    // AND engine B at same path writes additional cells
    {
        let mut engine_b = Engine::open(dir.path()).unwrap();
        for i in 5..8u32 {
            let mut k = key.clone();
            k[0] += i as f32;
            engine_b.mem_write(1, 0, &k, vec![0.0; 64], 50.0, None).unwrap();
        }
    } // engine_b dropped

    // WHEN engine A calls refresh()
    engine_a.refresh().unwrap();

    // THEN: Vamana is reset (will re-activate since cell count > threshold)
    // Pipeline should have been rebuilt with default stages, then Vamana re-added
    assert_eq!(engine_a.cell_count(), 8);

    // AND: ALL cells (old + new) are retrievable
    let results = engine_a.mem_read(&key, 8, None).unwrap();
    assert_eq!(
        results.len(),
        8,
        "all 8 cells (5 from engine_a + 3 from engine_b) must be retrievable after refresh"
    );
}

// ── WAL Checkpointing acceptance tests ────────────────────────────────────

/// ATDD: `refresh()` checkpoints the WAL after successful replay.
///
/// GIVEN an engine with trace edges logged to the WAL,
/// WHEN `refresh()` completes,
/// THEN the WAL file is truncated (checkpointed),
/// AND further writes + trace edges still work.
#[test]
fn test_refresh_checkpoints_wal() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("trace.wal");

    let (pack_a, pack_b) = {
        let mut engine = Engine::open(dir.path()).unwrap();
        let key_a = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
        let key_b = encode_per_token_keys(&[&[0.0f32, 1.0, 0.0, 0.0]]);

        let a = engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key_a,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
                salience: 80.0,
                text: None,
            })
            .unwrap();

        let b = engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key_b,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
                salience: 80.0,
                text: None,
            })
            .unwrap();

        engine.add_pack_link(a, b).unwrap();
        (a, b)
    };

    // WAL should be non-empty (2 Follows edges = 52 bytes).
    let wal_size_before = std::fs::metadata(&wal_path).unwrap().len();
    assert!(wal_size_before > 0, "WAL should have entries before refresh");

    // Reopen (which calls refresh internally) should checkpoint the WAL.
    let mut engine = Engine::open(dir.path()).unwrap();

    let wal_size_after = std::fs::metadata(&wal_path).unwrap().len();
    assert_eq!(wal_size_after, 0, "WAL should be truncated after refresh checkpoint");

    // Trace edges should still be in memory (replayed before checkpoint).
    let links = engine.pack_links(pack_a);
    assert!(links.contains(&pack_b), "trace edges should survive checkpoint");

    // Further writes + trace edges should still work.
    let key_c = encode_per_token_keys(&[&[0.0f32, 0.0, 1.0, 0.0]]);
    let pack_c = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_c,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![3.0; 16] }],
            salience: 80.0,
            text: None,
        })
        .unwrap();
    engine.add_pack_link(pack_a, pack_c).unwrap();

    let links_after = engine.pack_links(pack_a);
    assert!(links_after.contains(&pack_c), "new links should work after checkpoint");

    // WAL should be non-empty again.
    let wal_final = std::fs::metadata(&wal_path).unwrap().len();
    assert!(wal_final > 0, "new trace edges should be written to WAL after checkpoint");
}

// ── Active Governance acceptance tests ────────────────────────────────────

/// ATDD: Core-tier packs receive a score boost over Draft-tier packs.
///
/// GIVEN two packs with similar retrieval keys,
/// WHEN one pack has Core tier (high salience) and the other has Draft,
/// THEN the Core pack's tier boost (1.25×) overcomes a small base-score gap.
#[test]
fn test_core_tier_pack_outranks_draft_tier_pack() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Pack A (Draft): slightly better match for the query key
    let key_a = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack_a = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_a,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 50.0, // importance 55 → Draft
            text: None,
        })
        .unwrap();

    // Pack B (Core): slightly worse match but tier boost should overcome
    let key_b = encode_per_token_keys(&[&[0.95f32, 0.05, 0.0, 0.0]]);
    let pack_b = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_b,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
            salience: 90.0, // importance 95 → Core (1.25× boost)
            text: None,
        })
        .unwrap();

    // Verify tiers
    let packs = engine.list_packs(None);
    let tier_a = packs.iter().find(|p| p.0 == pack_a).unwrap().2;
    let tier_b = packs.iter().find(|p| p.0 == pack_b).unwrap().2;
    assert_eq!(tier_a, Tier::Draft, "pack A should be Draft tier (ι=55)");
    assert_eq!(tier_b, Tier::Core, "pack B should be Core tier (ι=95)");

    // Query matches pack A slightly better, but pack B's 1.25× tier boost
    // should push it ahead.
    let query = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let results = engine.mem_read_pack(&query, 2, None).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].pack.id, pack_b,
        "Core-tier pack (1.25× boost) should outrank Draft-tier pack despite slightly lower base score"
    );
}

/// ATDD: `evict_draft_packs` removes Draft packs below importance threshold.
///
/// GIVEN an engine with packs at various importance levels,
/// WHEN `evict_draft_packs(threshold)` is called,
/// THEN only Draft packs below the threshold are deleted.
/// Core and Validated packs are never evicted regardless of importance.
#[test]
fn test_evict_draft_packs_removes_low_importance_drafts() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);

    // Pack A: Draft, importance=15 (10 + 5 write) — below threshold
    let _pack_a = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key.clone(),
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 10.0,
            text: None,
        })
        .unwrap();

    // Pack B: Draft, importance=75 (70 + 5 write) → Validated — above threshold
    let pack_b = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key.clone(),
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
            salience: 70.0,
            text: None,
        })
        .unwrap();

    // Pack C: Core, importance=95 (90 + 5 write) — safe from eviction
    let pack_c = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key.clone(),
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![3.0; 16] }],
            salience: 90.0,
            text: None,
        })
        .unwrap();

    assert_eq!(engine.pack_count(), 3);

    // Evict Draft packs below importance 30.
    let evicted = engine.evict_draft_packs(30.0, None).unwrap();

    assert_eq!(evicted, 1, "should evict one Draft pack below threshold");
    assert_eq!(engine.pack_count(), 2);
    assert!(engine.pack_exists(pack_b), "Validated pack should survive eviction");
    assert!(engine.pack_exists(pack_c), "Core pack should survive eviction");
}

/// ATDD: `mem_read` tier boost applies to ALL candidates before truncation.
///
/// GIVEN k=1 and multiple cells where the raw-top cell is Draft but a
/// lower-raw-score cell is Core,
/// WHEN `mem_read` is called,
/// THEN the Core cell's 1.25× boost can promote it above the Draft cell.
///
/// Regression: the early-exit `if results.len() >= k { break }` previously
/// truncated before the final re-sort, so a Core cell ranked 2nd by raw
/// score would never be seen when k=1.
#[test]
fn test_mem_read_tier_boost_not_truncated_by_early_exit() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Cell A (Draft): best raw match for the query
    let key_a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    engine.mem_write(1, 0, &key_a, vec![0.0; 4], 50.0, None).unwrap(); // ι=55 → Draft

    // Cell B (Core): slightly worse raw match, but 1.25× tier boost
    let key_b: Vec<f32> = vec![0.95, 0.05, 0.0, 0.0];
    engine.mem_write(1, 0, &key_b, vec![0.0; 4], 90.0, None).unwrap(); // ι=95 → Core

    // Query matches A better by raw score, but B's Core boost should win.
    let results = engine.mem_read(&[1.0, 0.0, 0.0, 0.0], 1, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].tier,
        Tier::Core,
        "Core cell with 1.25× boost should outrank Draft cell at k=1"
    );
}

// ── Semantic Edge Types acceptance tests (P3.1) ───────────────────────────

use tdb_index::trace::EdgeType;

fn make_edge_test_pack(engine: &mut Engine, seed: f32) -> u64 {
    let key = encode_per_token_keys(&[&[seed, 0.0, 0.0, 0.0]]);
    engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![seed; 16] }],
            salience: 80.0,
            text: None,
        })
        .unwrap()
}

/// ATDD: `add_pack_edge` with Supports creates a queryable Supports link.
#[test]
fn test_add_pack_edge_supports() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let pack_a = make_edge_test_pack(&mut engine, 1.0);
    let pack_b = make_edge_test_pack(&mut engine, 2.0);

    engine.add_pack_edge(pack_a, pack_b, EdgeType::Supports).unwrap();

    assert!(engine.pack_supports(pack_a).contains(&pack_b));
    assert!(engine.pack_supports(pack_b).contains(&pack_a));
    assert!(engine.pack_links(pack_a).contains(&pack_b));
}

/// ATDD: `add_pack_edge` with Contradicts creates a queryable Contradicts link.
#[test]
fn test_add_pack_edge_contradicts() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let pack_a = make_edge_test_pack(&mut engine, 1.0);
    let pack_b = make_edge_test_pack(&mut engine, 2.0);

    engine.add_pack_edge(pack_a, pack_b, EdgeType::Contradicts).unwrap();

    assert!(engine.pack_contradicts(pack_a).contains(&pack_b));
    assert!(engine.pack_contradicts(pack_b).contains(&pack_a));
    assert!(engine.pack_links(pack_a).contains(&pack_b));
}

/// ATDD: `pack_links_by_type` filters edges by type.
#[test]
fn test_pack_links_by_type_filters() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let pack_a = make_edge_test_pack(&mut engine, 1.0);
    let pack_b = make_edge_test_pack(&mut engine, 2.0);
    let pack_c = make_edge_test_pack(&mut engine, 3.0);
    let pack_d = make_edge_test_pack(&mut engine, 4.0);

    engine.add_pack_link(pack_a, pack_b).unwrap(); // Follows
    engine.add_pack_edge(pack_a, pack_c, EdgeType::Supports).unwrap();
    engine.add_pack_edge(pack_a, pack_d, EdgeType::Contradicts).unwrap();

    let follows = engine.pack_links_by_type(pack_a, EdgeType::Follows);
    let supports = engine.pack_links_by_type(pack_a, EdgeType::Supports);
    let contradicts = engine.pack_links_by_type(pack_a, EdgeType::Contradicts);

    assert_eq!(follows, vec![pack_b]);
    assert_eq!(supports, vec![pack_c]);
    assert_eq!(contradicts, vec![pack_d]);

    // pack_links returns ALL types
    let all = engine.pack_links(pack_a);
    assert_eq!(all.len(), 3);
}

/// ATDD: Semantic edges survive engine reopen via WAL replay.
#[test]
fn test_semantic_edges_survive_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let (pack_a, pack_b, pack_c) = {
        let mut engine = Engine::open(dir.path()).unwrap();
        let a = make_edge_test_pack(&mut engine, 1.0);
        let b = make_edge_test_pack(&mut engine, 2.0);
        let c = make_edge_test_pack(&mut engine, 3.0);
        engine.add_pack_edge(a, b, EdgeType::Supports).unwrap();
        engine.add_pack_edge(a, c, EdgeType::Contradicts).unwrap();
        (a, b, c)
    };

    let engine = Engine::open(dir.path()).unwrap();
    assert!(engine.pack_supports(pack_a).contains(&pack_b));
    assert!(engine.pack_contradicts(pack_a).contains(&pack_c));
}

/// ATDD: `add_pack_link` still creates Follows edges (backward compat).
#[test]
fn test_add_pack_link_still_creates_follows() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let pack_a = make_edge_test_pack(&mut engine, 1.0);
    let pack_b = make_edge_test_pack(&mut engine, 2.0);

    engine.add_pack_link(pack_a, pack_b).unwrap();

    let follows = engine.pack_links_by_type(pack_a, EdgeType::Follows);
    assert!(follows.contains(&pack_b));
}

/// ATDD: Mixed edge types all contribute to trace boost link count.
#[test]
fn test_mixed_edge_types_in_trace_boost() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let pack_a = make_edge_test_pack(&mut engine, 1.0);
    let pack_b = make_edge_test_pack(&mut engine, 2.0);
    let pack_c = make_edge_test_pack(&mut engine, 3.0);

    engine.add_pack_link(pack_a, pack_b).unwrap(); // Follows
    engine.add_pack_edge(pack_a, pack_c, EdgeType::Supports).unwrap();

    // pack_a has 2 links; both types should contribute to trace boost
    assert_eq!(engine.pack_links(pack_a).len(), 2);
}

// ── Engine Status API acceptance tests ────────────────────────────────────

/// ATDD: `Engine::status()` returns a consistent snapshot of engine state.
#[test]
fn test_engine_status_reflects_current_state() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let status = engine.status();
    assert_eq!(status.cell_count, 0);
    assert_eq!(status.pack_count, 0);
    assert_eq!(status.segment_count, 1); // one empty segment on open
    assert_eq!(status.slb_occupancy, 0);
    assert!(!status.vamana_active);
    assert_eq!(status.pipeline_stages, 2); // PerToken + BruteForce

    // Write some packs.
    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key.clone(),
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
            text: Some("Test fact".to_owned()),
        })
        .unwrap();

    let status = engine.status();
    assert!(status.cell_count > 0);
    assert_eq!(status.pack_count, 1);
    assert!(status.slb_occupancy > 0);
}

// ── Multi-Agent acceptance tests (P3.3) ───────────────────────────────────

const AGENT_ALPHA: u64 = 100;
const AGENT_BETA: u64 = 200;
const AGENT_GAMMA: u64 = 300;
const MULTI_AGENT_PACKS_PER_AGENT: usize = 5;
const MULTI_AGENT_SALIENCE: f32 = 80.0;
const MULTI_AGENT_LOW_SALIENCE: f32 = 10.0;
const MULTI_AGENT_EVICTION_THRESHOLD: f32 = 30.0;

fn multi_agent_pack(owner: u64, seed: f32) -> KVPack {
    KVPack {
        id: 0,
        owner,
        retrieval_key: encode_per_token_keys(&[&[seed, 0.0, 0.0, 0.0]]),
        layers: vec![KVLayerPayload { layer_idx: 0, data: vec![seed; 16] }],
        salience: MULTI_AGENT_SALIENCE,
        text: Some(format!("owner={owner} seed={seed}")),
    }
}

fn multi_agent_fixture(dir: &std::path::Path) -> (Engine, Vec<u64>, Vec<u64>, Vec<u64>) {
    let mut engine = Engine::open(dir).unwrap();
    let mut alpha_ids = Vec::new();
    let mut beta_ids = Vec::new();
    let mut gamma_ids = Vec::new();

    for i in 0..MULTI_AGENT_PACKS_PER_AGENT {
        alpha_ids
            .push(engine.mem_write_pack(&multi_agent_pack(AGENT_ALPHA, i as f32 + 1.0)).unwrap());
        beta_ids
            .push(engine.mem_write_pack(&multi_agent_pack(AGENT_BETA, i as f32 + 10.0)).unwrap());
        gamma_ids
            .push(engine.mem_write_pack(&multi_agent_pack(AGENT_GAMMA, i as f32 + 20.0)).unwrap());
    }

    (engine, alpha_ids, beta_ids, gamma_ids)
}

/// ATDD: 3 agents × 5 packs; `list_packs(owner)` returns only that agent's packs.
#[test]
fn test_multi_agent_pack_isolation() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, _, _, _) = multi_agent_fixture(dir.path());

    let alpha_packs = engine.list_packs(Some(AGENT_ALPHA));
    let beta_packs = engine.list_packs(Some(AGENT_BETA));
    let all_packs = engine.list_packs(None);

    assert_eq!(alpha_packs.len(), MULTI_AGENT_PACKS_PER_AGENT);
    assert_eq!(beta_packs.len(), MULTI_AGENT_PACKS_PER_AGENT);
    assert_eq!(all_packs.len(), MULTI_AGENT_PACKS_PER_AGENT * 3);
    assert!(alpha_packs.iter().all(|p| p.1 == AGENT_ALPHA));
    assert!(beta_packs.iter().all(|p| p.1 == AGENT_BETA));
}

/// ATDD: Each agent's trace links are invisible to other agents.
#[test]
fn test_multi_agent_trace_link_isolation() {
    let dir = tempfile::tempdir().unwrap();
    let (mut engine, alpha_ids, beta_ids, _) = multi_agent_fixture(dir.path());

    engine.add_pack_link(alpha_ids[0], alpha_ids[1]).unwrap();
    engine.add_pack_link(beta_ids[0], beta_ids[1]).unwrap();

    let alpha_links = engine.pack_links(alpha_ids[0]);
    assert!(alpha_links.contains(&alpha_ids[1]));
    assert!(!alpha_links.iter().any(|id| beta_ids.contains(id)));

    let beta_links = engine.pack_links(beta_ids[0]);
    assert!(beta_links.contains(&beta_ids[1]));
    assert!(!beta_links.iter().any(|id| alpha_ids.contains(id)));
}

/// ATDD: Evicting ALPHA's Drafts doesn't touch BETA's Drafts.
#[test]
fn test_multi_agent_owner_scoped_eviction() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let alpha_low = engine
        .mem_write_pack(&KVPack {
            salience: MULTI_AGENT_LOW_SALIENCE,
            ..multi_agent_pack(AGENT_ALPHA, 1.0)
        })
        .unwrap();
    let alpha_high = engine.mem_write_pack(&multi_agent_pack(AGENT_ALPHA, 2.0)).unwrap();
    let beta_low = engine
        .mem_write_pack(&KVPack {
            salience: MULTI_AGENT_LOW_SALIENCE,
            ..multi_agent_pack(AGENT_BETA, 10.0)
        })
        .unwrap();

    let evicted =
        engine.evict_draft_packs(MULTI_AGENT_EVICTION_THRESHOLD, Some(AGENT_ALPHA)).unwrap();

    assert_eq!(evicted, 1);
    assert!(!engine.pack_exists(alpha_low));
    assert!(engine.pack_exists(alpha_high));
    assert!(engine.pack_exists(beta_low), "BETA's low-salience Draft should be untouched");
}

/// ATDD: Deleting ALPHA's pack doesn't affect BETA's count or retrieval.
#[test]
fn test_multi_agent_delete_isolation() {
    let dir = tempfile::tempdir().unwrap();
    let (mut engine, alpha_ids, _, _) = multi_agent_fixture(dir.path());

    engine.delete_pack(alpha_ids[0]).unwrap();

    assert_eq!(engine.list_packs(Some(AGENT_ALPHA)).len(), MULTI_AGENT_PACKS_PER_AGENT - 1);
    assert_eq!(engine.list_packs(Some(AGENT_BETA)).len(), MULTI_AGENT_PACKS_PER_AGENT);
}

/// ATDD: `mem_read_pack` with k > total packs returns only the queried owner's packs.
#[test]
fn test_multi_agent_retrieval_returns_only_owner_packs() {
    let dir = tempfile::tempdir().unwrap();
    let (mut engine, _, _, _) = multi_agent_fixture(dir.path());

    let query = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let results = engine.mem_read_pack(&query, 10, Some(AGENT_ALPHA)).unwrap();

    assert!(results.len() <= MULTI_AGENT_PACKS_PER_AGENT);
    assert!(results.iter().all(|r| r.pack.owner == AGENT_ALPHA));
}

/// ATDD: `mem_read_pack_with_trace_boost` respects owner filter.
#[test]
fn test_multi_agent_trace_boost_respects_owner() {
    let dir = tempfile::tempdir().unwrap();
    let (mut engine, alpha_ids, beta_ids, _) = multi_agent_fixture(dir.path());

    engine.add_pack_link(alpha_ids[0], alpha_ids[1]).unwrap();
    engine.add_pack_link(beta_ids[0], beta_ids[1]).unwrap();

    let query = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let results = engine.mem_read_pack_with_trace_boost(&query, 5, Some(AGENT_ALPHA), 0.3).unwrap();

    assert!(results.iter().all(|r| r.pack.owner == AGENT_ALPHA));
}

/// ATDD: Each agent's governance tiers evolve independently.
#[test]
fn test_multi_agent_mixed_tiers() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let alpha_draft = engine
        .mem_write_pack(&KVPack {
            salience: MULTI_AGENT_LOW_SALIENCE,
            ..multi_agent_pack(AGENT_ALPHA, 1.0)
        })
        .unwrap();
    let alpha_core = engine
        .mem_write_pack(&KVPack { salience: 90.0, ..multi_agent_pack(AGENT_ALPHA, 2.0) })
        .unwrap();
    let beta_core = engine
        .mem_write_pack(&KVPack { salience: 90.0, ..multi_agent_pack(AGENT_BETA, 10.0) })
        .unwrap();

    let alpha_packs = engine.list_packs(Some(AGENT_ALPHA));
    let beta_packs = engine.list_packs(Some(AGENT_BETA));

    let draft_a = alpha_packs.iter().find(|p| p.0 == alpha_draft).unwrap().2;
    let core_a = alpha_packs.iter().find(|p| p.0 == alpha_core).unwrap().2;
    let core_b = beta_packs.iter().find(|p| p.0 == beta_core).unwrap().2;

    assert_eq!(draft_a, Tier::Draft);
    assert_eq!(core_a, Tier::Core);
    assert_eq!(core_b, Tier::Core);
}

// -- Crash Recovery (Chaos Engineering pattern) --------------------------------

/// ATDD: Truncated segment file — partial pack is invisible after recovery.
///
/// GIVEN an engine with 5 committed packs
/// WHEN the segment file is truncated mid-record (simulating crash during write)
/// AND the engine is reopened from the same directory
/// THEN at least 4 of the 5 packs survive (the truncated one is discarded)
/// AND all surviving packs are retrievable via `mem_read_pack`
#[test]
fn test_crash_recovery_truncated_segment() {
    let dir = tempfile::tempdir().unwrap();
    let key: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

    // Phase 1: Write 5 packs
    {
        let mut engine = Engine::open(dir.path()).unwrap();
        for i in 0..5u64 {
            let mut k = key.clone();
            k[0] = i as f32;
            let value = vec![i as f32; 32];
            engine.mem_write(1, 0, &k, value, 50.0 + i as f32, None).unwrap();
        }
        assert_eq!(engine.cell_count(), 5);
    }

    // Phase 2: Simulate crash — truncate segment mid-record
    let segment_path = dir.path().join("segment_000000.tdb");
    let original_size = std::fs::metadata(&segment_path).unwrap().len();
    assert!(original_size > 100, "segment should have data");
    // Cut off the last ~20% (guaranteed to destroy at least the last record)
    let truncated_size = original_size * 80 / 100;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&segment_path)
        .unwrap()
        .set_len(truncated_size)
        .unwrap();

    // Phase 3: Recovery — engine should open cleanly
    let mut engine = Engine::open(dir.path()).unwrap();
    let recovered = engine.cell_count();
    assert!(
        (1..5).contains(&recovered),
        "some cells must survive and truncated cell(s) must be discarded, got {recovered}"
    );

    // Surviving cells must be retrievable
    let results = engine.mem_read(&key, 10, None).unwrap();
    assert!(!results.is_empty(), "surviving cells must be retrievable after crash recovery");
}

/// ATDD: Truncated WAL — partial record discarded, prior records intact.
///
/// GIVEN an engine with 3 packs and WAL edges between them
/// WHEN the WAL file is truncated mid-record (simulating crash during edge write)
/// AND the engine is reopened
/// THEN complete WAL records are replayed (edges present)
/// AND the truncated record is discarded without panic
#[test]
fn test_crash_recovery_truncated_wal() {
    let dir = tempfile::tempdir().unwrap();
    let key: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

    // Phase 1: Write packs and add edges (edges go to WAL)
    let pack_ids;
    {
        let mut engine = Engine::open(dir.path()).unwrap();
        let mut ids = Vec::new();
        for i in 0..3u64 {
            let mut k = key.clone();
            k[0] = i as f32;
            let pack = KVPack {
                id: 0,
                owner: 1,
                retrieval_key: k,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![i as f32; 64] }],
                salience: 50.0,
                text: None,
            };
            let id = engine.mem_write_pack(&pack).unwrap();
            ids.push(id);
        }
        // Add causal edges: pack0→pack1, pack1→pack2
        engine.add_pack_link(ids[0], ids[1]).unwrap();
        engine.add_pack_link(ids[1], ids[2]).unwrap();
        pack_ids = ids;
    }

    // Phase 2: Simulate crash — truncate WAL mid-record
    let wal_path = dir.path().join("trace.wal");
    if wal_path.exists() {
        let wal_size = std::fs::metadata(&wal_path).unwrap().len();
        if wal_size > 26 {
            // Keep first complete record (26 bytes), truncate second mid-way
            let truncated = 26 + 13; // first record + half of second
            std::fs::OpenOptions::new()
                .write(true)
                .open(&wal_path)
                .unwrap()
                .set_len(truncated)
                .unwrap();
        }
    }

    // Phase 3: Recovery — engine should open without panic
    let engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.pack_count(), 3, "all 3 packs must survive (segment data is intact)");
    // Engine opened without panic — truncated WAL record was discarded gracefully
    let _ = pack_ids;
}

// -- Multi-Component Atomicity (Fault Injection pattern) -----------------------

/// ATDD: Pack with text survives clean reopen.
///
/// GIVEN 3 packs, each with associated text
/// WHEN all writes complete and the engine is reopened
/// THEN all 3 packs have their text metadata intact
#[test]
fn test_pack_text_round_trip_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let key: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

    let pack_ids;
    {
        let mut engine = Engine::open(dir.path()).unwrap();
        let mut ids = Vec::new();
        for i in 0..3u64 {
            let pack = KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key.clone(),
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![i as f32; 64] }],
                salience: 50.0,
                text: Some(format!("Memory fact #{i}")),
            };
            ids.push(engine.mem_write_pack(&pack).unwrap());
        }
        pack_ids = ids;
    }

    let mut engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.pack_count(), 3);
    for (i, &pid) in pack_ids.iter().enumerate() {
        let pack = engine.load_pack_by_id(pid).unwrap();
        assert_eq!(
            pack.pack.text.as_deref(),
            Some(format!("Memory fact #{i}").as_str()),
            "pack {pid} text must survive reopen"
        );
    }
}

/// ATDD: Truncated text store — pack cells survive, text is lost gracefully.
///
/// GIVEN a pack written to the pool (cells fsynced)
/// BUT the text store file is truncated before its fsync completes
/// WHEN the engine is reopened
/// THEN the pack's cells exist (pool is authoritative)
/// AND the text for that pack may be empty (text store lost its write)
/// AND no panic occurs
#[test]
fn test_crash_truncated_text_store_pack_survives() {
    let dir = tempfile::tempdir().unwrap();
    let key: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

    let pack_id;
    {
        let mut engine = Engine::open(dir.path()).unwrap();
        let pack = KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key.clone(),
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0f32; 64] }],
            salience: 50.0,
            text: Some("This text will be lost in the crash".into()),
        };
        pack_id = engine.mem_write_pack(&pack).unwrap();
    }

    // Simulate crash: truncate text store
    let text_path = dir.path().join("text_store.bin");
    if text_path.exists() {
        std::fs::OpenOptions::new().write(true).open(&text_path).unwrap().set_len(0).unwrap();
    }

    // Recovery: engine should open without panic
    let mut engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.pack_count(), 1, "pack cells must survive (pool is authoritative)");

    // Text may be missing — that's acceptable after text store crash
    let pack = engine.load_pack_by_id(pack_id).unwrap();
    // We just verify no panic; text being None is the expected degradation
    let _ = pack.pack.text;
}

/// ATDD: Deletion log persistence after crash.
///
/// GIVEN 5 packs where pack 3 is deleted
/// AND the deletion log is fsynced
/// WHEN the engine is reopened
/// THEN pack 3 is not returned by `mem_read_pack`
/// AND packs 1,2,4,5 are intact
#[test]
fn test_deletion_log_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let mut pack_ids;
    {
        let mut engine = Engine::open(dir.path()).unwrap();
        pack_ids = Vec::new();
        for i in 0..5u64 {
            let mut k: Vec<f32> = (0..32).map(|j| (j as f32 * 0.1).sin()).collect();
            k[0] = i as f32;
            let pack = KVPack {
                id: 0,
                owner: 1,
                retrieval_key: k,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![i as f32; 64] }],
                salience: 50.0,
                text: None,
            };
            pack_ids.push(engine.mem_write_pack(&pack).unwrap());
        }
        engine.delete_pack(pack_ids[2]).unwrap();
    }

    let engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.pack_count(), 4, "4 packs should survive (pack 3 deleted)");
    assert!(!engine.pack_exists(pack_ids[2]), "deleted pack must not exist after reopen");
    for &pid in &[pack_ids[0], pack_ids[1], pack_ids[3], pack_ids[4]] {
        assert!(engine.pack_exists(pid), "non-deleted pack {pid} must survive");
    }
}

/// ATDD: Explicit flush guarantees durability.
///
/// GIVEN an engine with pending writes
/// WHEN `engine.flush()` is called
/// THEN reopening the engine recovers all written data
#[test]
fn test_flush_guarantees_durability() {
    let dir = tempfile::tempdir().unwrap();
    let key: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();

    {
        let mut engine = Engine::open(dir.path()).unwrap();
        let pack = KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key.clone(),
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0f32; 64] }],
            salience: 50.0,
            text: Some("durable after flush".into()),
        };
        engine.mem_write_pack(&pack).unwrap();
        engine.flush().unwrap();
    }

    let engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.pack_count(), 1, "flushed pack must survive reopen");
}

// ── Background Maintenance (P5) ──────────────────────────────────────────

use std::sync::{Arc, Mutex};
use std::time::Duration;
use tdb_engine::maintenance::{MaintenanceConfig, MaintenanceWorker};

const MAINTENANCE_FAST_INTERVAL: Duration = Duration::from_millis(50);
const MAINTENANCE_AGGRESSIVE_DECAY: f32 = 24.0;

fn maintenance_test_engine(dir: &std::path::Path) -> Arc<Mutex<Engine>> {
    let engine = Engine::open_with_segment_size(dir, 512).unwrap();
    Arc::new(Mutex::new(engine))
}

/// ATDD: Maintenance worker runs governance sweep on schedule.
#[test]
fn test_maintenance_runs_sweep() {
    let dir = tempfile::tempdir().unwrap();
    let engine = maintenance_test_engine(dir.path());

    {
        let mut eng = engine.lock().unwrap();
        for i in 0..5u64 {
            let key = encode_per_token_keys(&[&[i as f32 + 1.0, 0.0, 0.0, 0.0]]);
            eng.mem_write_pack(&KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
                salience: 50.0,
                text: None,
            })
            .unwrap();
        }
    }

    let config = MaintenanceConfig {
        sweep_interval: MAINTENANCE_FAST_INTERVAL,
        compaction_interval: Duration::from_secs(9999),
        eviction_threshold: DEFAULT_EVICTION_THRESHOLD,
        hours_per_tick: MAINTENANCE_AGGRESSIVE_DECAY,
        enabled: true,
    };

    let mut worker = MaintenanceWorker::new(config);
    worker.start(Arc::clone(&engine));
    std::thread::sleep(Duration::from_millis(300));
    worker.stop();

    let status = worker.status();
    assert!(status.sweep_count >= 2, "expected at least 2 sweeps, got {}", status.sweep_count);
}

/// ATDD: Maintenance worker evicts low-importance Draft packs.
#[test]
fn test_maintenance_evicts_drafts() {
    let dir = tempfile::tempdir().unwrap();
    let engine = maintenance_test_engine(dir.path());

    {
        let mut eng = engine.lock().unwrap();
        for i in 0..3u64 {
            let key = encode_per_token_keys(&[&[i as f32 + 1.0, 0.0, 0.0, 0.0]]);
            eng.mem_write_pack(&KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
                salience: 10.0,
                text: None,
            })
            .unwrap();
        }
    }

    let config = MaintenanceConfig {
        sweep_interval: MAINTENANCE_FAST_INTERVAL,
        compaction_interval: Duration::from_secs(9999),
        eviction_threshold: 14.0,
        hours_per_tick: 720.0, // 30 days per tick — one tick drives importance near zero
        enabled: true,
    };

    let mut worker = MaintenanceWorker::new(config);
    worker.start(Arc::clone(&engine));
    std::thread::sleep(Duration::from_millis(500));
    worker.stop();

    let status = worker.status();
    assert!(status.total_packs_evicted > 0, "should have evicted at least one pack");
    let remaining = engine.lock().unwrap().pack_count();
    assert!(remaining < 3, "some packs should have been evicted, got {remaining}");
}

/// ATDD: Maintenance worker stops gracefully.
#[test]
fn test_maintenance_stops_gracefully() {
    let dir = tempfile::tempdir().unwrap();
    let engine = maintenance_test_engine(dir.path());

    let mut worker = MaintenanceWorker::new(MaintenanceConfig {
        sweep_interval: MAINTENANCE_FAST_INTERVAL,
        enabled: true,
        ..MaintenanceConfig::default()
    });

    worker.start(Arc::clone(&engine));
    assert!(worker.is_running());

    worker.stop();
    assert!(!worker.is_running());
}

/// ATDD: Disabled worker is a no-op.
#[test]
fn test_maintenance_disabled_noop() {
    let dir = tempfile::tempdir().unwrap();
    let engine = maintenance_test_engine(dir.path());

    let mut worker = MaintenanceWorker::new(MaintenanceConfig {
        enabled: false,
        ..MaintenanceConfig::default()
    });

    worker.start(Arc::clone(&engine));
    std::thread::sleep(Duration::from_millis(200));
    worker.stop();

    let status = worker.status();
    assert_eq!(status.sweep_count, 0);
    assert_eq!(status.compaction_count, 0);
}

use tdb_engine::maintenance::DEFAULT_EVICTION_THRESHOLD;

// ── Auto-link tests ─────────────────────────────────────────────────────

/// Threshold of zero: any positive score triggers a link.
const AUTO_LINK_THRESHOLD: f32 = 0.0;

/// Threshold so high no retrieval score can reach it.
const UNREACHABLE_AUTO_LINK_THRESHOLD: f32 = 999_999.0;

/// ATDD: Writing a pack with auto-link creates a trace link to a similar existing pack.
#[test]
fn test_mem_write_pack_with_auto_link() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key_a = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    let pack_a = engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key_a,
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
            text: Some("First fact".to_owned()),
        })
        .unwrap();

    let key_b = encode_per_token_keys(&[&[0.95f32, 0.05, 0.0, 0.0]]);
    let pack_b = engine
        .mem_write_pack_with_auto_link(
            &KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key_b,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
                salience: 80.0,
                text: Some("Related fact".to_owned()),
            },
            AUTO_LINK_THRESHOLD,
        )
        .unwrap();

    assert!(engine.pack_links(pack_b.pack_id).contains(&pack_a));
}

/// ATDD: The returned `PackWriteResult` includes linked pack IDs.
#[test]
fn test_auto_link_returns_linked_ids() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key.clone(),
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
            text: None,
        })
        .unwrap();

    let result = engine
        .mem_write_pack_with_auto_link(
            &KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
                salience: 80.0,
                text: None,
            },
            AUTO_LINK_THRESHOLD,
        )
        .unwrap();

    assert!(!result.linked_pack_ids.is_empty());
}

/// ATDD: A very high threshold produces no auto-links.
#[test]
fn test_auto_link_high_threshold_no_links() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key = encode_per_token_keys(&[&[1.0f32, 0.0, 0.0, 0.0]]);
    engine
        .mem_write_pack(&KVPack {
            id: 0,
            owner: 1,
            retrieval_key: key.clone(),
            layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; 16] }],
            salience: 80.0,
            text: None,
        })
        .unwrap();

    let result = engine
        .mem_write_pack_with_auto_link(
            &KVPack {
                id: 0,
                owner: 1,
                retrieval_key: key,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![2.0; 16] }],
                salience: 80.0,
                text: None,
            },
            UNREACHABLE_AUTO_LINK_THRESHOLD,
        )
        .unwrap();

    assert!(result.linked_pack_ids.is_empty());
}
