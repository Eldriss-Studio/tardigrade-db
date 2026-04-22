use std::sync::{Arc, Barrier};
use std::thread;

use tdb_core::CellId;
use tdb_index::trace::{EdgeType, TraceGraph};
use tdb_index::vamana::VamanaIndex;
use tdb_index::wal::{Wal, WalEntry};

/// ATDD Test 1: Insert 1K cells into Vamana. Query with k=10.
/// Assert approximate best score is ≥80% of exact best score.
#[test]
fn test_vamana_recall_at_1k() {
    let dim = 32;
    let n = 1_000;
    let max_degree = 16;
    let mut index = VamanaIndex::new(dim, max_degree);

    // Generate well-separated vectors: each cell has a dominant dimension.
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut v = vec![0.01f32; dim];
            v[i % dim] += 1.0; // primary signal
            v[(i * 7) % dim] += (i as f32) * 0.001; // secondary spread
            v
        })
        .collect();

    for (i, vec) in vectors.iter().enumerate() {
        index.insert(i as CellId, vec);
    }
    index.build();

    let query = &vectors[500];

    // Brute-force exact top-10.
    let mut exact: Vec<(CellId, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            (i as CellId, dot)
        })
        .collect();
    exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    // Vamana approximate top-10.
    let approx = index.query(query, 10);

    // Verify quality: the approximate best score should be at least 80% of the exact best.
    let exact_best_score = exact[0].1;
    let approx_best_score = approx[0].1;
    let score_ratio = approx_best_score / exact_best_score;
    assert!(
        score_ratio >= 0.8,
        "Best approximate score ({approx_best_score:.4}) is only {score_ratio:.2}x \
         of exact best ({exact_best_score:.4}). Approx IDs: {:?}",
        approx.iter().map(|r| r.0).collect::<Vec<_>>()
    );

    // Also verify we get 10 results.
    assert_eq!(approx.len(), 10, "Expected 10 results from Vamana query");
}

/// ATDD Test 2: Insert 1K cells, measure query latency.
/// Assert queries complete in reasonable time (<50ms in debug mode).
#[test]
fn test_vamana_query_latency() {
    let dim = 32;
    let n = 1_000;
    let mut index = VamanaIndex::new(dim, 16);

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut v = vec![0.01f32; dim];
            v[i % dim] += 1.0;
            v
        })
        .collect();

    for (i, vec) in vectors.iter().enumerate() {
        index.insert(i as CellId, vec);
    }

    let query = &vectors[500];

    // Warm up.
    for _ in 0..10 {
        let _ = index.query(query, 10);
    }

    // Measure.
    let start = std::time::Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        let _ = index.query(query, 10);
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;

    // In debug mode, just verify it completes in reasonable time (<50ms per query).
    assert!(avg_us < 50_000.0, "Average query latency {avg_us:.0}μs exceeds 50ms");
}

/// ATDD Test 3: Insert cells A → B → C with causal edges.
/// Query ancestors of C. Assert returns {A, B}.
#[test]
fn test_trace_causal_chain() {
    let mut trace = TraceGraph::new();

    // A(0) → B(1) → C(2)
    trace.add_edge(1, 0, EdgeType::CausedBy, 1000); // B caused by A
    trace.add_edge(2, 1, EdgeType::CausedBy, 2000); // C caused by B

    // Ancestors of C (following CausedBy edges transitively).
    let ancestors = trace.ancestors(2, EdgeType::CausedBy);
    assert!(ancestors.contains(&0), "Ancestor A(0) not found. Got: {ancestors:?}");
    assert!(ancestors.contains(&1), "Ancestor B(1) not found. Got: {ancestors:?}");
    assert_eq!(ancestors.len(), 2);
}

/// ATDD Test 4: Write WAL entries, simulate crash (drop without checkpoint),
/// recover from WAL, assert all entries are replayed.
#[test]
fn test_wal_crash_recovery() {
    let dir = tempfile::tempdir().unwrap();

    // Write phase: log 100 edge additions.
    {
        let mut wal = Wal::open(dir.path()).unwrap();
        for i in 0..100u64 {
            wal.append(&WalEntry::AddEdge {
                src: i,
                dst: i + 1,
                edge_type: 0, // CausedBy
                timestamp: i * 1000,
            })
            .unwrap();
        }
        // Drop without checkpoint — simulates crash.
    }

    // Recovery phase: replay WAL entries.
    {
        let wal = Wal::open(dir.path()).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(
            entries.len(),
            100,
            "Expected 100 WAL entries after recovery, got {}",
            entries.len()
        );

        // Verify first and last entries.
        match &entries[0] {
            WalEntry::AddEdge { src, dst, .. } => {
                assert_eq!(*src, 0);
                assert_eq!(*dst, 1);
            }
        }
        match &entries[99] {
            WalEntry::AddEdge { src, dst, .. } => {
                assert_eq!(*src, 99);
                assert_eq!(*dst, 100);
            }
        }
    }

    // Checkpoint: clear the WAL.
    {
        let mut wal = Wal::open(dir.path()).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 100); // still 100 before checkpoint

        wal.checkpoint().unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 0, "WAL should be empty after checkpoint");
    }
}

/// ATDD Test 5: Concurrent reads and writes on the Trace graph.
/// 16 reader threads + 1 writer thread. Assert no panics and data integrity.
#[test]
fn test_concurrent_trace_reads() {
    let trace = Arc::new(std::sync::RwLock::new(TraceGraph::new()));

    // Pre-populate with some edges.
    {
        let mut t = trace.write().unwrap();
        for i in 0..100u64 {
            t.add_edge(i, i + 1, EdgeType::Follows, i * 100);
        }
    }

    let barrier = Arc::new(Barrier::new(17)); // 16 readers + 1 writer
    let mut handles = Vec::new();

    // 16 reader threads.
    for _ in 0..16 {
        let trace = Arc::clone(&trace);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 0..50u64 {
                let t = trace.read().unwrap();
                let neighbors = t.outgoing(i, None);
                // Should find at least one neighbor (the Follows edge).
                assert!(!neighbors.is_empty() || i >= 100, "Expected neighbors for node {i}");
            }
        }));
    }

    // 1 writer thread — adds more edges concurrently.
    {
        let trace = Arc::clone(&trace);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 100..200u64 {
                let mut t = trace.write().unwrap();
                t.add_edge(i, i + 1, EdgeType::Follows, i * 100);
            }
        }));
    }

    for h in handles {
        h.join().expect("Thread panicked");
    }

    // Verify all edges present.
    let t = trace.read().unwrap();
    assert!(t.edge_count() >= 200, "Expected ≥200 edges, got {}", t.edge_count());
}

// ── Phase 9: Incremental Vamana Build (Strategy + Template Method) ────────

/// ATDD Test 6: Insert 500 nodes one-by-one via `insert_online()`.
/// Every node has ≥1 neighbor. None exceeds `max_degree`.
#[test]
fn test_incremental_insert_maintains_connectivity() {
    let dim = 16;
    let max_degree = 8;
    let mut index = VamanaIndex::new(dim, max_degree);

    for i in 0..500u64 {
        let mut v = vec![0.01f32; dim];
        v[(i as usize) % dim] = 1.0;
        index.insert_online(i, &v);
    }

    // Every node should have at least 1 neighbor (except possibly the very first).
    for i in 1..500u64 {
        let neighbors = index.neighbor_count(i);
        assert!(neighbors >= 1, "Node {i} has 0 neighbors after incremental insert");
        assert!(neighbors <= max_degree, "Node {i} has {neighbors} neighbors (max {max_degree})");
    }
}

/// ATDD Test 7: 1K nodes via `insert_online()`. Approximate best score ≥80% of exact.
#[test]
fn test_incremental_recall_at_1k() {
    let dim = 32;
    let mut index = VamanaIndex::new(dim, 16);

    let vectors: Vec<Vec<f32>> = (0..1000)
        .map(|i| {
            let mut v = vec![0.01f32; dim];
            v[i % dim] += 1.0;
            v[(i * 7) % dim] += (i as f32) * 0.001;
            v
        })
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        index.insert_online(i as CellId, v);
    }

    let query = &vectors[500];

    // Brute-force exact best.
    let exact_best: f32 = vectors
        .iter()
        .map(|v| query.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>())
        .fold(f32::NEG_INFINITY, f32::max);

    let approx = index.query(query, 10);
    let approx_best = approx[0].1;
    let ratio = approx_best / exact_best;

    assert!(
        ratio >= 0.8,
        "Incremental Vamana: approx best ({approx_best:.4}) is only {ratio:.2}x of exact ({exact_best:.4})"
    );
}

/// ATDD Test 8: Given 20 clustered candidates, `robust_prune` with R=8 should produce
/// angularly diverse neighbors (no two with dot product > 0.99).
#[test]
fn test_robust_prune_angular_diversity() {
    use tdb_index::vamana::prune::robust_prune;

    let dim = 16;
    // Node at origin-ish.
    let node_vec = vec![1.0f32; dim];

    // 20 candidates: 10 in cluster A (near [1,0,...]), 10 in cluster B (near [0,1,...]).
    let mut candidate_vecs: Vec<Vec<f32>> = Vec::new();
    for i in 0..10 {
        let mut v = vec![0.0f32; dim];
        v[0] = 1.0 + (i as f32) * 0.001; // cluster A: slight variations
        candidate_vecs.push(v);
    }
    for i in 0..10 {
        let mut v = vec![0.0f32; dim];
        v[1] = 1.0 + (i as f32) * 0.001; // cluster B: slight variations
        candidate_vecs.push(v);
    }

    let candidate_indices: Vec<usize> = (0..20).collect();
    let all_vecs: Vec<&[f32]> = candidate_vecs.iter().map(Vec::as_slice).collect();

    let selected = robust_prune(&node_vec, &candidate_indices, &all_vecs, 1.2, 8);

    assert_eq!(selected.len(), 8, "Should select exactly R=8 neighbors");

    // Check diversity: selected should include neighbors from BOTH clusters.
    let from_a = selected.iter().filter(|&&idx| idx < 10).count();
    let from_b = selected.iter().filter(|&&idx| idx >= 10).count();
    assert!(
        from_a >= 1 && from_b >= 1,
        "Robust prune should select from both clusters. A={from_a}, B={from_b}"
    );
}

/// ATDD Test 9: 200 nodes built both ways. Recall difference <20%.
#[test]
fn test_incremental_vs_batch_parity() {
    let dim = 16;
    let n = 200;

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut v = vec![0.01f32; dim];
            v[i % dim] = 1.0;
            v
        })
        .collect();

    // Batch build.
    let mut batch_idx = VamanaIndex::new(dim, 8);
    for (i, v) in vectors.iter().enumerate() {
        batch_idx.insert(i as CellId, v);
    }
    batch_idx.build();

    // Incremental build.
    let mut inc_idx = VamanaIndex::new(dim, 8);
    for (i, v) in vectors.iter().enumerate() {
        inc_idx.insert_online(i as CellId, v);
    }

    let query = &vectors[100];

    let batch_results = batch_idx.query(query, 10);
    let inc_results = inc_idx.query(query, 10);

    let batch_best = batch_results[0].1;
    let inc_best = inc_results[0].1;

    // Both should find similar quality results.
    let ratio = inc_best / batch_best;
    assert!(
        ratio >= 0.8,
        "Incremental best ({inc_best:.4}) is only {ratio:.2}x of batch best ({batch_best:.4})"
    );
}

/// ATDD Test 10: Duplicate `CellId` via `insert_online` should panic.
#[test]
#[should_panic(expected = "duplicate CellId")]
fn test_insert_online_duplicate_panics() {
    let mut index = VamanaIndex::new(4, 8);
    index.insert_online(0, &[1.0, 0.0, 0.0, 0.0]);
    index.insert_online(0, &[0.0, 1.0, 0.0, 0.0]); // should panic
}
