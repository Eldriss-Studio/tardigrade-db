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
    assert!(
        avg_us < 50_000.0,
        "Average query latency {avg_us:.0}μs exceeds 50ms"
    );
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
    assert!(
        ancestors.contains(&0),
        "Ancestor A(0) not found. Got: {ancestors:?}"
    );
    assert!(
        ancestors.contains(&1),
        "Ancestor B(1) not found. Got: {ancestors:?}"
    );
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
                assert!(
                    !neighbors.is_empty() || i >= 100,
                    "Expected neighbors for node {i}"
                );
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
    assert!(
        t.edge_count() >= 200,
        "Expected ≥200 edges, got {}",
        t.edge_count()
    );
}
