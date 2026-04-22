use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tdb_index::wal::{Wal, WalEntry};

fn bench_wal_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("WAL append — fsync'd causal edge writes");

    for count in [100, 1000] {
        group.bench_with_input(BenchmarkId::new("edges", count), &count, |bench, &count| {
            bench.iter(|| {
                let dir = tempfile::tempdir().unwrap();
                let mut wal = Wal::open(dir.path()).unwrap();
                for i in 0..count {
                    wal.append(&WalEntry::AddEdge {
                        src: i,
                        dst: i + 1,
                        edge_type: 0,
                        timestamp: i * 1000,
                    })
                    .unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_wal_replay(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let mut wal = Wal::open(dir.path()).unwrap();
    for i in 0..1000u64 {
        wal.append(&WalEntry::AddEdge { src: i, dst: i + 1, edge_type: 0, timestamp: i * 1000 })
            .unwrap();
    }

    c.bench_function("WAL replay — crash recovery: read 1K edges from disk", |bench| {
        bench.iter(|| {
            let wal = Wal::open(dir.path()).unwrap();
            let entries = wal.replay().unwrap();
            assert_eq!(entries.len(), 1000);
        });
    });
}

criterion_group!(benches, bench_wal_append, bench_wal_replay);
criterion_main!(benches);
