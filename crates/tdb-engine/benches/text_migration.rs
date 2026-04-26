//! Benchmark: single-fsync-per-entry vs batched-single-fsync for text migration.
//!
//! The legacy migration path (`Engine::set_pack_text` called N times) does N
//! fsyncs. The batched path (`Engine::set_pack_texts` once with N entries)
//! does 1 fsync. This bench quantifies the speedup at typical migration
//! sizes — which on local SSD is dominated by fsync latency (~ms each).
//!
//! Run with:
//!
//! ```bash
//! cargo bench -p tdb-engine --bench text_migration
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::engine::Engine;
use tdb_retrieval::per_token::encode_per_token_keys;

const ENTRY_COUNTS: [usize; 3] = [100, 1_000, 10_000];
const OWNER: u64 = 1;
const SALIENCE: f32 = 80.0;
const PAYLOAD_DIM: usize = 16;

/// Build an engine pre-populated with `n` packs that have no text yet.
/// Returns the engine and the assigned pack IDs.
fn build_engine_with_packs(n: usize) -> (tempfile::TempDir, Engine, Vec<u64>) {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let key = encode_per_token_keys(&[&[(i as f32) * 0.001, 0.0, 0.0, 0.0]]);
        let id = engine
            .mem_write_pack(&KVPack {
                id: 0,
                owner: OWNER,
                retrieval_key: key,
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![1.0; PAYLOAD_DIM] }],
                salience: SALIENCE,
                text: None,
            })
            .unwrap();
        ids.push(id);
    }
    (dir, engine, ids)
}

fn bench_text_migration(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_migration");
    group.sample_size(10); // each sample does up to 10K fsyncs — keep wall-clock sane

    for &n in &ENTRY_COUNTS {
        // Single-fsync-per-entry: the pre-batch baseline.
        group.bench_with_input(BenchmarkId::new("single_fsync_per_entry", n), &n, |bench, &n| {
            bench.iter_with_setup(
                || build_engine_with_packs(n),
                |(dir, mut engine, ids)| {
                    for (i, id) in ids.iter().enumerate() {
                        engine.set_pack_text(*id, black_box(&format!("text-{i}"))).unwrap();
                    }
                    drop(engine);
                    drop(dir);
                },
            );
        });

        // Batched single-fsync: the new path.
        group.bench_with_input(BenchmarkId::new("batched_single_fsync", n), &n, |bench, &n| {
            bench.iter_with_setup(
                || build_engine_with_packs(n),
                |(dir, mut engine, ids)| {
                    let entries: Vec<(u64, String)> =
                        ids.iter().enumerate().map(|(i, id)| (*id, format!("text-{i}"))).collect();
                    let borrowed: Vec<(u64, &str)> =
                        entries.iter().map(|(id, t)| (*id, t.as_str())).collect();
                    engine.set_pack_texts(black_box(&borrowed)).unwrap();
                    drop(engine);
                    drop(dir);
                },
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_text_migration);
criterion_main!(benches);
