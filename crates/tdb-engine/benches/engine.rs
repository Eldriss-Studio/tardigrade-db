use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::engine::{Engine, WriteRequest};
use tdb_retrieval::per_token::encode_per_token_keys;

fn bench_engine_write(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();
    let value: Vec<f32> = vec![0.0; 64];

    c.bench_function("Engine mem_write — single cell persist with fsync (dim=64)", |bench| {
        bench.iter(|| {
            engine.mem_write(1, 0, black_box(&key), value.clone(), 50.0, None).unwrap();
        });
    });
}

fn bench_engine_read(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Pre-populate with 1000 cells.
    for i in 0..1000u64 {
        let mut key = vec![0.01f32; 64];
        key[(i as usize) % 64] = 1.0;
        engine.mem_write(1, 0, &key, vec![0.0; 64], 50.0, None).unwrap();
    }

    let mut query = vec![0.01f32; 64];
    query[10] = 1.0;

    c.bench_function(
        "Engine mem_read — full pipeline: SLB → retriever → governance (1K cells, dim=64, top-5)",
        |bench| {
            bench.iter(|| {
                let _ = engine.mem_read(black_box(&query), 5, None).unwrap();
            });
        },
    );
}

fn token_vector(dim: usize, cell_id: usize, token_id: usize) -> Vec<f32> {
    (0..dim)
        .map(|d| {
            let phase = (cell_id * 31 + token_id * 17 + d * 7) as f32 * 0.013;
            phase.sin() * 0.5 + phase.cos() * 0.25
        })
        .collect()
}

fn encoded_cell(dim: usize, cell_id: usize, tokens_per_cell: usize) -> Vec<f32> {
    let tokens: Vec<Vec<f32>> =
        (0..tokens_per_cell).map(|token_id| token_vector(dim, cell_id, token_id)).collect();
    let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
    encode_per_token_keys(&refs)
}

fn bench_engine_read_encoded_per_token(c: &mut Criterion) {
    // Template Method benchmark flow:
    // build deterministic encoded corpus → batch populate engine → query → measure.
    let mut group = c.benchmark_group("Engine mem_read — encoded per-token Top5Avg path");
    let dim = 128;
    let tokens_per_cell = 8;

    for cell_count in [100usize, 1_000, 10_000] {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        let requests: Vec<WriteRequest> = (0..cell_count)
            .map(|cell_id| WriteRequest {
                owner: 1,
                layer: 0,
                key: encoded_cell(dim, cell_id, tokens_per_cell),
                value: vec![0.0; dim],
                salience: 50.0,
                parent_cell_id: None,
            })
            .collect();
        engine.mem_write_batch(&requests).unwrap();
        let query = encoded_cell(dim, cell_count / 2, tokens_per_cell);

        group.bench_function(BenchmarkId::new("cells", cell_count), |bench| {
            bench.iter(|| {
                let _ = engine.mem_read(black_box(&query), 5, Some(1)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_engine_read_pack_encoded_per_token(c: &mut Criterion) {
    // Fixture Builder / Object Mother: each pack has one small layer payload;
    // retrieval key shape matches the validated hidden-state Top5Avg path.
    let mut group = c.benchmark_group("Engine mem_read_pack — encoded per-token Top5Avg path");
    let dim = 128;
    let tokens_per_cell = 8;

    for pack_count in [100usize, 1_000, 10_000] {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        for pack_id in 0..pack_count {
            let pack = KVPack {
                id: 0,
                owner: 1,
                retrieval_key: encoded_cell(dim, pack_id, tokens_per_cell),
                layers: vec![KVLayerPayload { layer_idx: 0, data: vec![pack_id as f32; dim] }],
                salience: 80.0,
            };
            engine.mem_write_pack(&pack).unwrap();
        }
        let query = encoded_cell(dim, pack_count / 2, tokens_per_cell);

        group.bench_function(BenchmarkId::new("packs", pack_count), |bench| {
            bench.iter(|| {
                let _ = engine.mem_read_pack(black_box(&query), 5, Some(1)).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_engine_write,
    bench_engine_read,
    bench_engine_read_encoded_per_token,
    bench_engine_read_pack_encoded_per_token
);
criterion_main!(benches);
