use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tdb_engine::engine::Engine;

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

criterion_group!(benches, bench_engine_write, bench_engine_read);
criterion_main!(benches);
