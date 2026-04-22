use criterion::{Criterion, criterion_group, criterion_main};
use tdb_core::memory_cell::MemoryCellBuilder;
use tdb_storage::block_pool::BlockPool;

fn bench_append(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open(dir.path()).unwrap();
    let mut id = 0u64;

    c.bench_function(
        "Block pool append — Q4-compress and fsync one cell to segment (dim=128)",
        |b| {
            b.iter(|| {
                let cell = MemoryCellBuilder::new(id, 1, 0, vec![1.0; 128], vec![2.0; 128]).build();
                pool.append(&cell).unwrap();
                id += 1;
            });
        },
    );
}

fn bench_random_read(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open(dir.path()).unwrap();

    // Pre-populate with 10K cells.
    for i in 0..10_000u64 {
        let cell = MemoryCellBuilder::new(i, 1, 0, vec![1.0; 128], vec![2.0; 128]).build();
        pool.append(&cell).unwrap();
    }

    let mut idx = 0u64;
    c.bench_function(
        "Block pool random read — dequantize one cell from 10K on disk (dim=128)",
        |b| {
            b.iter(|| {
                pool.get(idx % 10_000).unwrap();
                idx = idx.wrapping_add(7); // pseudo-random stride
            });
        },
    );
}

criterion_group!(benches, bench_append, bench_random_read);
criterion_main!(benches);
