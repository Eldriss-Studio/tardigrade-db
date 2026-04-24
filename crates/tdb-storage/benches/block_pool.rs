use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tdb_core::memory_cell::MemoryCellBuilder;
use tdb_storage::block_pool::BlockPool;

const APPEND_OWNER: u64 = 1;
const APPEND_LAYER: u16 = 0;
const APPEND_DIM: usize = 128;
const APPEND_KEY_VALUE: f32 = 1.0;
const APPEND_PAYLOAD_VALUE: f32 = 2.0;
const RANDOM_READ_CELL_COUNT: u64 = 10_000;
const RANDOM_READ_DIM: usize = 128;
const RANDOM_READ_STRIDE: u64 = 7;
const HYDRATION_CELL_COUNTS: [u64; 3] = [100, 1_000, 10_000];
const HYDRATION_PAYLOAD_DIMS: [usize; 2] = [128, 256];
const HYDRATION_KEY_DIM: usize = 128;
const HYDRATION_READ_STRIDE: u64 = 7;

#[derive(Debug, Clone, Copy)]
struct StorageHydrationLabel {
    payload_dim: usize,
    cell_count: usize,
}

impl StorageHydrationLabel {
    fn as_benchmark_id(self) -> String {
        format!("payload-{}-cells-{}", self.payload_dim, self.cell_count)
    }
}

fn bench_append(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let mut pool = BlockPool::open(dir.path()).unwrap();
    let mut id = 0u64;

    c.bench_function(
        "Block pool append — Q4-compress and fsync one cell to segment (dim=128)",
        |b| {
            b.iter(|| {
                let cell = MemoryCellBuilder::new(
                    id,
                    APPEND_OWNER,
                    APPEND_LAYER,
                    vec![APPEND_KEY_VALUE; APPEND_DIM],
                    vec![APPEND_PAYLOAD_VALUE; APPEND_DIM],
                )
                .build();
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
    for i in 0..RANDOM_READ_CELL_COUNT {
        let cell = MemoryCellBuilder::new(
            i,
            APPEND_OWNER,
            APPEND_LAYER,
            vec![APPEND_KEY_VALUE; RANDOM_READ_DIM],
            vec![APPEND_PAYLOAD_VALUE; RANDOM_READ_DIM],
        )
        .build();
        pool.append(&cell).unwrap();
    }

    let mut idx = 0u64;
    c.bench_function(
        "Block pool random read — dequantize one cell from 10K on disk (dim=128)",
        |b| {
            b.iter(|| {
                pool.get(idx % RANDOM_READ_CELL_COUNT).unwrap();
                idx = idx.wrapping_add(RANDOM_READ_STRIDE);
            });
        },
    );
}

fn bench_block_pool_get_hydration(c: &mut Criterion) {
    let mut group = c.benchmark_group("BlockPool get — storage hydration");

    for cell_count in HYDRATION_CELL_COUNTS {
        for payload_dim in HYDRATION_PAYLOAD_DIMS {
            let dir = tempfile::tempdir().unwrap();
            let mut pool = BlockPool::open(dir.path()).unwrap();

            for id in 0..cell_count {
                let cell = MemoryCellBuilder::new(
                    id,
                    APPEND_OWNER,
                    APPEND_LAYER,
                    vec![APPEND_KEY_VALUE; HYDRATION_KEY_DIM],
                    vec![APPEND_PAYLOAD_VALUE; payload_dim],
                )
                .build();
                pool.append(&cell).unwrap();
            }

            let label = StorageHydrationLabel { payload_dim, cell_count: cell_count as usize };
            let mut idx = 0u64;

            group.bench_function(BenchmarkId::new(label.as_benchmark_id(), cell_count), |b| {
                b.iter(|| {
                    let cell = pool.get(black_box(idx % cell_count)).unwrap();
                    black_box(cell.value.len());
                    idx = idx.wrapping_add(HYDRATION_READ_STRIDE);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_append, bench_random_read, bench_block_pool_get_hydration);
criterion_main!(benches);
