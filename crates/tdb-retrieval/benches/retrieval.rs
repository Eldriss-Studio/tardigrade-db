use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tdb_retrieval::int8_quant::Int8Quantizer;
use tdb_retrieval::per_token::{PerTokenRetriever, ScoringMode, encode_per_token_keys};
use tdb_retrieval::retriever::Retriever;
use tdb_retrieval::simd_distance::DotProduct;

fn bench_f32_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("FP32 dot product — baseline attention score");
    for dim in [64, 128, 256, 512] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();
        group.bench_with_input(BenchmarkId::new("dim", dim), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| DotProduct::f32_dot(black_box(a), black_box(b)));
        });
    }
    group.finish();
}

fn bench_int8_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("INT8 dot product — NEON-accelerated attention score");
    for dim in [64, 128, 256, 512] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();
        let qa = Int8Quantizer::quantize(&a);
        let qb = Int8Quantizer::quantize(&b);
        group.bench_with_input(BenchmarkId::new("dim", dim), &(&qa, &qb), |bench, (qa, qb)| {
            bench.iter(|| DotProduct::int8_dot(black_box(qa), black_box(qb)));
        });
    }
    group.finish();
}

fn bench_slb_query(c: &mut Criterion) {
    use tdb_retrieval::slb::SemanticLookasideBuffer;

    let mut group = c.benchmark_group("SLB query — hot-path INT8 cache lookup (top-5)");
    for capacity in [256, 1024, 4096] {
        let dim = 128;
        let mut slb = SemanticLookasideBuffer::new(capacity, dim);
        for i in 0..capacity as u64 {
            let key: Vec<f32> =
                (0..dim).map(|d| ((i * 3 + d as u64) as f32 * 0.01).sin()).collect();
            slb.insert(i, 1, &key);
        }
        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.02).cos()).collect();

        group.bench_function(BenchmarkId::new("entries", capacity), |bench| {
            bench.iter(|| {
                let _ = slb.query(black_box(&query), 5);
            });
        });
    }
    group.finish();
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

fn bench_per_token_top5avg_query(c: &mut Criterion) {
    // Template Method benchmark flow:
    // build deterministic encoded corpus → populate strategy → query → measure.
    let mut group = c.benchmark_group("PerTokenRetriever Top5Avg query — encoded per-token keys");
    let dim = 128;
    let tokens_per_cell = 8;

    for cell_count in [100usize, 1_000, 10_000] {
        let mut retriever = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        for cell_id in 0..cell_count {
            let encoded = encoded_cell(dim, cell_id, tokens_per_cell);
            retriever.insert(cell_id as u64, 1, &encoded);
        }
        let query = encoded_cell(dim, cell_count / 2, tokens_per_cell);

        group.bench_function(BenchmarkId::new("cells", cell_count), |bench| {
            bench.iter(|| {
                let _ = retriever.query(black_box(&query), 5, Some(1));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_f32_dot,
    bench_int8_dot,
    bench_slb_query,
    bench_per_token_top5avg_query
);
criterion_main!(benches);
