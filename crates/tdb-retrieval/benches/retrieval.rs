use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tdb_retrieval::int8_quant::Int8Quantizer;
use tdb_retrieval::simd_distance::DotProduct;

fn bench_f32_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_dot");
    for dim in [64, 128, 256, 512] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(dim), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| DotProduct::f32_dot(black_box(a), black_box(b)));
        });
    }
    group.finish();
}

fn bench_int8_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_dot");
    for dim in [64, 128, 256, 512] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();
        let qa = Int8Quantizer::quantize(&a);
        let qb = Int8Quantizer::quantize(&b);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &(&qa, &qb), |bench, (qa, qb)| {
            bench.iter(|| DotProduct::int8_dot(black_box(qa), black_box(qb)));
        });
    }
    group.finish();
}

fn bench_slb_query(c: &mut Criterion) {
    use tdb_retrieval::slb::SemanticLookasideBuffer;

    let mut group = c.benchmark_group("slb_query");
    for capacity in [256, 1024, 4096] {
        let dim = 128;
        let mut slb = SemanticLookasideBuffer::new(capacity, dim);
        for i in 0..capacity as u64 {
            let key: Vec<f32> =
                (0..dim).map(|d| ((i * 3 + d as u64) as f32 * 0.01).sin()).collect();
            slb.insert(i, 1, &key);
        }
        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.02).cos()).collect();

        group.bench_function(BenchmarkId::from_parameter(capacity), |bench| {
            bench.iter(|| {
                let _ = slb.query(black_box(&query), 5);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_f32_dot, bench_int8_dot, bench_slb_query);
criterion_main!(benches);
