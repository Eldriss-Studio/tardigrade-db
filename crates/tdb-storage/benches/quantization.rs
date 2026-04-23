use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tdb_storage::quantization::{DequantizeStrategy, Q4, QuantizeStrategy};

fn bench_q4_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("Q4 quantize — compress f32 → 4-bit (GGML Q4_0)");
    for size in [64, 128, 256, 512, 1024] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        group.bench_with_input(BenchmarkId::new("floats", size), &data, |b, data| {
            b.iter(|| Q4::quantize(black_box(data)));
        });
    }
    group.finish();
}

fn bench_q4_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("Q4 dequantize — decompress 4-bit → f32");
    for size in [64, 128, 256, 512, 1024] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        let quantized = Q4::quantize(&data);
        group.bench_with_input(BenchmarkId::new("floats", size), &quantized, |b, q| {
            b.iter(|| Q4::dequantize(black_box(q)));
        });
    }
    group.finish();
}

fn bench_q4_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("Q4 round-trip — quantize + dequantize end-to-end");
    for size in [128, 256, 512] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        group.bench_with_input(BenchmarkId::new("floats", size), &data, |b, data| {
            b.iter(|| {
                let q = Q4::quantize(black_box(data));
                Q4::dequantize(&q)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_q4_quantize, bench_q4_dequantize, bench_q4_round_trip);
criterion_main!(benches);
