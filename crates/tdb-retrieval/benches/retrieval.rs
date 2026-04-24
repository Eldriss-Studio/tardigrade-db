use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tdb_retrieval::int8_quant::Int8Quantizer;
use tdb_retrieval::per_token::{PerTokenRetriever, ScoringMode, encode_per_token_keys};
use tdb_retrieval::retriever::Retriever;
use tdb_retrieval::simd_distance::DotProduct;

const FIXTURE_DIM: usize = 128;
const FIXTURE_TOKENS_PER_CELL: usize = 8;
const DOT_PRODUCT_DIMS: [usize; 4] = [64, 128, 256, 512];
const SLB_CAPACITIES: [usize; 3] = [256, 1024, 4096];
const SLB_BENCH_DIM: usize = 128;
const FIXTURE_QUERY_COUNT: usize = 30;
const FIXTURE_QUERY_STRIDE: usize = 7;
const FIXTURE_QUERY_OFFSET: usize = 3;
const BENCH_TOP_K: usize = 5;
const OWNER_ONE: u64 = 1;
const BENCH_CELL_COUNTS: [usize; 3] = [100, 1_000, 10_000];
const CANDIDATE_REDUCTION_THRESHOLD: usize = 512;
const MIN_CANDIDATES: usize = 256;
const CANDIDATE_MULTIPLIER: usize = 64;
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const CELL_SEED_MULTIPLIER: u64 = 0x9E37_79B1_85EB_CA87;
const TOKEN_SEED_MULTIPLIER: u64 = 0xC2B2_AE3D_27D4_EB4F;
const RANDOM_SHIFT: u32 = 40;
const RANDOM_DENOMINATOR: f32 = (1u64 << 24) as f32;

fn bench_f32_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("FP32 dot product — baseline attention score");
    for dim in DOT_PRODUCT_DIMS {
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
    for dim in DOT_PRODUCT_DIMS {
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
    for capacity in SLB_CAPACITIES {
        let mut slb = SemanticLookasideBuffer::new(capacity, SLB_BENCH_DIM);
        for i in 0..capacity as u64 {
            let key: Vec<f32> =
                (0..SLB_BENCH_DIM).map(|d| ((i * 3 + d as u64) as f32 * 0.01).sin()).collect();
            slb.insert(i, OWNER_ONE, &key);
        }
        let query: Vec<f32> = (0..SLB_BENCH_DIM).map(|d| (d as f32 * 0.02).cos()).collect();

        group.bench_function(BenchmarkId::new("entries", capacity), |bench| {
            bench.iter(|| {
                let _ = slb.query(black_box(&query), 5);
            });
        });
    }
    group.finish();
}

fn token_vector(dim: usize, cell_id: usize, token_id: usize) -> Vec<f32> {
    let mut state = (cell_id as u64 + 1).wrapping_mul(CELL_SEED_MULTIPLIER)
        ^ (token_id as u64 + 1).wrapping_mul(TOKEN_SEED_MULTIPLIER);
    let mut vector = Vec::with_capacity(dim);
    for _ in 0..dim {
        state = state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(1);
        vector.push(((state >> RANDOM_SHIFT) as f32 / RANDOM_DENOMINATOR) * 2.0 - 1.0);
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn encoded_cell(dim: usize, cell_id: usize, tokens_per_cell: usize) -> Vec<f32> {
    let tokens: Vec<Vec<f32>> =
        (0..tokens_per_cell).map(|token_id| token_vector(dim, cell_id, token_id)).collect();
    let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
    encode_per_token_keys(&refs)
}

fn recall_at(rankings: &[Vec<u64>], expected: &[u64], k: usize) -> f32 {
    let hits = rankings
        .iter()
        .zip(expected)
        .filter(|(ranking, expected_id)| ranking.iter().take(k).any(|id| id == *expected_id))
        .count();
    hits as f32 / expected.len() as f32
}

fn worst_top1_concentration(rankings: &[Vec<u64>]) -> usize {
    let mut counts = std::collections::HashMap::new();
    for ranking in rankings {
        if let Some(top1) = ranking.first() {
            *counts.entry(*top1).or_insert(0usize) += 1;
        }
    }
    counts.values().copied().max().unwrap_or(0)
}

struct CorrectnessReport {
    recall_at_1: f32,
    recall_at_5: f32,
    worst_top1: usize,
    candidate_count: usize,
}

fn per_token_correctness_report(
    retriever: &mut PerTokenRetriever,
    cell_count: usize,
    dim: usize,
    tokens_per_cell: usize,
) -> CorrectnessReport {
    // Template Method correctness flow:
    // choose deterministic targets → query → score recall and top-1 concentration.
    let query_count = FIXTURE_QUERY_COUNT.min(cell_count);
    let expected: Vec<u64> = (0..query_count)
        .map(|idx| ((idx * FIXTURE_QUERY_STRIDE + FIXTURE_QUERY_OFFSET) % cell_count) as u64)
        .collect();
    let rankings: Vec<Vec<u64>> = expected
        .iter()
        .map(|target| {
            let query = encoded_cell(dim, *target as usize, tokens_per_cell);
            retriever
                .query(&query, BENCH_TOP_K, Some(OWNER_ONE))
                .into_iter()
                .map(|r| r.cell_id)
                .collect()
        })
        .collect();

    CorrectnessReport {
        recall_at_1: recall_at(&rankings, &expected, 1),
        recall_at_5: recall_at(&rankings, &expected, BENCH_TOP_K),
        worst_top1: worst_top1_concentration(&rankings),
        candidate_count: candidate_count(cell_count, BENCH_TOP_K),
    }
}

fn candidate_count(cell_count: usize, k: usize) -> usize {
    if cell_count <= CANDIDATE_REDUCTION_THRESHOLD {
        cell_count
    } else {
        cell_count.min(MIN_CANDIDATES.max(k * CANDIDATE_MULTIPLIER))
    }
}

fn bench_per_token_top5avg_query(c: &mut Criterion) {
    // Template Method benchmark flow:
    // build deterministic encoded corpus → populate strategy → query → measure.
    let mut group = c.benchmark_group("PerTokenRetriever Top5Avg query — encoded per-token keys");
    for cell_count in BENCH_CELL_COUNTS {
        let mut retriever = PerTokenRetriever::with_scoring_mode(ScoringMode::Top5Avg);
        for cell_id in 0..cell_count {
            let encoded = encoded_cell(FIXTURE_DIM, cell_id, FIXTURE_TOKENS_PER_CELL);
            retriever.insert(cell_id as u64, OWNER_ONE, &encoded);
        }
        let report = per_token_correctness_report(
            &mut retriever,
            cell_count,
            FIXTURE_DIM,
            FIXTURE_TOKENS_PER_CELL,
        );
        let query = encoded_cell(FIXTURE_DIM, cell_count / 2, FIXTURE_TOKENS_PER_CELL);

        let bench_id = BenchmarkId::new(
            format!(
                "cells-r1-{:.0}-r5-{:.0}-gw-{}-cand-{}",
                report.recall_at_1 * 100.0,
                report.recall_at_5 * 100.0,
                report.worst_top1,
                report.candidate_count
            ),
            cell_count,
        );
        group.bench_function(bench_id, |bench| {
            bench.iter(|| {
                let _ = retriever.query(black_box(&query), BENCH_TOP_K, Some(OWNER_ONE));
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
