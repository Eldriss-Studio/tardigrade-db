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
    let mut state = ((cell_id as u64 + 1) * 0x9E37_79B1_85EB_CA87)
        ^ ((token_id as u64 + 1) * 0xC2B2_AE3D_27D4_EB4F);
    let mut vector = Vec::with_capacity(dim);
    for _ in 0..dim {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        vector.push(((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0);
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

struct EngineFixture {
    _dir: tempfile::TempDir,
    engine: Engine,
}

fn build_encoded_engine(
    cell_count: usize,
    dim: usize,
    tokens_per_cell: usize,
    vamana_threshold: usize,
) -> EngineFixture {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open_with_vamana_threshold(dir.path(), vamana_threshold).unwrap();
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
    EngineFixture { _dir: dir, engine }
}

fn collect_engine_rankings(
    engine: &mut Engine,
    cell_count: usize,
    dim: usize,
    tokens_per_cell: usize,
) -> (Vec<u64>, Vec<Vec<u64>>) {
    let query_count = 30.min(cell_count);
    let expected: Vec<u64> =
        (0..query_count).map(|idx| ((idx * 7 + 3) % cell_count) as u64).collect();
    let rankings: Vec<Vec<u64>> = expected
        .iter()
        .map(|target| {
            let query = encoded_cell(dim, *target as usize, tokens_per_cell);
            engine.mem_read(&query, 5, Some(1)).unwrap().into_iter().map(|r| r.cell.id).collect()
        })
        .collect();
    (expected, rankings)
}

struct CorrectnessReport {
    recall_at_1: f32,
    recall_at_5: f32,
    worst_top1: usize,
    vamana_changed: bool,
}

fn engine_correctness_report(
    cell_count: usize,
    dim: usize,
    tokens_per_cell: usize,
) -> CorrectnessReport {
    // Template Method correctness flow:
    // build no-Vamana baseline → build Vamana-active engine → compare rankings.
    let mut baseline = build_encoded_engine(cell_count, dim, tokens_per_cell, usize::MAX);
    let (expected, baseline_rankings) =
        collect_engine_rankings(&mut baseline.engine, cell_count, dim, tokens_per_cell);

    let mut with_vamana = build_encoded_engine(cell_count, dim, tokens_per_cell, 1);
    let (_expected, vamana_rankings) =
        collect_engine_rankings(&mut with_vamana.engine, cell_count, dim, tokens_per_cell);

    CorrectnessReport {
        recall_at_1: recall_at(&baseline_rankings, &expected, 1),
        recall_at_5: recall_at(&baseline_rankings, &expected, 5),
        worst_top1: worst_top1_concentration(&baseline_rankings),
        vamana_changed: baseline_rankings != vamana_rankings,
    }
}

fn bench_engine_read_encoded_per_token(c: &mut Criterion) {
    // Template Method benchmark flow:
    // build deterministic encoded corpus → batch populate engine → query → measure.
    let mut group = c.benchmark_group("Engine mem_read — encoded per-token Top5Avg path");
    let dim = 128;
    let tokens_per_cell = 8;

    for cell_count in [100usize, 1_000, 10_000] {
        let report = engine_correctness_report(cell_count, dim, tokens_per_cell);
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

        let bench_id = BenchmarkId::new(
            format!(
                "cells-r1-{:.0}-r5-{:.0}-gw-{}-vamana-changed-{}",
                report.recall_at_1 * 100.0,
                report.recall_at_5 * 100.0,
                report.worst_top1,
                report.vamana_changed
            ),
            cell_count,
        );
        group.bench_function(bench_id, |bench| {
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
