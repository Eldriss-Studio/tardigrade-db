use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::engine::{Engine, WriteRequest};
use tdb_retrieval::per_token::encode_per_token_keys;

const FIXTURE_DIM: usize = 128;
const FIXTURE_TOKENS_PER_CELL: usize = 8;
const LEGACY_BENCH_DIM: usize = 64;
const LEGACY_READ_CELL_COUNT: u64 = 1_000;
const LEGACY_TARGET_CELL: usize = 10;
const LEGACY_BASE_VALUE: f32 = 0.01;
const FIXTURE_QUERY_COUNT: usize = 30;
const FIXTURE_QUERY_STRIDE: usize = 7;
const FIXTURE_QUERY_OFFSET: usize = 3;
const BENCH_TOP_K: usize = 5;
const OWNER_ONE: u64 = 1;
const DEFAULT_LAYER: u16 = 0;
const DEFAULT_SALIENCE: f32 = 50.0;
const PACK_SALIENCE: f32 = 80.0;
const BENCH_CELL_COUNTS: [usize; 3] = [100, 1_000, 10_000];
const PACK_LAYER_COUNTS: [usize; 2] = [1, 4];
const PACK_TARGET_DIVISOR: usize = 2;
const INDEXED_CELLS_PER_PACK: usize = 1;
const CANDIDATE_REDUCTION_THRESHOLD: usize = 512;
const MIN_CANDIDATES: usize = 256;
const CANDIDATE_MULTIPLIER: usize = 64;
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const CELL_SEED_MULTIPLIER: u64 = 0x9E37_79B1_85EB_CA87;
const TOKEN_SEED_MULTIPLIER: u64 = 0xC2B2_AE3D_27D4_EB4F;
const RANDOM_SHIFT: u32 = 40;
const RANDOM_DENOMINATOR: f32 = (1u64 << 24) as f32;

fn bench_engine_write(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let key: Vec<f32> =
        (0..LEGACY_BENCH_DIM).map(|i| (i as f32 * LEGACY_BASE_VALUE).sin()).collect();
    let value: Vec<f32> = vec![0.0; LEGACY_BENCH_DIM];

    c.bench_function("Engine mem_write — single cell persist with fsync (dim=64)", |bench| {
        bench.iter(|| {
            engine
                .mem_write(
                    OWNER_ONE,
                    DEFAULT_LAYER,
                    black_box(&key),
                    value.clone(),
                    DEFAULT_SALIENCE,
                    None,
                )
                .unwrap();
        });
    });
}

fn bench_engine_read(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Pre-populate with 1000 cells.
    for i in 0..LEGACY_READ_CELL_COUNT {
        let mut key = vec![LEGACY_BASE_VALUE; LEGACY_BENCH_DIM];
        key[(i as usize) % LEGACY_BENCH_DIM] = 1.0;
        engine
            .mem_write(
                OWNER_ONE,
                DEFAULT_LAYER,
                &key,
                vec![0.0; LEGACY_BENCH_DIM],
                DEFAULT_SALIENCE,
                None,
            )
            .unwrap();
    }

    let mut query = vec![LEGACY_BASE_VALUE; LEGACY_BENCH_DIM];
    query[LEGACY_TARGET_CELL] = 1.0;

    c.bench_function(
        "Engine mem_read — full pipeline: SLB → retriever → governance (1K cells, dim=64, top-5)",
        |bench| {
            bench.iter(|| {
                let _ = engine.mem_read(black_box(&query), BENCH_TOP_K, None).unwrap();
            });
        },
    );
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
            owner: OWNER_ONE,
            layer: DEFAULT_LAYER,
            key: encoded_cell(dim, cell_id, tokens_per_cell),
            value: vec![0.0; dim],
            salience: DEFAULT_SALIENCE,
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
    let query_count = FIXTURE_QUERY_COUNT.min(cell_count);
    let expected: Vec<u64> = (0..query_count)
        .map(|idx| ((idx * FIXTURE_QUERY_STRIDE + FIXTURE_QUERY_OFFSET) % cell_count) as u64)
        .collect();
    let rankings: Vec<Vec<u64>> = expected
        .iter()
        .map(|target| {
            let query = encoded_cell(dim, *target as usize, tokens_per_cell);
            engine
                .mem_read(&query, BENCH_TOP_K, Some(OWNER_ONE))
                .unwrap()
                .into_iter()
                .map(|r| r.cell.id)
                .collect()
        })
        .collect();
    (expected, rankings)
}

struct CorrectnessReport {
    recall_at_1: f32,
    recall_at_5: f32,
    worst_top1: usize,
    vamana_changed: bool,
    candidate_count: usize,
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
        recall_at_5: recall_at(&baseline_rankings, &expected, BENCH_TOP_K),
        worst_top1: worst_top1_concentration(&baseline_rankings),
        vamana_changed: baseline_rankings != vamana_rankings,
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

fn bench_engine_read_encoded_per_token(c: &mut Criterion) {
    // Template Method benchmark flow:
    // build deterministic encoded corpus → batch populate engine → query → measure.
    let mut group = c.benchmark_group("Engine mem_read — encoded per-token Top5Avg path");
    for cell_count in BENCH_CELL_COUNTS {
        let report = engine_correctness_report(cell_count, FIXTURE_DIM, FIXTURE_TOKENS_PER_CELL);
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        let requests: Vec<WriteRequest> = (0..cell_count)
            .map(|cell_id| WriteRequest {
                owner: OWNER_ONE,
                layer: DEFAULT_LAYER,
                key: encoded_cell(FIXTURE_DIM, cell_id, FIXTURE_TOKENS_PER_CELL),
                value: vec![0.0; FIXTURE_DIM],
                salience: DEFAULT_SALIENCE,
                parent_cell_id: None,
            })
            .collect();
        engine.mem_write_batch(&requests).unwrap();
        let query = encoded_cell(FIXTURE_DIM, cell_count / 2, FIXTURE_TOKENS_PER_CELL);

        let bench_id = BenchmarkId::new(
            format!(
                "cells-r1-{:.0}-r5-{:.0}-gw-{}-cand-{}-vamana-changed-{}",
                report.recall_at_1 * 100.0,
                report.recall_at_5 * 100.0,
                report.worst_top1,
                report.candidate_count,
                report.vamana_changed
            ),
            cell_count,
        );
        group.bench_function(bench_id, |bench| {
            bench.iter(|| {
                let _ = engine.mem_read(black_box(&query), BENCH_TOP_K, Some(OWNER_ONE)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_engine_read_pack_encoded_per_token(c: &mut Criterion) {
    // Fixture Builder / Object Mother: each pack has one small layer payload;
    // retrieval key shape matches the validated hidden-state Top5Avg path.
    let mut group = c.benchmark_group("Engine mem_read_pack — encoded per-token Top5Avg path");
    for pack_count in BENCH_CELL_COUNTS {
        for layer_count in PACK_LAYER_COUNTS {
            let dir = tempfile::tempdir().unwrap();
            let mut engine = Engine::open(dir.path()).unwrap();
            for pack_id in 0..pack_count {
                engine.mem_write_pack(&bench_pack(pack_id, layer_count)).unwrap();
            }
            let target_pack = target_pack_id(pack_count);
            let query = encoded_cell(FIXTURE_DIM, target_pack, FIXTURE_TOKENS_PER_CELL);
            let report = pack_read_correctness_report(&mut engine, &query, target_pack);

            let bench_id = BenchmarkId::new(
                format!(
                    "target-{}-dedup-{}-layers-{}-indexed-{}",
                    report.target_top1, report.dedup_ok, layer_count, INDEXED_CELLS_PER_PACK
                ),
                pack_count,
            );
            group.bench_function(bench_id, |bench| {
                bench.iter(|| {
                    let _ = engine
                        .mem_read_pack(black_box(&query), BENCH_TOP_K, Some(OWNER_ONE))
                        .unwrap();
                });
            });
        }
    }
    group.finish();
}

struct PackReadCorrectnessReport {
    target_top1: bool,
    dedup_ok: bool,
}

fn bench_pack(pack_id: usize, layer_count: usize) -> KVPack {
    KVPack {
        id: 0,
        owner: OWNER_ONE,
        retrieval_key: encoded_cell(FIXTURE_DIM, pack_id, FIXTURE_TOKENS_PER_CELL),
        layers: (0..layer_count)
            .map(|layer_idx| KVLayerPayload {
                layer_idx: DEFAULT_LAYER + layer_idx as u16,
                data: vec![pack_id as f32; FIXTURE_DIM],
            })
            .collect(),
        salience: PACK_SALIENCE,
    }
}

fn target_pack_id(pack_count: usize) -> usize {
    pack_count / PACK_TARGET_DIVISOR
}

fn pack_read_correctness_report(
    engine: &mut Engine,
    query: &[f32],
    target_pack: usize,
) -> PackReadCorrectnessReport {
    let results = engine.mem_read_pack(query, BENCH_TOP_K, Some(OWNER_ONE)).unwrap();
    let pack_ids: Vec<usize> =
        results.iter().map(|result| result.pack.layers[0].data[0].round() as usize).collect();
    let unique_pack_ids: std::collections::HashSet<usize> = pack_ids.iter().copied().collect();

    PackReadCorrectnessReport {
        target_top1: pack_ids.first().is_some_and(|pack_id| *pack_id == target_pack),
        dedup_ok: unique_pack_ids.len() == pack_ids.len(),
    }
}

criterion_group!(
    benches,
    bench_engine_write,
    bench_engine_read,
    bench_engine_read_encoded_per_token,
    bench_engine_read_pack_encoded_per_token
);
criterion_main!(benches);
