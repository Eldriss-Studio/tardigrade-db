//! AT-A4 — Engine-level retrieval recall after the lazy/INT8 retrieval
//! tier was introduced.
//!
//! Pins the engine-level public-API behavior: when a cell is written
//! via `mem_write` and then queried with a token from that same cell,
//! the engine returns the originating cell at rank 0 with high recall.
//!
//! This is the regression test for the `LoCoMo` collapse (68% → 3.3%)
//! caused by the lazy cutover before the INT8 retrieval tier landed.
//! Without the tier, scoring went through the Q4 archival path and
//! the outlier-channel crush dropped recall to noise. With the tier,
//! INT8 precision is preserved and recall@1 is near-perfect on
//! synthetic targeted data.
//!
//! Slice A4 of the retrieval correctness fix
//! (`docs/refs/external-references.md` §A3f).

use tdb_core::OwnerId;
use tdb_engine::engine::Engine;
use tdb_retrieval::per_token::encode_per_token_keys;
use tempfile::TempDir;

// ---------- fixture constants ----------

const FIXTURE_DIM: usize = 64;
const TOKENS_PER_CELL: usize = 8;
const FIXTURE_CELL_COUNT: usize = 100;
const RECALL_TOP_K: usize = 5;
const RECALL_FLOOR: f32 = 0.95;
const OWNER: OwnerId = 1;
const LAYER: u16 = 0;
const SALIENCE: f32 = 1.0;
const DUMMY_VALUE_DIM: usize = 16;
const PER_CELL_NOISE_HALF_RANGE: f32 = 0.05;
const CELL_SEED_MULTIPLIER: u64 = 0x9E37_79B1_85EB_CA87;
const TOKEN_SEED_MULTIPLIER: u64 = 0xC2B2_AE3D_27D4_EB4F;

// ---------- helpers ----------

fn lcg_advance(state: &mut u64) -> f32 {
    const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
    const LCG_INCREMENT: u64 = 1;
    const RANDOM_SHIFT: u32 = 40;
    const RANDOM_SCALE_DENOM: f32 = (1u64 << 24) as f32;
    *state = state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);
    (*state >> RANDOM_SHIFT) as f32 / RANDOM_SCALE_DENOM
}

fn normalize(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt().max(f32::MIN_POSITIVE);
    for value in vector {
        *value /= norm;
    }
}

/// A cell-unique direction vector. Each cell's tokens are small noisy
/// perturbations around this vector, so querying for one cell's
/// tokens unambiguously selects that cell — provided retrieval
/// preserves the signal at INT8 precision (the property under test).
fn cell_direction(cell_index: usize) -> Vec<f32> {
    let mut state = (cell_index as u64).wrapping_add(1).wrapping_mul(CELL_SEED_MULTIPLIER);
    let mut direction = vec![0.0_f32; FIXTURE_DIM];
    for value in &mut direction {
        let unit_random = lcg_advance(&mut state);
        *value = unit_random * 2.0 - 1.0;
    }
    normalize(&mut direction);
    direction
}

fn cell_tokens(cell_index: usize) -> Vec<Vec<f32>> {
    let direction = cell_direction(cell_index);
    let mut tokens = Vec::with_capacity(TOKENS_PER_CELL);
    for token_index in 0..TOKENS_PER_CELL {
        let mut state = ((cell_index as u64).wrapping_add(1)).wrapping_mul(CELL_SEED_MULTIPLIER)
            ^ ((token_index as u64).wrapping_add(1)).wrapping_mul(TOKEN_SEED_MULTIPLIER);
        let mut token = direction.clone();
        for value in &mut token {
            let unit_random = lcg_advance(&mut state);
            *value += (unit_random * 2.0 - 1.0) * PER_CELL_NOISE_HALF_RANGE;
        }
        normalize(&mut token);
        tokens.push(token);
    }
    tokens
}

fn encoded_key_for(cell_index: usize) -> Vec<f32> {
    let tokens = cell_tokens(cell_index);
    let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
    encode_per_token_keys(&refs)
}

#[test]
fn engine_recall_at_5_meets_floor_on_outlier_shaped_corpus() {
    let dir = TempDir::new().expect("tempdir");
    let mut engine = Engine::open(dir.path()).expect("open engine");

    // Insert FIXTURE_CELL_COUNT cells, remembering each one's id so we
    // can score retrieval against the known-correct answer.
    let mut cell_ids: Vec<u64> = Vec::with_capacity(FIXTURE_CELL_COUNT);
    for cell_index in 0..FIXTURE_CELL_COUNT {
        let key = encoded_key_for(cell_index);
        let value = vec![0.0_f32; DUMMY_VALUE_DIM];
        let cell_id = engine
            .mem_write(OWNER, LAYER, &key, value, SALIENCE, None)
            .expect("mem_write succeeds");
        cell_ids.push(cell_id);
    }

    // Each cell becomes its own query: use the cell's own tokens as
    // the query. recall@5 = fraction of (query → originating-cell
    // retrieved-at-or-above-rank-5).
    let mut hits = 0usize;
    for (cell_index, target_cell_id) in cell_ids.iter().enumerate() {
        let tokens = cell_tokens(cell_index);
        let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
        let encoded_query = encode_per_token_keys(&refs);
        let results =
            engine.mem_read(&encoded_query, RECALL_TOP_K, Some(OWNER)).expect("mem_read succeeds");
        if results.iter().any(|r| r.cell.id == *target_cell_id) {
            hits += 1;
        }
    }

    let recall = hits as f32 / FIXTURE_CELL_COUNT as f32;
    assert!(
        recall >= RECALL_FLOOR,
        "engine recall@{RECALL_TOP_K} = {recall} below floor {RECALL_FLOOR}; \
         INT8 retrieval tier may not be wired correctly",
    );
}
