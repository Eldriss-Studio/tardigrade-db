//! Acceptance test: Vamana activation survives realistic-scale ingest.
//!
//! Regression for the panic discovered while running the LoCoMo benchmark on
//! CUDA: crossing the Vamana activation threshold (10,000 cells) panicked the
//! engine because stored per-token encoded keys could not be re-pooled into
//! fixed-dim vectors. Root cause was Q4 quantization corrupting the
//! `n_tokens` slot of the encoded header; the fix made
//! `decode_per_token_keys` infer `n` from `data.len() / d` rather than
//! trusting the header.
//!
//! This test exercises the full ingest → threshold-cross → Vamana-build →
//! query pipeline at realistic dimensions (Qwen3-0.6B-shaped 1024-dim
//! keys, 127 tokens per cell). Without the decoder fix it panics at write
//! #10,000.

use tdb_core::OwnerId;
use tdb_engine::engine::Engine;
use tdb_retrieval::per_token::encode_per_token_keys;
use tempfile::TempDir;

const DIM: usize = 1024;
const TOKENS_PER_CELL: usize = 127;
const VAMANA_THRESHOLD: usize = 10_000;
/// Cross the threshold by a comfortable margin so we exercise both activation
/// (at cell #10,000) and post-activation writes.
const CELL_COUNT: usize = VAMANA_THRESHOLD + 5;
const OWNER: OwnerId = 1;
const LAYER: u16 = 14;

/// LCG-derived random per-token vector. Mirrors the pattern used in
/// `per_token` unit tests: enough cell-to-cell variation that mean-pooled
/// vectors are discriminative for ANN retrieval.
fn token_vector(cell_id: u64, token_idx: usize) -> Vec<f32> {
    const CELL_SEED: u64 = 0x9E37_79B1_85EB_CA87;
    const TOKEN_SEED: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const LCG: u64 = 6_364_136_223_846_793_005;
    let mut state =
        (cell_id + 1).wrapping_mul(CELL_SEED) ^ (token_idx as u64 + 1).wrapping_mul(TOKEN_SEED);
    let mut v = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        state = state.wrapping_mul(LCG).wrapping_add(1);
        let raw = (state >> 40) as f32 / ((1u64 << 24) as f32);
        v.push(raw * 2.0 - 1.0);
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn encoded_key_for(cell_id: u64) -> Vec<f32> {
    let tokens: Vec<Vec<f32>> = (0..TOKENS_PER_CELL).map(|t| token_vector(cell_id, t)).collect();
    let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
    encode_per_token_keys(&refs)
}

#[test]
#[ignore = "bench-class: 10k-cell ingest; ~3 min in --release, ~20+ min in debug. \
            Run via `cargo test -- --ignored`."]
fn vamana_activation_survives_threshold_crossing() {
    let dir = TempDir::new().expect("tempdir");
    let mut engine = Engine::open(dir.path()).expect("open engine");

    // Sanity: Vamana inactive before we cross the threshold.
    assert!(!engine.has_vamana(), "Vamana should not be active before threshold");

    // Pick a target cell well before the threshold so its key is stored,
    // pooled into Vamana on activation, and queryable afterwards.
    let target_cell_id_seed: u64 = 4242;
    let mut target_cell_id: Option<u64> = None;
    let mut activation_observed_at: Option<usize> = None;

    for i in 0..CELL_COUNT {
        let seed = i as u64;
        let key = encoded_key_for(seed);
        let value = vec![0.0f32; 16]; // payload not exercised here

        let cell_id = engine
            .mem_write(OWNER, LAYER, &key, value, 1.0, None)
            .expect("mem_write must not panic");

        if seed == target_cell_id_seed {
            target_cell_id = Some(cell_id);
        }
        if engine.has_vamana() && activation_observed_at.is_none() {
            activation_observed_at = Some(i + 1);
        }
    }

    let activation_at =
        activation_observed_at.expect("Vamana must activate when threshold crossed");
    assert!(
        activation_at >= VAMANA_THRESHOLD,
        "Vamana activated before threshold ({activation_at} < {VAMANA_THRESHOLD})",
    );
    assert!(engine.has_vamana(), "Vamana stays active after build");

    // Retrieval probe: query with the original tokens of `target_cell_id` and
    // assert that cell appears in the top-K results. This proves that the
    // pooled vectors in the Vamana index aren't garbage — without the decoder
    // fix the index would either fail to build or be filled with empty
    // vectors, and the target cell would not surface.
    let target_id = target_cell_id.expect("target cell was written");
    let query_tokens: Vec<f32> =
        (0..TOKENS_PER_CELL).flat_map(|t| token_vector(target_cell_id_seed, t)).collect();

    let results = engine
        .mem_read_tokens(&query_tokens, TOKENS_PER_CELL, DIM, 20, Some(OWNER))
        .expect("mem_read_tokens must succeed after Vamana activation");

    assert!(
        results.iter().any(|r| r.cell.id == target_id),
        "target cell {target_id} not in top-20 after Vamana build; \
         pipeline may be returning empty / garbage results",
    );
}
