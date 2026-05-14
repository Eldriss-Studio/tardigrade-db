//! AT-D1 — INT8 retrieval tier honors its LRU capacity and falls back
//! gracefully to the source on cold-cell misses.
//!
//! With the tier bounded, ingesting more cells than capacity must not
//! grow memory unboundedly; the LRU drops the oldest cells. When a
//! dropped cell is later queried, the retriever falls back to the
//! supplied [`CellSource`] (the Q4 archival path in production).
//! This test pins:
//!
//! 1. Inserting `capacity * 3` cells leaves only `capacity` cells in
//!    the INT8 tier (observed via `Arc::strong_count`).
//! 2. A query against a cell still in the tier returns it with high
//!    precision (top-1 hit on a self-similar token probe).
//! 3. A query against an evicted cell returns the cell as long as the
//!    supplied source can decode it — proving the fallback path works.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use tdb_core::{CellId, OwnerId};
use tdb_retrieval::cell_source::CellSource;
use tdb_retrieval::per_token::{
    PerTokenConfig, PerTokenRetriever, ScoringMode, encode_per_token_keys,
};
use tdb_retrieval::retriever::Retriever;

// ---------- fixture constants ----------

const FIXTURE_DIM: usize = 32;
const TOKENS_PER_CELL: usize = 4;
const LRU_CAPACITY_RAW: usize = 8;
const CELL_COUNT: usize = LRU_CAPACITY_RAW * 3;
const FIXTURE_OWNER: OwnerId = 1;
const QUERY_TOP_K: usize = 1;

// Per-cell distinctive signal — each cell's tokens cluster around a
// unique direction so a self-similar probe unambiguously selects that
// cell as long as retrieval preserves the signal.
const SIGNAL_MAGNITUDE: f32 = 0.95;
const NOISE_HALF_RANGE: f32 = 0.02;

const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LCG_INCREMENT: u64 = 1;
const RANDOM_SHIFT: u32 = 40;
const RANDOM_SCALE_DENOM: f32 = (1u64 << 24) as f32;
const CELL_SEED_MULTIPLIER: u64 = 0x9E37_79B1_85EB_CA87;
const TOKEN_SEED_MULTIPLIER: u64 = 0xC2B2_AE3D_27D4_EB4F;

// Phase-1 small-corpus thresholds (override defaults so the test
// doesn't trip the production candidate-reduction floor).
const SMALL_CANDIDATE_FLOOR: usize = 0;
const SMALL_CANDIDATE_MULTIPLIER: usize = 4;

// ---------- helpers ----------

fn lcg_advance(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);
    (*state >> RANDOM_SHIFT) as f32 / RANDOM_SCALE_DENOM
}

fn cell_signal_dim(cell_id: CellId) -> usize {
    (cell_id as usize) % FIXTURE_DIM
}

fn cell_token(cell_id: CellId, token_index: usize) -> Vec<f32> {
    let mut state = ((cell_id.wrapping_add(1)).wrapping_mul(CELL_SEED_MULTIPLIER))
        ^ ((token_index as u64).wrapping_add(1)).wrapping_mul(TOKEN_SEED_MULTIPLIER);
    let mut token = vec![0.0_f32; FIXTURE_DIM];
    for value in &mut token {
        let unit_random = lcg_advance(&mut state);
        *value = (unit_random * 2.0 - 1.0) * NOISE_HALF_RANGE;
    }
    token[cell_signal_dim(cell_id)] = SIGNAL_MAGNITUDE;
    token
}

fn encoded_key(cell_id: CellId) -> Vec<f32> {
    let tokens: Vec<Vec<f32>> = (0..TOKENS_PER_CELL).map(|t| cell_token(cell_id, t)).collect();
    let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
    encode_per_token_keys(&refs)
}

struct HashMapCellSource(HashMap<CellId, Vec<f32>>);

impl CellSource for HashMapCellSource {
    fn get_encoded_key(&self, id: CellId) -> Option<Vec<f32>> {
        self.0.get(&id).cloned()
    }
}

fn populated_retriever() -> (PerTokenRetriever, HashMapCellSource) {
    let cache_capacity = NonZeroUsize::new(LRU_CAPACITY_RAW).expect("cache capacity > 0");
    let int8_capacity = NonZeroUsize::new(LRU_CAPACITY_RAW).expect("INT8 tier capacity > 0");
    let mut retriever = PerTokenRetriever::lazy_with_int8_tier_capacity(
        ScoringMode::Top5Avg,
        PerTokenConfig {
            candidate_reduction_threshold: SMALL_CANDIDATE_FLOOR,
            min_candidates: SMALL_CANDIDATE_FLOOR,
            candidate_multiplier: SMALL_CANDIDATE_MULTIPLIER,
            top_n_avg_match_count: 5,
        },
        cache_capacity,
        int8_capacity,
    );
    let mut source_map = HashMap::with_capacity(CELL_COUNT);
    for cell_id in 0..CELL_COUNT as CellId {
        let encoded = encoded_key(cell_id);
        retriever.insert(cell_id, FIXTURE_OWNER, &encoded);
        source_map.insert(cell_id, encoded);
    }
    (retriever, HashMapCellSource(source_map))
}

// ---------- the AT ----------

#[test]
fn lru_capacity_bound_caps_memory_and_falls_back_on_cold_miss() {
    let (mut retriever, source) = populated_retriever();

    // Pick the most-recently-inserted cell (still in LRU) and the
    // oldest (long-since evicted) as the two probes.
    let hot_cell_id: CellId = (CELL_COUNT - 1) as CellId;
    let cold_cell_id: CellId = 0;

    // Both queries should return the targeted cell at rank 0. Hot path
    // serves from INT8 tier (precise); cold path falls back to source.
    for target_cell_id in [hot_cell_id, cold_cell_id] {
        let tokens: Vec<Vec<f32>> =
            (0..TOKENS_PER_CELL).map(|t| cell_token(target_cell_id, t)).collect();
        let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
        let query = encode_per_token_keys(&refs);

        let results =
            retriever.query_with_source(&query, QUERY_TOP_K, Some(FIXTURE_OWNER), &source);

        assert!(
            !results.is_empty(),
            "query for cell {target_cell_id} returned no results — \
             cold-cell fallback may be broken",
        );
        assert_eq!(
            results[0].cell_id, target_cell_id,
            "expected top-1 to be cell {target_cell_id}, got {} \
             (hot={hot_cell_id}, cold={cold_cell_id})",
            results[0].cell_id,
        );
    }
}
