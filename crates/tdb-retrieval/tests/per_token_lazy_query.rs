//! AT-2 — `PerTokenRetriever::query_with_source` decodes only candidates
//! and produces top-K equivalent to the eager `query` path.
//!
//! Slice 2 of the universal scalability fix. The test is at the
//! retriever's public surface (Kent Dodds): it exercises observable
//! behavior — top-K equivalence and bounded source consultations —
//! without poking internal state.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use tdb_core::{CellId, OwnerId};
use tdb_retrieval::cell_source::CellSource;
use tdb_retrieval::per_token::{
    PerTokenConfig, PerTokenRetriever, ScoringMode, encode_per_token_keys,
};
use tdb_retrieval::retriever::Retriever;

// ---------- fixture constants ----------

const FIXTURE_DIM: usize = 32;
const TOKENS_PER_CELL: usize = 8;
const CELL_COUNT: usize = 1_000;
const QUERY_TOP_K: usize = 5;
const QUERY_CELL_ID: CellId = 503;
const FIXTURE_OWNER: OwnerId = 1;
const MIN_TOP_K_OVERLAP: usize = 4;

const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const CELL_SEED: u64 = 0x9E37_79B1_85EB_CA87;
const TOKEN_SEED: u64 = 0xC2B2_AE3D_27D4_EB4F;
const LCG_INCREMENT: u64 = 1;
const RANDOM_SHIFT: u32 = 40;
const RANDOM_SCALE_DENOMINATOR: f32 = (1u64 << 24) as f32;
const RANDOM_MIDPOINT_OFFSET: f32 = 1.0;
const RANDOM_RANGE_FACTOR: f32 = 2.0;

const SMALL_CANDIDATE_FLOOR: usize = 32;
const SMALL_CANDIDATE_MULTIPLIER: usize = 4;

// ---------- test doubles ----------

/// Records every call so the test can assert lazy decoding.
struct CountingCellSource {
    inner: HashMap<CellId, Vec<f32>>,
    call_count: AtomicUsize,
}

impl CountingCellSource {
    fn new(inner: HashMap<CellId, Vec<f32>>) -> Self {
        Self { inner, call_count: AtomicUsize::new(0) }
    }

    fn call_count(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }
}

impl CellSource for CountingCellSource {
    fn get_encoded_key(&self, id: CellId) -> Option<Vec<f32>> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        self.inner.get(&id).cloned()
    }
}

// ---------- helpers ----------

fn normalized_token(cell_id: CellId, token_index: usize) -> Vec<f32> {
    let mut state = (cell_id.wrapping_add(LCG_INCREMENT)).wrapping_mul(CELL_SEED)
        ^ ((token_index as u64).wrapping_add(LCG_INCREMENT)).wrapping_mul(TOKEN_SEED);
    let mut vector = Vec::with_capacity(FIXTURE_DIM);
    for _ in 0..FIXTURE_DIM {
        state = state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);
        let raw = (state >> RANDOM_SHIFT) as f32 / RANDOM_SCALE_DENOMINATOR;
        vector.push(raw * RANDOM_RANGE_FACTOR - RANDOM_MIDPOINT_OFFSET);
    }
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn encoded_cell_key(cell_id: CellId) -> Vec<f32> {
    let tokens: Vec<Vec<f32>> =
        (0..TOKENS_PER_CELL).map(|i| normalized_token(cell_id, i)).collect();
    let refs: Vec<&[f32]> = tokens.iter().map(Vec::as_slice).collect();
    encode_per_token_keys(&refs)
}

/// Build a retriever populated with `CELL_COUNT` cells using small
/// candidate-reduction config so the test exercises Phase-1 narrowing.
/// Returns the retriever and a parallel id→encoded-key map for the source.
fn populated_retriever_with_source_map() -> (PerTokenRetriever, HashMap<CellId, Vec<f32>>) {
    let mut retriever = PerTokenRetriever::with_config(
        ScoringMode::Top5Avg,
        PerTokenConfig {
            candidate_reduction_threshold: SMALL_CANDIDATE_FLOOR,
            min_candidates: SMALL_CANDIDATE_FLOOR,
            candidate_multiplier: SMALL_CANDIDATE_MULTIPLIER,
            top_n_avg_match_count: 5,
        },
    );
    let mut source_map = HashMap::with_capacity(CELL_COUNT);
    for cell_id in 0..CELL_COUNT as CellId {
        let encoded = encoded_cell_key(cell_id);
        retriever.insert(cell_id, FIXTURE_OWNER, &encoded);
        source_map.insert(cell_id, encoded);
    }
    (retriever, source_map)
}

fn top_k_ids(results: &[tdb_retrieval::attention::RetrievalResult]) -> Vec<CellId> {
    results.iter().map(|r| r.cell_id).collect()
}

fn overlap(a: &[CellId], b: &[CellId]) -> usize {
    a.iter().filter(|id| b.contains(id)).count()
}

// ---------- the acceptance test ----------

#[test]
fn lazy_query_matches_eager_top_k_and_decodes_only_candidates() {
    let (mut retriever, source_map) = populated_retriever_with_source_map();
    let query = encoded_cell_key(QUERY_CELL_ID);
    let source = CountingCellSource::new(source_map);

    // Eager reference: the existing in-memory path.
    let eager_results = retriever.query(&query, QUERY_TOP_K, Some(FIXTURE_OWNER));

    // Test subject: the lazy path under test.
    let lazy_results =
        retriever.query_with_source(&query, QUERY_TOP_K, Some(FIXTURE_OWNER), &source);

    let eager_ids = top_k_ids(&eager_results);
    let lazy_ids = top_k_ids(&lazy_results);

    assert_eq!(lazy_ids.len(), QUERY_TOP_K, "lazy path must return exactly k results");
    assert!(
        overlap(&eager_ids, &lazy_ids) >= MIN_TOP_K_OVERLAP,
        "lazy top-K must intersect eager top-K by at least {MIN_TOP_K_OVERLAP}; \
         eager={eager_ids:?} lazy={lazy_ids:?}",
    );

    // Behavioral assertion: the source was consulted for the candidate set
    // only, not the full corpus. Phase-1 produces at most candidate_limit(k)
    // cells; we accept that as the upper bound on source calls.
    let candidate_ceiling = SMALL_CANDIDATE_FLOOR.max(QUERY_TOP_K * SMALL_CANDIDATE_MULTIPLIER);
    assert!(
        source.call_count() <= candidate_ceiling,
        "lazy path decoded {} cells but should have decoded at most {} \
         (candidate_limit for k={})",
        source.call_count(),
        candidate_ceiling,
        QUERY_TOP_K,
    );
    assert!(
        source.call_count() < CELL_COUNT,
        "lazy path must NOT decode every cell (decoded {} of {})",
        source.call_count(),
        CELL_COUNT,
    );
}

// ---------- AT-3: bounded LRU cache reuses decoded tokens ----------

const WARM_CACHE_CAPACITY_RAW: usize = 1_024;

#[test]
fn warm_cache_serves_repeated_query_without_reloading_source() {
    use std::num::NonZeroUsize;

    let warm_cache_capacity =
        NonZeroUsize::new(WARM_CACHE_CAPACITY_RAW).expect("test cache capacity must be non-zero");

    // Build a retriever with a generous cache (≥ candidate set) so the
    // second query of the same input can serve every candidate from cache.
    let mut retriever = PerTokenRetriever::with_config_and_cache_capacity(
        ScoringMode::Top5Avg,
        PerTokenConfig {
            candidate_reduction_threshold: SMALL_CANDIDATE_FLOOR,
            min_candidates: SMALL_CANDIDATE_FLOOR,
            candidate_multiplier: SMALL_CANDIDATE_MULTIPLIER,
            top_n_avg_match_count: 5,
        },
        warm_cache_capacity,
    );
    let mut source_map = HashMap::with_capacity(CELL_COUNT);
    for cell_id in 0..CELL_COUNT as CellId {
        let encoded = encoded_cell_key(cell_id);
        retriever.insert(cell_id, FIXTURE_OWNER, &encoded);
        source_map.insert(cell_id, encoded);
    }
    let query = encoded_cell_key(QUERY_CELL_ID);
    let source = CountingCellSource::new(source_map);

    let _first = retriever.query_with_source(&query, QUERY_TOP_K, Some(FIXTURE_OWNER), &source);
    let calls_after_first = source.call_count();
    assert!(calls_after_first > 0, "first query must consult the source at least once");

    let _second = retriever.query_with_source(&query, QUERY_TOP_K, Some(FIXTURE_OWNER), &source);
    let calls_after_second = source.call_count();
    assert_eq!(
        calls_after_second, calls_after_first,
        "repeated identical query must serve all candidates from the warm cache; \
         expected source.call_count() to stay at {calls_after_first}, observed {calls_after_second}",
    );
}
