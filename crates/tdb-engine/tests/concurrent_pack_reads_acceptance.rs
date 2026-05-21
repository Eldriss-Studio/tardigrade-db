// Test fixtures use small bounded values (cell counts < 10K, dims < 1024) —
// these casts cannot truncate at the scales exercised here.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

//! Concurrent pack-read acceptance.
//!
//! `mem_read_pack` takes `&self` and is shareable across threads through
//! an outer `Arc`. The per-cell governance `Mutex` serializes the
//! tier-state bookkeeping, but multiple readers querying different cells
//! contend only on bucket-level `HashMap` reads, which are lock-free
//! through a shared reference.
//!
//! Property under test: under concurrent reads of the same query, every
//! thread observes the same retrieved cell set. Scores may drift across
//! threads because tier transitions race on hot cells — set-membership
//! is the meaningful invariant, not exact ordering.

use std::sync::Arc;
use std::thread;

use tdb_core::kv_pack::{KVLayerPayload, KVPack, PackReadResult};
use tdb_engine::engine::Engine;
use tdb_retrieval::per_token::encode_per_token_keys;

const CORPUS_OWNERS: &[u64] = &[1, 2, 3];
const PACKS_PER_OWNER: usize = 32;
const KEY_DIM: usize = 64;
const VALUE_DIM: usize = 128;
const READER_THREADS: usize = 8;
const QUERIES_PER_READER: usize = 25;
const TOP_K: usize = 5;
const QUERY_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const CORPUS_SEED: u64 = 0xC2B2_AE3D_27D4_EB4F;

fn deterministic_key(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn seed_engine(engine: &mut Engine) {
    let mut cell_seed = CORPUS_SEED;
    for &owner in CORPUS_OWNERS {
        for _ in 0..PACKS_PER_OWNER {
            let token = deterministic_key(cell_seed, KEY_DIM);
            let retrieval_key = encode_per_token_keys(&[&token]);
            let value = deterministic_key(cell_seed.wrapping_add(1), VALUE_DIM);
            let pack = KVPack {
                id: 0,
                owner,
                retrieval_key,
                layers: vec![KVLayerPayload { layer_idx: 0, data: value }],
                salience: 50.0,
                text: None,
            };
            engine.mem_write_pack(&pack).unwrap();
            cell_seed = cell_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        }
    }
}

fn pack_id_set(results: &[PackReadResult]) -> Vec<u64> {
    let mut ids: Vec<u64> = results.iter().map(|r| r.pack.id).collect();
    ids.sort_unstable();
    ids
}

#[test]
fn concurrent_pack_reads_return_same_cell_set_across_threads() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    seed_engine(&mut engine);

    let query_token = deterministic_key(QUERY_SEED, KEY_DIM);
    let query = encode_per_token_keys(&[&query_token]);

    // Warm up tier transitions so the steady-state ranking is stable.
    // Otherwise the first few calls promote cells from Draft to Validated
    // and the boost flips the ordering mid-run.
    for _ in 0..50 {
        let _ = engine.mem_read_pack(&query, TOP_K, None).unwrap();
    }
    let baseline = pack_id_set(&engine.mem_read_pack(&query, TOP_K, None).unwrap());

    let engine = Arc::new(engine);
    let mut handles = Vec::with_capacity(READER_THREADS);
    for _ in 0..READER_THREADS {
        let engine = Arc::clone(&engine);
        let query = query.clone();
        handles.push(thread::spawn(move || {
            let mut sets = Vec::with_capacity(QUERIES_PER_READER);
            for _ in 0..QUERIES_PER_READER {
                let results = engine.mem_read_pack(&query, TOP_K, None).unwrap();
                sets.push(pack_id_set(&results));
            }
            sets
        }));
    }

    let per_thread: Vec<Vec<Vec<u64>>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    for (thread_idx, thread_sets) in per_thread.iter().enumerate() {
        for (q_idx, ids) in thread_sets.iter().enumerate() {
            assert_eq!(
                ids, &baseline,
                "thread {thread_idx} query {q_idx} diverged: got {ids:?}, baseline {baseline:?}",
            );
        }
    }
}
