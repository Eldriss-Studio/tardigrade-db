//! ATDD for `Engine::sweep_now` — M3.2 from the foundation plan.
//!
//! The background [`tdb_engine::maintenance::MaintenanceWorker`]
//! advances time and evicts Draft packs on a 1-hour cadence. That's
//! fine for steady-state retention but consumers that want
//! immediate tier transitions after a known event (episode boundary,
//! save-game checkpoint, agent reset) need a synchronous trigger.
//!
//! These tests pin the synchronous path: a single call decays
//! importance and evicts qualifying Draft packs, returning the
//! eviction count.

use tdb_core::OwnerId;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::engine::Engine;
use tdb_retrieval::per_token::encode_per_token_keys;

const LOW_SALIENCE: f32 = 5.0;
const HIGH_SALIENCE: f32 = 95.0;
const EVICTION_THRESHOLD: f32 = 15.0;

fn make_pack(owner: OwnerId, marker: f32, salience: f32) -> KVPack {
    let retrieval_key = encode_per_token_keys(&[&[marker, 0.0, 0.0, 0.0]]);
    KVPack {
        id: 0,
        owner,
        retrieval_key,
        layers: (0..2).map(|i| KVLayerPayload { layer_idx: i, data: vec![marker; 64] }).collect(),
        salience,
        text: None,
    }
}

#[test]
fn sweep_now_on_empty_engine_returns_zero() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let evicted = engine.sweep_now(0.0, EVICTION_THRESHOLD).unwrap();
    assert_eq!(evicted, 0);
}

#[test]
fn sweep_now_evicts_low_importance_draft_packs() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let low_pack = engine.mem_write_pack(&make_pack(1, 1.0, LOW_SALIENCE)).unwrap();
    let high_pack = engine.mem_write_pack(&make_pack(1, 1.1, HIGH_SALIENCE)).unwrap();

    let evicted = engine.sweep_now(0.0, EVICTION_THRESHOLD).unwrap();

    // Only the low-importance Draft pack is removed; the high-
    // importance one survives.
    assert_eq!(evicted, 1);
    assert!(!engine.pack_exists(low_pack));
    assert!(engine.pack_exists(high_pack));
}

#[test]
fn sweep_now_with_hours_advances_decay() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Salience above the threshold *now*, but daily decay (0.995/day)
    // takes ~280 days to drag importance below 15. Advancing the
    // clock 300 days inside one sweep call should cross that line.
    let pack = engine.mem_write_pack(&make_pack(1, 1.0, 60.0)).unwrap();

    let evicted = engine.sweep_now(300.0 * HOURS_PER_DAY, EVICTION_THRESHOLD).unwrap();
    assert_eq!(evicted, 1);
    assert!(!engine.pack_exists(pack));
}

#[test]
fn sweep_now_is_idempotent_after_first_call_clears_evictable_set() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    engine.mem_write_pack(&make_pack(1, 1.0, LOW_SALIENCE)).unwrap();
    engine.mem_write_pack(&make_pack(1, 1.1, HIGH_SALIENCE)).unwrap();

    let first = engine.sweep_now(0.0, EVICTION_THRESHOLD).unwrap();
    let second = engine.sweep_now(0.0, EVICTION_THRESHOLD).unwrap();
    assert_eq!(first, 1);
    assert_eq!(second, 0);
}

// Local mirror of the constant in `engine.rs` so the test file
// stays self-contained.
const HOURS_PER_DAY: f32 = 24.0;
