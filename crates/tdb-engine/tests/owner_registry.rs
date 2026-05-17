//! ATDD for the owner registry — `list_owners`, `owner_exists`, `delete_owner`.
//!
//! Pins M1.1 from the foundation plan. Owner enumeration is
//! foundational for any consumer with multiple parties (multi-agent
//! runtimes, multi-NPC games, multi-tenant `SaaS`). Before this slice
//! owners were implicit integers managed entirely by the consumer;
//! the engine had no way to list them, check existence, or remove
//! one cleanly.

use tdb_core::OwnerId;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::engine::Engine;
use tdb_retrieval::per_token::encode_per_token_keys;

fn make_pack(owner: OwnerId, marker: f32) -> KVPack {
    let retrieval_key = encode_per_token_keys(&[&[marker, 0.0, 0.0, 0.0]]);
    KVPack {
        id: 0,
        owner,
        retrieval_key,
        layers: (0..2).map(|i| KVLayerPayload { layer_idx: i, data: vec![marker; 64] }).collect(),
        salience: 80.0,
        text: None,
    }
}

// ─── list_owners ────────────────────────────────────────────────────

#[test]
fn list_owners_on_empty_engine_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.list_owners(), Vec::<OwnerId>::new());
}

#[test]
fn list_owners_returns_distinct_owners_sorted_ascending() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Multiple packs per owner — list_owners must dedupe.
    engine.mem_write_pack(&make_pack(42, 1.0)).unwrap();
    engine.mem_write_pack(&make_pack(7, 1.1)).unwrap();
    engine.mem_write_pack(&make_pack(42, 1.2)).unwrap();
    engine.mem_write_pack(&make_pack(1, 1.3)).unwrap();
    engine.mem_write_pack(&make_pack(7, 1.4)).unwrap();

    assert_eq!(engine.list_owners(), vec![1, 7, 42]);
}

// ─── owner_exists ────────────────────────────────────────────────────

#[test]
fn owner_exists_false_on_empty_engine() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(dir.path()).unwrap();
    assert!(!engine.owner_exists(1));
}

#[test]
fn owner_exists_true_after_first_write_for_that_owner() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    engine.mem_write_pack(&make_pack(99, 1.0)).unwrap();
    assert!(engine.owner_exists(99));
    assert!(!engine.owner_exists(98));
    assert!(!engine.owner_exists(100));
}

// ─── delete_owner ────────────────────────────────────────────────────

#[test]
fn delete_owner_removes_only_target_packs() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    engine.mem_write_pack(&make_pack(1, 1.0)).unwrap();
    engine.mem_write_pack(&make_pack(7, 1.1)).unwrap();
    engine.mem_write_pack(&make_pack(7, 1.2)).unwrap();
    engine.mem_write_pack(&make_pack(42, 1.3)).unwrap();
    assert_eq!(engine.pack_count(), 4);

    let removed = engine.delete_owner(7).unwrap();
    assert_eq!(removed, 2, "delete_owner should report exactly 2 packs removed");

    // Owners 1 and 42 untouched; owner 7 gone.
    assert_eq!(engine.list_owners(), vec![1, 42]);
    assert_eq!(engine.pack_count(), 2);
    assert!(!engine.owner_exists(7));
}

#[test]
fn delete_owner_with_no_packs_returns_zero() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    engine.mem_write_pack(&make_pack(1, 1.0)).unwrap();

    let removed = engine.delete_owner(999).unwrap();
    assert_eq!(removed, 0);
    // Existing owner unaffected.
    assert!(engine.owner_exists(1));
}

#[test]
fn delete_owner_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    engine.mem_write_pack(&make_pack(5, 1.0)).unwrap();

    let first = engine.delete_owner(5).unwrap();
    let second = engine.delete_owner(5).unwrap();
    assert_eq!(first, 1);
    assert_eq!(second, 0);
    assert!(!engine.owner_exists(5));
}
