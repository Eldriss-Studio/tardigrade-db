//! ATDD for `Engine::list_packs` correctness against the
//! `PackDirectory` owner index.
//!
//! These tests pin the observable contract: enumeration returns each
//! pack's stored owner, deleted packs disappear, the importance-descending
//! sort is preserved, owner filtering is applied, and the directory's
//! state survives engine reopen (the recovery contract per CLAUDE.md).
//!
//! Performance — that `Engine::list_packs` answers from the in-memory
//! index without re-reading the block pool — is covered by
//! `experiments/list_packs_microbench.py`.

use tdb_core::OwnerId;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::engine::Engine;
use tdb_retrieval::per_token::encode_per_token_keys;

fn make_pack(owner: OwnerId, salience: f32, marker: f32) -> KVPack {
    let retrieval_key = encode_per_token_keys(&[&[marker, 0.0, 0.0, 0.0]]);
    KVPack {
        id: 0,
        owner,
        retrieval_key,
        layers: (0..2).map(|i| KVLayerPayload { layer_idx: i, data: vec![marker; 64] }).collect(),
        salience,
        text: Some(format!("pack for owner {owner} marker {marker}")),
    }
}

#[test]
fn it_lists_correct_owners_after_writes_to_multiple_owners() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let p1 = engine.mem_write_pack(&make_pack(1, 70.0, 0.1)).unwrap();
    let p2 = engine.mem_write_pack(&make_pack(2, 70.0, 0.2)).unwrap();
    let p3 = engine.mem_write_pack(&make_pack(1, 80.0, 0.3)).unwrap();
    let p4 = engine.mem_write_pack(&make_pack(3, 60.0, 0.4)).unwrap();

    let rows = engine.list_packs(None);
    let by_id: std::collections::HashMap<_, _> =
        rows.into_iter().map(|(pid, owner, _tier, _importance)| (pid, owner)).collect();

    assert_eq!(by_id.get(&p1), Some(&1));
    assert_eq!(by_id.get(&p2), Some(&2));
    assert_eq!(by_id.get(&p3), Some(&1));
    assert_eq!(by_id.get(&p4), Some(&3));
}

#[test]
fn it_returns_only_packs_for_filtered_owner() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    engine.mem_write_pack(&make_pack(1, 70.0, 0.1)).unwrap();
    engine.mem_write_pack(&make_pack(2, 70.0, 0.2)).unwrap();
    engine.mem_write_pack(&make_pack(1, 80.0, 0.3)).unwrap();
    engine.mem_write_pack(&make_pack(3, 60.0, 0.4)).unwrap();

    let only_owner_1 = engine.list_packs(Some(1));
    assert_eq!(only_owner_1.len(), 2);
    for row in &only_owner_1 {
        assert_eq!(row.1, 1, "expected only owner 1, got {}", row.1);
    }
}

#[test]
fn it_excludes_deleted_packs_from_list_packs() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let keep = engine.mem_write_pack(&make_pack(1, 70.0, 0.1)).unwrap();
    let drop = engine.mem_write_pack(&make_pack(1, 70.0, 0.2)).unwrap();

    engine.delete_pack(drop).unwrap();

    let rows = engine.list_packs(Some(1));
    let ids: Vec<_> = rows.into_iter().map(|(pid, _, _, _)| pid).collect();
    assert!(ids.contains(&keep));
    assert!(!ids.contains(&drop), "deleted pack {drop} still listed");
}

#[test]
fn it_returns_correct_owners_after_engine_reopen() {
    // Recovery contract per CLAUDE.md: derived state (the owner index) is
    // rebuildable from the durable segment trail. Write packs, drop the
    // engine, reopen, and verify list_packs still returns the right owners.
    let dir = tempfile::tempdir().unwrap();

    let (p1, p2, p3) = {
        let mut engine = Engine::open(dir.path()).unwrap();
        let p1 = engine.mem_write_pack(&make_pack(1, 70.0, 0.1)).unwrap();
        let p2 = engine.mem_write_pack(&make_pack(2, 70.0, 0.2)).unwrap();
        let p3 = engine.mem_write_pack(&make_pack(1, 80.0, 0.3)).unwrap();
        engine.flush().unwrap();
        (p1, p2, p3)
    };

    let engine = Engine::open(dir.path()).unwrap();
    let rows = engine.list_packs(None);
    let by_id: std::collections::HashMap<_, _> =
        rows.into_iter().map(|(pid, owner, _, _)| (pid, owner)).collect();

    assert_eq!(by_id.get(&p1), Some(&1), "owner lost for p1 after reopen");
    assert_eq!(by_id.get(&p2), Some(&2), "owner lost for p2 after reopen");
    assert_eq!(by_id.get(&p3), Some(&1), "owner lost for p3 after reopen");
}

#[test]
fn it_preserves_importance_descending_sort_through_owner_index() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    engine.mem_write_pack(&make_pack(1, 40.0, 0.1)).unwrap();
    engine.mem_write_pack(&make_pack(1, 90.0, 0.2)).unwrap();
    engine.mem_write_pack(&make_pack(1, 60.0, 0.3)).unwrap();
    engine.mem_write_pack(&make_pack(1, 75.0, 0.4)).unwrap();

    let rows = engine.list_packs(Some(1));
    let importances: Vec<f32> = rows.iter().map(|r| r.3).collect();
    for window in importances.windows(2) {
        assert!(window[0] >= window[1], "importance not sorted descending: {importances:?}");
    }
}

#[test]
fn it_returns_empty_for_owner_with_no_packs() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    engine.mem_write_pack(&make_pack(1, 70.0, 0.1)).unwrap();

    let rows = engine.list_packs(Some(999));
    assert!(rows.is_empty());
}
