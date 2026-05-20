//! ATDD for `Engine::mem_write_batch_packs` — eager batched write API.
//!
//! Distinct from the streaming `WriteBuffer`: the caller already has N
//! packs in hand and wants them persisted with a single fsync now. The
//! API is the right shape for calibration, file ingest, and bulk
//! migration paths that know their batch size up front.

use tdb_core::OwnerId;
use tdb_core::Tier;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::engine::Engine;
use tdb_retrieval::per_token::encode_per_token_keys;

const OWNER: OwnerId = 1;

fn make_pack(marker: f32, text: Option<&str>) -> KVPack {
    KVPack {
        id: 0,
        owner: OWNER,
        retrieval_key: encode_per_token_keys(&[&[marker, 0.0, 0.0, 0.0]]),
        layers: vec![KVLayerPayload { layer_idx: 0, data: vec![marker; 64] }],
        salience: 70.0,
        text: text.map(str::to_owned),
    }
}

#[test]
fn it_returns_pack_ids_in_input_order_strictly_increasing() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let packs: Vec<KVPack> =
        (0..10).map(|i| make_pack(0.1 + i as f32 * 0.01, Some(&format!("pack {i}")))).collect();
    let ids = engine.mem_write_batch_packs(&packs).unwrap();

    assert_eq!(ids.len(), 10);
    for window in ids.windows(2) {
        assert!(window[0] < window[1], "ids not strictly increasing: {ids:?}");
    }
}

#[test]
fn it_persists_every_pack_in_the_batch() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let packs: Vec<KVPack> =
        (0..15).map(|i| make_pack(0.1 + i as f32 * 0.01, Some(&format!("pack {i}")))).collect();
    let ids = engine.mem_write_batch_packs(&packs).unwrap();

    assert_eq!(engine.pack_count(), 15);
    let listed: std::collections::HashSet<_> =
        engine.list_packs(Some(OWNER)).into_iter().map(|(pid, _, _, _)| pid).collect();
    for id in &ids {
        assert!(listed.contains(id), "pack id {id} missing from list_packs");
    }
}

#[test]
fn batch_with_one_pack_equals_single_write_in_salience_and_tier() {
    // A one-pack batch must match the legacy single-pack write in every
    // observable field — same pack id semantics (monotonic), same salience,
    // same starting tier.
    let dir_a = tempfile::tempdir().unwrap();
    let mut eng_a = Engine::open(dir_a.path()).unwrap();
    let id_a = eng_a.mem_write_pack(&make_pack(0.5, Some("solo"))).unwrap();
    let listed_a = eng_a.list_packs(Some(OWNER));

    let dir_b = tempfile::tempdir().unwrap();
    let mut eng_b = Engine::open(dir_b.path()).unwrap();
    let ids_b = eng_b.mem_write_batch_packs(&[make_pack(0.5, Some("solo"))]).unwrap();
    let listed_b = eng_b.list_packs(Some(OWNER));

    assert_eq!(ids_b.len(), 1);
    assert_eq!(id_a, ids_b[0]);
    assert_eq!(listed_a.len(), listed_b.len());
    let (_, _, tier_a, importance_a) = listed_a[0];
    let (_, _, tier_b, importance_b) = listed_b[0];
    assert_eq!(tier_a, tier_b);
    assert!((importance_a - importance_b).abs() < 1e-5);
}

#[test]
fn empty_batch_is_a_no_op_returning_empty_vec() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let ids = engine.mem_write_batch_packs(&[]).unwrap();
    assert!(ids.is_empty());
    assert_eq!(engine.pack_count(), 0);
}

#[test]
fn batch_visible_immediately_without_explicit_flush() {
    // Distinct from the streaming write buffer: batch writes are eager and
    // already-persisted on return.
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let ids = engine
        .mem_write_batch_packs(&[make_pack(0.1, Some("a")), make_pack(0.2, Some("b"))])
        .unwrap();

    // No flush; no refresh — query immediately.
    let listed: Vec<_> =
        engine.list_packs(Some(OWNER)).into_iter().map(|(pid, _, _, _)| pid).collect();
    for id in &ids {
        assert!(listed.contains(id), "pack {id} not visible immediately after batch write");
    }
}

#[test]
fn batch_packs_start_in_draft_tier() {
    // Tier semantics must match single-pack writes: new packs land in Draft
    // at default salience (well below the Validated threshold).
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let mut low_salience = make_pack(0.1, None);
    low_salience.salience = 10.0; // well under Validated threshold (65)

    let ids = engine.mem_write_batch_packs(&[low_salience]).unwrap();
    let listed = engine.list_packs(Some(OWNER));
    let (_, _, tier, _) = listed.iter().find(|(pid, _, _, _)| *pid == ids[0]).unwrap();
    assert_eq!(*tier, Tier::Draft);
}

#[test]
fn batch_survives_engine_reopen() {
    // Recovery contract: batch writes use the same single-fsync path as the
    // streaming buffer's flush, so they must survive engine reopen
    // identically.
    let dir = tempfile::tempdir().unwrap();
    let ids = {
        let mut engine = Engine::open(dir.path()).unwrap();
        engine
            .mem_write_batch_packs(&[
                make_pack(0.1, Some("a")),
                make_pack(0.2, Some("b")),
                make_pack(0.3, Some("c")),
            ])
            .unwrap()
    };

    let engine = Engine::open(dir.path()).unwrap();
    let listed: std::collections::HashSet<_> =
        engine.list_packs(Some(OWNER)).into_iter().map(|(pid, _, _, _)| pid).collect();
    for id in &ids {
        assert!(listed.contains(id), "pack {id} lost across reopen");
    }
}
