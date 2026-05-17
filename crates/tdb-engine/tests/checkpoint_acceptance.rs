//! ATDD for `CheckpointRepository`.
//!
//! Labeled checkpoints sit on top of the portable snapshot API:
//! each `save_from(engine, label)` writes a tar archive into
//! `<root>/<label>/<seq>.tar` and returns a `CheckpointEntry`.
//! Consumers can list, find the latest by label, and restore the
//! latest into a fresh directory — the foundation primitive behind
//! save-game slots, agent-state autosaves, and replay testing.

use tdb_core::OwnerId;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::checkpoint::CheckpointRepository;
use tdb_engine::engine::Engine;
use tdb_retrieval::per_token::encode_per_token_keys;

const PACK_SALIENCE: f32 = 80.0;
const LAYER_COUNT: u16 = 2;
const LAYER_DIM: usize = 64;

fn make_pack(owner: OwnerId, marker: f32) -> KVPack {
    let retrieval_key = encode_per_token_keys(&[&[marker, 0.0, 0.0, 0.0]]);
    KVPack {
        id: 0,
        owner,
        retrieval_key,
        layers: (0..LAYER_COUNT)
            .map(|i| KVLayerPayload { layer_idx: i, data: vec![marker; LAYER_DIM] })
            .collect(),
        salience: PACK_SALIENCE,
        text: None,
    }
}

fn fresh_engine_with_packs() -> (tempfile::TempDir, Engine) {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    engine.mem_write_pack(&make_pack(1, 1.0)).unwrap();
    engine.mem_write_pack(&make_pack(7, 1.1)).unwrap();
    engine.mem_write_pack(&make_pack(42, 1.2)).unwrap();
    (dir, engine)
}

#[test]
fn save_from_creates_label_subdirectory_with_seq_one() {
    let (_engine_dir, mut engine) = fresh_engine_with_packs();
    let repo_dir = tempfile::tempdir().unwrap();
    let repo = CheckpointRepository::new(repo_dir.path().to_path_buf());

    let entry = repo.save_from(&mut engine, "autosave").unwrap();
    assert_eq!(entry.label, "autosave");
    assert_eq!(entry.seq, 1);
    assert!(entry.path.is_file(), "checkpoint file should exist at {:?}", entry.path);
    assert!(entry.path.starts_with(repo_dir.path().join("autosave")));
    assert_eq!(entry.manifest.stats.pack_count, 3);
}

#[test]
fn save_from_assigns_monotonic_sequence_per_label() {
    let (_engine_dir, mut engine) = fresh_engine_with_packs();
    let repo_dir = tempfile::tempdir().unwrap();
    let repo = CheckpointRepository::new(repo_dir.path().to_path_buf());

    let a = repo.save_from(&mut engine, "autosave").unwrap();
    let b = repo.save_from(&mut engine, "autosave").unwrap();
    let c = repo.save_from(&mut engine, "manual").unwrap();

    assert_eq!((a.seq, b.seq, c.seq), (1, 2, 1));
}

#[test]
fn list_returns_all_checkpoints_sorted_by_label_then_seq() {
    let (_engine_dir, mut engine) = fresh_engine_with_packs();
    let repo_dir = tempfile::tempdir().unwrap();
    let repo = CheckpointRepository::new(repo_dir.path().to_path_buf());

    repo.save_from(&mut engine, "manual").unwrap();
    repo.save_from(&mut engine, "autosave").unwrap();
    repo.save_from(&mut engine, "autosave").unwrap();

    let all = repo.list(None).unwrap();
    let summary: Vec<(String, u32)> = all.into_iter().map(|e| (e.label, e.seq)).collect();
    assert_eq!(
        summary,
        vec![("autosave".to_string(), 1), ("autosave".to_string(), 2), ("manual".to_string(), 1),],
    );
}

#[test]
fn list_filters_by_label_when_provided() {
    let (_engine_dir, mut engine) = fresh_engine_with_packs();
    let repo_dir = tempfile::tempdir().unwrap();
    let repo = CheckpointRepository::new(repo_dir.path().to_path_buf());

    repo.save_from(&mut engine, "autosave").unwrap();
    repo.save_from(&mut engine, "manual").unwrap();

    let only_auto = repo.list(Some("autosave")).unwrap();
    assert_eq!(only_auto.len(), 1);
    assert_eq!(only_auto[0].label, "autosave");
}

#[test]
fn latest_returns_highest_seq_for_label() {
    let (_engine_dir, mut engine) = fresh_engine_with_packs();
    let repo_dir = tempfile::tempdir().unwrap();
    let repo = CheckpointRepository::new(repo_dir.path().to_path_buf());

    repo.save_from(&mut engine, "autosave").unwrap();
    repo.save_from(&mut engine, "autosave").unwrap();
    let third = repo.save_from(&mut engine, "autosave").unwrap();

    let latest = repo.latest("autosave").unwrap().unwrap();
    assert_eq!(latest.seq, third.seq);
}

#[test]
fn latest_on_unknown_label_returns_ok_none() {
    let repo_dir = tempfile::tempdir().unwrap();
    let repo = CheckpointRepository::new(repo_dir.path().to_path_buf());
    assert!(repo.latest("nope").unwrap().is_none());
}

#[test]
fn restore_latest_yields_engine_with_same_packs() {
    let (_engine_dir, mut engine) = fresh_engine_with_packs();
    let repo_dir = tempfile::tempdir().unwrap();
    let repo = CheckpointRepository::new(repo_dir.path().to_path_buf());

    repo.save_from(&mut engine, "autosave").unwrap();

    let restore_dir = tempfile::tempdir().unwrap();
    let target = restore_dir.path().join("restored");
    let restored = repo.restore_latest("autosave", &target).unwrap();

    assert_eq!(restored.list_owners(), vec![1, 7, 42]);
    assert_eq!(restored.pack_count(), 3);
}

#[test]
fn restore_latest_on_unknown_label_returns_typed_error() {
    let repo_dir = tempfile::tempdir().unwrap();
    let repo = CheckpointRepository::new(repo_dir.path().to_path_buf());

    let restore_dir = tempfile::tempdir().unwrap();
    let target = restore_dir.path().join("restored");
    let result = repo.restore_latest("missing", &target);
    assert!(result.is_err(), "expected error, got {:?}", result.as_ref().map(|_| "Ok"));
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.to_lowercase().contains("missing") || msg.to_lowercase().contains("no checkpoint"),
        "expected message to mention the missing label, got: {msg}",
    );
}
