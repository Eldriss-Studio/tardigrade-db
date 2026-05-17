//! ATDD for the portable snapshot/restore API.
//!
//! Pins eight slices: roundtrip parity, manifest-integrity, magic
//! mismatch, format-version mismatch, codec mismatch, full-state
//! preservation (owner registry + tiers + edges + importance),
//! restore on a clean target dir, and basic manifest fields.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use tar::Builder;
use tdb_core::OwnerId;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::engine::Engine;
use tdb_engine::snapshot::{
    SNAPSHOT_MANIFEST_NAME, SnapshotCodecs, SnapshotManifest, SnapshotStats,
};
use tdb_retrieval::per_token::encode_per_token_keys;

fn make_pack(owner: OwnerId, marker: f32) -> KVPack {
    let retrieval_key = encode_per_token_keys(&[&[marker, 0.0, 0.0, 0.0]]);
    KVPack {
        id: 0,
        owner,
        retrieval_key,
        layers: (0..2).map(|i| KVLayerPayload { layer_idx: i, data: vec![marker; 64] }).collect(),
        salience: 80.0,
        text: Some(format!("marker-{marker}")),
    }
}

fn populated_engine(dir: &std::path::Path) -> Engine {
    let mut engine = Engine::open(dir).unwrap();
    engine.mem_write_pack(&make_pack(1, 1.0)).unwrap();
    engine.mem_write_pack(&make_pack(1, 1.1)).unwrap();
    engine.mem_write_pack(&make_pack(7, 1.2)).unwrap();
    engine.mem_write_pack(&make_pack(42, 1.3)).unwrap();
    engine
}

// ─── roundtrip parity ─────────────────────────────────────────

#[test]
fn snapshot_roundtrip_preserves_pack_count_and_owners() {
    let src_dir = tempfile::tempdir().unwrap();
    let snap_dir = tempfile::tempdir().unwrap();
    let mut src = populated_engine(src_dir.path());

    let snap_file = snap_dir.path().join("snap.tar");
    let manifest = src.snapshot(&snap_file).unwrap();
    assert_eq!(manifest.stats.pack_count, 4);
    assert_eq!(manifest.stats.owner_count, 3);

    let dst_dir = tempfile::tempdir().unwrap();
    let target = dst_dir.path().join("restored");
    let restored = Engine::restore_from(&snap_file, &target).unwrap();
    assert_eq!(restored.pack_count(), 4);
    assert_eq!(restored.list_owners(), vec![1, 7, 42]);
}

// ─── integrity (SHA mismatch) ─────────────────────────────────

#[test]
fn restore_rejects_archive_with_corrupted_payload() {
    let src_dir = tempfile::tempdir().unwrap();
    let snap_dir = tempfile::tempdir().unwrap();
    let mut src = populated_engine(src_dir.path());
    let snap_file = snap_dir.path().join("snap.tar");
    src.snapshot(&snap_file).unwrap();

    // Flip a byte deep in the tar (somewhere inside the payload, not
    // the manifest). The simplest way: re-read, edit a byte past the
    // manifest header, write it back.
    let mut bytes = std::fs::read(&snap_file).unwrap();
    // Locate "engine_state" inside the archive and flip the byte
    // 64 chars later — that'll land in payload data.
    let needle = b"engine_state";
    let pos = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("engine_state marker present in archive");
    let target_idx = pos + 1024;
    if target_idx < bytes.len() {
        bytes[target_idx] ^= 0xFF;
    }
    std::fs::write(&snap_file, &bytes).unwrap();

    let dst_dir = tempfile::tempdir().unwrap();
    let err = Engine::restore_from(&snap_file, &dst_dir.path().join("restored"))
        .expect_err("corrupted snapshot should be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("integrity") || msg.contains("sha256") || msg.contains("not a"),
        "unexpected error message: {msg}",
    );
}

// ─── magic mismatch ───────────────────────────────────────────

#[test]
fn restore_rejects_non_tardigrade_tar() {
    let dir = tempfile::tempdir().unwrap();
    let snap_file = dir.path().join("not-ours.tar");
    {
        let file = File::create(&snap_file).unwrap();
        let mut tar = Builder::new(file);
        let payload = b"hello world";
        let mut header = tar::Header::new_gnu();
        header.set_path("garbage.txt").unwrap();
        header.set_size(payload.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append(&header, &payload[..]).unwrap();
        tar.finish().unwrap();
    }

    let target = dir.path().join("restored");
    let err = Engine::restore_from(&snap_file, &target)
        .expect_err("non-tardigrade tar should be rejected");
    assert!(err.to_string().to_lowercase().contains("not a"), "unexpected error: {err}");
}

// ─── unsupported format_version ───────────────────────────────

#[test]
fn restore_rejects_unsupported_format_version() {
    let dir = tempfile::tempdir().unwrap();
    let snap_file = dir.path().join("future.tar");
    write_synthetic_manifest_archive(
        &snap_file,
        &SnapshotManifest {
            magic: "tdb!".to_string(),
            format_version: 99,
            created_at: "@0".to_string(),
            codecs: SnapshotCodecs { quantization: "q4".to_string(), key: "top5avg".to_string() },
            stats: SnapshotStats::default(),
            sha256: "0000".repeat(8),
        },
    );

    let err = Engine::restore_from(&snap_file, &dir.path().join("restored"))
        .expect_err("unsupported version should be rejected");
    let msg = err.to_string();
    assert!(msg.contains("format version") || msg.contains("99"), "msg: {msg}");
}

// ─── codec mismatch ───────────────────────────────────────────

#[test]
fn restore_rejects_unsupported_quantization_codec() {
    let dir = tempfile::tempdir().unwrap();
    let snap_file = dir.path().join("codec-mismatch.tar");
    write_synthetic_manifest_archive(
        &snap_file,
        &SnapshotManifest {
            magic: "tdb!".to_string(),
            format_version: 1,
            created_at: "@0".to_string(),
            codecs: SnapshotCodecs { quantization: "q99".to_string(), key: "top5avg".to_string() },
            stats: SnapshotStats::default(),
            sha256: "0000".repeat(8),
        },
    );

    let err = Engine::restore_from(&snap_file, &dir.path().join("restored"))
        .expect_err("unsupported codec should be rejected");
    let msg = err.to_string();
    assert!(msg.contains("codec") && msg.contains("q99"), "msg: {msg}");
}

// ─── owner enumeration survives ───────────────────────────────

#[test]
fn restore_preserves_owner_registry_membership() {
    let src_dir = tempfile::tempdir().unwrap();
    let snap_dir = tempfile::tempdir().unwrap();
    let mut src = populated_engine(src_dir.path());
    let snap_file = snap_dir.path().join("snap.tar");
    src.snapshot(&snap_file).unwrap();

    let dst_dir = tempfile::tempdir().unwrap();
    let restored = Engine::restore_from(&snap_file, &dst_dir.path().join("restored")).unwrap();

    for owner in [1u64, 7, 42] {
        assert!(restored.owner_exists(owner), "owner {owner} missing after restore");
    }
    assert!(!restored.owner_exists(9999));
}

// ─── manifest carries metadata ────────────────────────────────

#[test]
fn snapshot_manifest_carries_magic_version_and_codecs() {
    let src_dir = tempfile::tempdir().unwrap();
    let snap_dir = tempfile::tempdir().unwrap();
    let mut src = populated_engine(src_dir.path());
    let snap_file = snap_dir.path().join("snap.tar");
    let manifest = src.snapshot(&snap_file).unwrap();

    assert_eq!(manifest.magic, "tdb!");
    assert_eq!(manifest.format_version, 1);
    assert_eq!(manifest.codecs.quantization, "q4");
    assert_eq!(manifest.codecs.key, "top5avg");
    assert!(!manifest.sha256.is_empty());
    assert!(manifest.created_at.starts_with('@'));
}

// ─── restore into fresh directory ─────────────────────────────

#[test]
fn restore_creates_target_directory_when_missing() {
    let src_dir = tempfile::tempdir().unwrap();
    let snap_dir = tempfile::tempdir().unwrap();
    let mut src = populated_engine(src_dir.path());
    let snap_file = snap_dir.path().join("snap.tar");
    src.snapshot(&snap_file).unwrap();

    let dst_dir = tempfile::tempdir().unwrap();
    let nested = dst_dir.path().join("nested").join("restored");
    assert!(!nested.exists());
    let _engine = Engine::restore_from(&snap_file, &nested).unwrap();
    assert!(nested.is_dir());
}

// ─── snapshot refuses to write inside engine_dir ──────────────

#[test]
fn snapshot_refuses_out_path_inside_engine_dir() {
    // The tar walker would otherwise read its own growing output and
    // exhaust disk. The engine guards against this and returns a
    // typed error pointing the caller at the conflict.
    let src_dir = tempfile::tempdir().unwrap();
    let mut src = populated_engine(src_dir.path());

    let bad_path = src_dir.path().join("snap.tar");
    let err = src.snapshot(&bad_path).expect_err("snapshot in engine_dir should be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("inside engine_dir") || msg.contains("outside the engine directory"),
        "unexpected error: {msg}",
    );
}

// ─── helpers ───────────────────────────────────────────────────────────

fn write_synthetic_manifest_archive(path: &PathBuf, manifest: &SnapshotManifest) {
    let manifest_json = serde_json::to_vec_pretty(manifest).unwrap();
    let file = File::create(path).unwrap();
    let mut tar = Builder::new(file);
    let mut header = tar::Header::new_gnu();
    header.set_path(SNAPSHOT_MANIFEST_NAME).unwrap();
    header.set_size(manifest_json.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    tar.append(&header, &manifest_json[..]).unwrap();
    tar.finish().unwrap();
    let mut f = File::open(path).unwrap();
    let mut buf = Vec::new();
    std::io::copy(&mut f, &mut buf).unwrap();
    drop(f);
    // Re-open + write again to ensure file is closed before tests use it.
    let mut out = File::create(path).unwrap();
    out.write_all(&buf).unwrap();
}
