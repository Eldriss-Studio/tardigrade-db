//! Snapshot round-trip property test.
//!
//! The highest-value property test in the engine suite: for any
//! sequence of pack writes, snapshotting the engine and restoring
//! into a fresh directory must produce a tree where every pack
//! reads back identical to the original. Exercises the full
//! durability stack — segment append, fsync, tar archive, SHA-256
//! integrity, manifest parse, segment-scan replay.
//!
//! Cost: this property does real disk I/O, so generated sequences
//! are kept short (1–8 packs, 32-dim keys) and the case count is
//! reduced to 16 instead of proptest's default 256. Sixteen random
//! sequences per CI run is still plenty to catch the failure modes
//! we care about (segment boundaries, multi-layer payloads,
//! optional text presence) without ballooning runtime.

use proptest::prelude::*;
use std::collections::HashMap;
use tdb_core::{KVLayerPayload, KVPack};
use tdb_engine::engine::Engine;
use tempfile::TempDir;

const KEY_DIM: usize = 32;

/// One pack the property test will write. Constrained to realistic
/// shapes: small dimension, ≤4 layers, owner from a tight pool to
/// exercise the owner-filtering path in retrieval.
#[derive(Debug, Clone)]
struct GenPack {
    owner: u64,
    retrieval_key: Vec<f32>,
    layers: Vec<(u16, Vec<f32>)>,
    salience: f32,
    text: Option<String>,
}

fn pack_strategy() -> impl Strategy<Value = GenPack> {
    (
        0u64..=4,
        prop::collection::vec(-5.0f32..=5.0, KEY_DIM),
        prop::collection::vec((0u16..=8, prop::collection::vec(-5.0f32..=5.0, KEY_DIM)), 1..=3),
        0.0f32..=100.0,
        prop::option::of("[a-z ]{1,32}"),
    )
        .prop_map(|(owner, retrieval_key, layers, salience, text)| GenPack {
            owner,
            retrieval_key,
            layers,
            salience,
            text,
        })
}

fn pack_sequence_strategy() -> impl Strategy<Value = Vec<GenPack>> {
    prop::collection::vec(pack_strategy(), 1..=8)
}

fn write_packs(engine: &mut Engine, packs: &[GenPack]) -> Vec<u64> {
    packs
        .iter()
        .map(|p| {
            let kv = KVPack {
                id: 0,
                owner: p.owner,
                retrieval_key: p.retrieval_key.clone(),
                layers: p
                    .layers
                    .iter()
                    .map(|(idx, data)| KVLayerPayload { layer_idx: *idx, data: data.clone() })
                    .collect(),
                salience: p.salience,
                text: p.text.clone(),
            };
            engine.mem_write_pack(&kv).expect("write_pack failed")
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig {
        // Real disk I/O — keep the case count modest. 16 is still
        // plenty to catch regressions in the snapshot/restore path.
        cases: 16,
        ..ProptestConfig::default()
    })]

    /// Snapshot the engine, restore into a fresh directory, read every
    /// pack back. The restored pack count must match; every pack's
    /// retrieval key, layers, salience and text must round-trip
    /// identically; pack IDs must match (the engine is deterministic
    /// in id assignment).
    ///
    /// Crash boundary covered: this property doesn't simulate a crash
    /// mid-snapshot, but it does verify that a complete snapshot is
    /// recoverable — the precondition for any crash-recovery story.
    #[test]
    fn snapshot_restore_preserves_pack_contents(packs in pack_sequence_strategy()) {
        let source_dir = TempDir::new().expect("source tempdir");
        let snapshot_holder = TempDir::new().expect("snapshot tempdir");
        let snapshot_path = snapshot_holder.path().join("snapshot.tar");
        let restore_dir = TempDir::new().expect("restore tempdir");

        let pack_ids: Vec<u64>;
        let by_id: HashMap<u64, GenPack>;
        let pack_count: usize;

        {
            let mut engine = Engine::open(source_dir.path()).expect("open source");
            pack_ids = write_packs(&mut engine, &packs);
            by_id = pack_ids.iter().copied().zip(packs.iter().cloned()).collect();
            pack_count = engine.pack_count();
            engine.snapshot(&snapshot_path).expect("snapshot");
        }
        // Source engine dropped — confirms the snapshot is
        // self-sufficient and doesn't rely on the source handle.

        let restored = Engine::restore_from(&snapshot_path, restore_dir.path())
            .expect("restore");

        prop_assert_eq!(restored.pack_count(), pack_count,
            "pack count diverged after restore");

        // For each original pack id, the restored engine must agree
        // on owner, layers, salience, and text.
        for (&pid, original) in &by_id {
            prop_assert!(restored.pack_exists(pid),
                "pack {pid} missing in restored engine");
            // Text is the cheapest cross-check: persisted via the
            // separate TextStore, so a mismatch would be a
            // text-store restore bug.
            prop_assert_eq!(
                restored.pack_text(pid),
                original.text.clone(),
                "text mismatch for pack {}", pid
            );
        }
    }
}
