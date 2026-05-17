//! ATDD for the streaming write buffer.
//!
//! Pins the buffered-ingest contract: when an engine is opened with
//! a `BufferConfig`, single-pack writes accumulate in a bounded
//! buffer and flush with a single fsync once `max_batch_size` is
//! reached or `flush_buffer` is called. The unbuffered path
//! (default) is unchanged.
//!
//! Reliability contract per the plan:
//! - Unbuffered = confirmed-immediately.
//! - Buffered = confirmed-on-flush. Queries do not see buffered
//!   packs until flush.
//! - `Engine::flush` auto-flushes the write buffer, so callers
//!   that already use flush as the durability barrier stay correct.

use tdb_core::OwnerId;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::engine::{BufferConfig, Engine};
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

// ─── buffer disabled is the default (parity with prior) ────────

#[test]
fn unbuffered_engine_makes_writes_immediately_visible() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    let pid = engine.mem_write_pack(&make_pack(1, 1.0)).unwrap();
    // Without a buffer, the pack must be queryable immediately and
    // pack_count reflects it.
    assert_eq!(engine.pack_count(), 1);
    assert!(engine.pack_exists(pid));
}

// ─── buffered writes batch into a single fsync window ──────────

#[test]
fn buffered_writes_are_invisible_until_flush() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open_with_write_buffer(
        dir.path(),
        BufferConfig { max_batch_size: 100, max_idle_ms: 0 },
    )
    .unwrap();

    // 5 writes < buffer threshold ⇒ all pending, none visible.
    for i in 0..5 {
        engine.mem_write_pack(&make_pack(1, 1.0 + i as f32 * 0.1)).unwrap();
    }
    assert_eq!(engine.pack_count(), 0, "buffered packs must not bump pack_count before flush");

    engine.flush_buffer().unwrap();
    assert_eq!(engine.pack_count(), 5, "after flush_buffer all 5 packs must be visible");
}

#[test]
fn buffer_auto_flushes_when_max_batch_size_reached() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open_with_write_buffer(
        dir.path(),
        BufferConfig { max_batch_size: 4, max_idle_ms: 0 },
    )
    .unwrap();

    // 4 writes hits the threshold mid-loop ⇒ auto-flushes.
    for i in 0..4 {
        engine.mem_write_pack(&make_pack(1, 1.0 + i as f32 * 0.1)).unwrap();
    }
    assert_eq!(engine.pack_count(), 4, "buffer should have auto-flushed at max_batch_size");
}

// ─── engine.flush() drains the write buffer ───────────────────

#[test]
fn engine_flush_drains_the_write_buffer() {
    // Callers that already use `flush()` as their durability barrier
    // must not have to learn about the buffer separately.
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open_with_write_buffer(
        dir.path(),
        BufferConfig { max_batch_size: 100, max_idle_ms: 0 },
    )
    .unwrap();

    engine.mem_write_pack(&make_pack(1, 1.0)).unwrap();
    engine.mem_write_pack(&make_pack(7, 1.1)).unwrap();
    assert_eq!(engine.pack_count(), 0);

    engine.flush().unwrap();
    assert_eq!(engine.pack_count(), 2);
}

// ─── flush_buffer is idempotent ───────────────────────────────

#[test]
fn flush_buffer_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open_with_write_buffer(
        dir.path(),
        BufferConfig { max_batch_size: 100, max_idle_ms: 0 },
    )
    .unwrap();

    engine.mem_write_pack(&make_pack(1, 1.0)).unwrap();
    engine.flush_buffer().unwrap();
    let count_after_first = engine.pack_count();
    engine.flush_buffer().unwrap();
    engine.flush_buffer().unwrap();
    // No spurious writes.
    assert_eq!(engine.pack_count(), count_after_first);
}

// ─── pack ids are assigned at enqueue, stable through flush ────

#[test]
fn pack_ids_are_returned_synchronously_and_resolve_after_flush() {
    // Consumers need the pack_id immediately so they can link
    // memories together — buffering must not defer ID assignment.
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open_with_write_buffer(
        dir.path(),
        BufferConfig { max_batch_size: 100, max_idle_ms: 0 },
    )
    .unwrap();

    let pid_a = engine.mem_write_pack(&make_pack(1, 1.0)).unwrap();
    let pid_b = engine.mem_write_pack(&make_pack(1, 1.1)).unwrap();
    assert_ne!(pid_a, pid_b, "pack ids must be distinct even before flush");

    engine.flush_buffer().unwrap();
    assert!(engine.pack_exists(pid_a));
    assert!(engine.pack_exists(pid_b));
}

// ─── durability after flush ────────────────────────────────────

#[test]
fn buffered_writes_survive_close_and_reopen_after_flush() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut engine = Engine::open_with_write_buffer(
            dir.path(),
            BufferConfig { max_batch_size: 100, max_idle_ms: 0 },
        )
        .unwrap();
        engine.mem_write_pack(&make_pack(1, 1.0)).unwrap();
        engine.mem_write_pack(&make_pack(7, 1.1)).unwrap();
        engine.flush_buffer().unwrap();
    }

    let reopened = Engine::open(dir.path()).unwrap();
    assert_eq!(reopened.pack_count(), 2);
    assert_eq!(reopened.list_owners(), vec![1, 7]);
}

#[test]
fn unflushed_writes_are_lost_on_close_by_contract() {
    // Buffered = confirmed-on-flush. The contract is explicit: a
    // process that crashes mid-batch loses unflushed packs. Pin this
    // so future "convenience" auto-flush-on-drop changes don't
    // silently violate the documented semantics.
    let dir = tempfile::tempdir().unwrap();
    {
        let mut engine = Engine::open_with_write_buffer(
            dir.path(),
            BufferConfig { max_batch_size: 100, max_idle_ms: 0 },
        )
        .unwrap();
        engine.mem_write_pack(&make_pack(1, 1.0)).unwrap();
        // No flush — drop here.
    }

    let reopened = Engine::open(dir.path()).unwrap();
    assert_eq!(reopened.pack_count(), 0, "unflushed buffered packs must not appear on reopen");
}
