//! ATDD for the durable action scheduler.
//!
//! The scheduler lets consumers enqueue actions to fire at a
//! future `SystemTime`. Critical property: the schedule survives
//! process restart — an NPC that scheduled "remind me about
//! this in a week" should not lose that intent if the game
//! restarts. The maintenance worker reads + fires due actions
//! on its tick; this suite drives the firing path directly via
//! `Engine::fire_due_scheduled(now)`.

use std::time::{Duration, SystemTime};

use tdb_core::OwnerId;
use tdb_core::kv_pack::{KVLayerPayload, KVPack};
use tdb_engine::engine::Engine;
use tdb_engine::scheduler::ScheduledAction;
use tdb_retrieval::per_token::encode_per_token_keys;

const LOW_SALIENCE: f32 = 5.0;
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
fn empty_engine_has_no_scheduled_actions() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(dir.path()).unwrap();
    assert_eq!(engine.list_scheduled().len(), 0);
}

#[test]
fn schedule_returns_monotonic_ids() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let at = SystemTime::now() + Duration::from_mins(1);

    let a = engine
        .schedule(at, ScheduledAction::EvictDraft { owner: 1, threshold: EVICTION_THRESHOLD })
        .unwrap();
    let b = engine
        .schedule(at, ScheduledAction::EvictDraft { owner: 2, threshold: EVICTION_THRESHOLD })
        .unwrap();
    let c = engine
        .schedule(at, ScheduledAction::EvictDraft { owner: 3, threshold: EVICTION_THRESHOLD })
        .unwrap();

    assert_eq!((a, b, c), (1, 2, 3));
}

#[test]
fn list_scheduled_returns_entries_sorted_by_fire_time() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let now = SystemTime::now();

    // Schedule out of order so the sort is exercised.
    engine
        .schedule(
            now + Duration::from_secs(30),
            ScheduledAction::EvictDraft { owner: 1, threshold: EVICTION_THRESHOLD },
        )
        .unwrap();
    engine
        .schedule(
            now + Duration::from_secs(10),
            ScheduledAction::EvictDraft { owner: 2, threshold: EVICTION_THRESHOLD },
        )
        .unwrap();
    engine
        .schedule(
            now + Duration::from_secs(20),
            ScheduledAction::EvictDraft { owner: 3, threshold: EVICTION_THRESHOLD },
        )
        .unwrap();

    let entries = engine.list_scheduled();
    assert_eq!(entries.len(), 3);
    // First entry fires first.
    assert!(entries[0].fires_at <= entries[1].fires_at);
    assert!(entries[1].fires_at <= entries[2].fires_at);
}

#[test]
fn fire_due_runs_actions_whose_time_has_passed() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();

    // Two low-importance draft packs that the scheduled eviction
    // should remove when fired.
    let pack_a = engine.mem_write_pack(&make_pack(1, 1.0, LOW_SALIENCE)).unwrap();
    let pack_b = engine.mem_write_pack(&make_pack(1, 1.1, LOW_SALIENCE)).unwrap();

    let past = SystemTime::now() - Duration::from_secs(1);
    engine
        .schedule(past, ScheduledAction::EvictDraft { owner: 1, threshold: EVICTION_THRESHOLD })
        .unwrap();

    let fired = engine.fire_due_scheduled(SystemTime::now()).unwrap();
    assert_eq!(fired, 1, "expected one action to fire");
    assert!(!engine.pack_exists(pack_a));
    assert!(!engine.pack_exists(pack_b));
    // Action removed from schedule after firing.
    assert_eq!(engine.list_scheduled().len(), 0);
}

#[test]
fn fire_due_leaves_future_actions_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let future = SystemTime::now() + Duration::from_hours(1);
    let id = engine
        .schedule(future, ScheduledAction::EvictDraft { owner: 1, threshold: EVICTION_THRESHOLD })
        .unwrap();

    let fired = engine.fire_due_scheduled(SystemTime::now()).unwrap();
    assert_eq!(fired, 0);

    let entries = engine.list_scheduled();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].id, id);
}

#[test]
fn schedule_survives_engine_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let at = SystemTime::now() + Duration::from_mins(1);

    {
        let mut engine = Engine::open(dir.path()).unwrap();
        engine
            .schedule(at, ScheduledAction::EvictDraft { owner: 7, threshold: EVICTION_THRESHOLD })
            .unwrap();
        assert_eq!(engine.list_scheduled().len(), 1);
    }

    let reopened = Engine::open(dir.path()).unwrap();
    let entries = reopened.list_scheduled();
    assert_eq!(entries.len(), 1, "schedule must survive engine reopen");
    match entries[0].action {
        ScheduledAction::EvictDraft { owner, threshold } => {
            assert_eq!(owner, 7);
            assert!((threshold - EVICTION_THRESHOLD).abs() < f32::EPSILON);
        }
    }
}

#[test]
fn cancel_removes_a_scheduled_action() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let at = SystemTime::now() + Duration::from_mins(1);
    let id_a = engine
        .schedule(at, ScheduledAction::EvictDraft { owner: 1, threshold: EVICTION_THRESHOLD })
        .unwrap();
    let _id_b = engine
        .schedule(at, ScheduledAction::EvictDraft { owner: 2, threshold: EVICTION_THRESHOLD })
        .unwrap();

    let removed = engine.cancel_scheduled(id_a).unwrap();
    assert!(removed, "expected cancel to report removal");
    assert_eq!(engine.list_scheduled().len(), 1);
    assert_eq!(engine.list_scheduled()[0].id, 2);
}

#[test]
fn at_least_once_replay_is_idempotent_for_evict_draft() {
    // Simulates the crash-after-execute-before-commit window:
    // the action's effect (eviction) has already landed but the
    // on-disk schedule still has the entry. On restart, fire
    // again — for an idempotent action this must be a no-op.
    let dir = tempfile::tempdir().unwrap();
    {
        let mut engine = Engine::open(dir.path()).unwrap();
        // Two low-importance draft packs that the eviction
        // would target.
        engine.mem_write_pack(&make_pack(1, 1.0, LOW_SALIENCE)).unwrap();
        engine.mem_write_pack(&make_pack(1, 1.1, LOW_SALIENCE)).unwrap();

        // Eviction runs (e.g. via sweep_now); we apply the
        // same effect without going through the scheduler so
        // we can then schedule it as a "pending replay".
        let evicted = engine.sweep_now(0.0, EVICTION_THRESHOLD).unwrap();
        assert_eq!(evicted, 2);

        // Now schedule the same action with a past fire time —
        // representing the durable schedule entry that would
        // remain after a mid-fire crash.
        let past = SystemTime::now() - Duration::from_secs(1);
        engine
            .schedule(past, ScheduledAction::EvictDraft { owner: 1, threshold: EVICTION_THRESHOLD })
            .unwrap();
    }

    // Reopen — the schedule survives, the eviction's effect
    // also survives. Replay must not blow up or have any
    // negative effect.
    let mut reopened = Engine::open(dir.path()).unwrap();
    assert_eq!(reopened.list_scheduled().len(), 1, "schedule survives reopen");

    let fired = reopened.fire_due_scheduled(SystemTime::now()).unwrap();
    assert_eq!(fired, 1, "replayed action executes");
    assert_eq!(reopened.pack_count(), 0, "no harm — packs were already gone");
    assert_eq!(reopened.list_scheduled().len(), 0, "entry now committed");
}

#[test]
fn cancel_unknown_id_is_noop() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path()).unwrap();
    let removed = engine.cancel_scheduled(999).unwrap();
    assert!(!removed);
}
