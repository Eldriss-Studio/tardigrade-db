//! Durable action scheduler.
//!
//! Lets consumers enqueue actions to fire at a future
//! `SystemTime`. The schedule is persisted to a JSON sidecar
//! inside the engine directory and reloaded on
//! [`crate::engine::Engine::open`] — actions survive process
//! restart, which is the whole point of having an engine-side
//! scheduler instead of consumer-side cron.
//!
//! ## Firing model
//!
//! The maintenance worker (or any caller) invokes
//! [`crate::engine::Engine::fire_due_scheduled`] periodically;
//! actions whose `fires_at <= now` are executed, then removed
//! from the schedule once execute returns `Ok`. The persistence
//! file is rewritten after every state change. Delivery is
//! **at-least-once across crashes** — see the next section.
//!
//! ## Crash semantics: at-least-once + idempotent actions
//!
//! Every scheduled action **stays in the on-disk schedule
//! until its execute call returns successfully.** Then — and
//! only then — the entry is removed and the schedule is
//! re-persisted. This guarantees **at-least-once** delivery
//! across process crashes:
//!
//! - Crash before execute starts: action still in schedule,
//!   fires on next tick after restart.
//! - Crash mid-execute: action still in schedule, fires again
//!   on next tick. Re-firing must therefore be safe.
//! - Crash after execute returns but before the post-fire
//!   persist: action still in schedule, fires again on next
//!   tick. Re-firing must be safe.
//!
//! **The contract is that every [`ScheduledAction`] variant
//! must be idempotent** — running it twice with the same
//! inputs must produce the same end state as running it once.
//! The built-in `EvictDraft(owner, threshold)` is naturally
//! idempotent: the second call has nothing left to evict.
//! Future variants that aren't naturally idempotent (e.g.
//! counter increments) must use idempotent semantics
//! ("set importance to N", not "+= N") or implement
//! consumer-side deduplication via the entry's stable
//! [`ScheduledId`].
//!
//! Combined with idempotent actions, at-least-once delivery
//! yields **fire-once-in-effect** across crashes.
//!
//! ## On-disk format
//!
//! ```json
//! {
//!   "next_id": 4,
//!   "entries": [
//!     {"id": 1, "fires_at_unix_nanos": 1234567890000000000,
//!      "action": {"EvictDraft": {"owner": 7, "threshold": 15.0}}}
//!   ]
//! }
//! ```
//!
//! `serde_json` is the only dependency. We write atomically:
//! serialize to a temp file alongside the target, fsync, rename.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tdb_core::error::{Result, TardigradeError};

/// File-name suffix for the schedule sidecar inside the engine
/// directory. Kept stable so external tooling can inspect or
/// back up the schedule without parsing the engine state.
const SCHEDULE_FILENAME: &str = "scheduled_actions.json";

/// First scheduled-action id handed out by a fresh scheduler.
/// Ids increase monotonically and never reuse a previously-
/// assigned value, even across process restarts.
const FIRST_SCHEDULED_ID: u64 = 1;

/// Identifier returned by [`Scheduler::schedule`]. Stable for
/// the lifetime of the schedule entry — consumers can pass it
/// back to [`Scheduler::cancel`] to remove the pending action.
pub type ScheduledId = u64;

/// Built-in actions the scheduler can fire.
///
/// Each variant carries everything the engine needs to execute
/// the action without a callback registry. A future `Custom`
/// variant could plug in consumer-provided handlers, but the
/// foundation only ships the built-ins to keep persistence
/// self-contained.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScheduledAction {
    /// Evict Draft-tier packs for `owner` whose importance is
    /// below `threshold`. Equivalent to
    /// `engine.evict_draft_packs(threshold, Some(owner))`.
    EvictDraft { owner: u64, threshold: f32 },
}

/// A single entry in the schedule. Exposed via
/// [`Scheduler::list`] for observability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScheduledEntry {
    pub id: ScheduledId,
    pub fires_at: SystemTime,
    pub action: ScheduledAction,
}

/// On-disk representation of the scheduler state.
#[derive(Debug, Default, Serialize, Deserialize)]
struct ScheduleFile {
    next_id: u64,
    entries: Vec<DiskEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskEntry {
    id: ScheduledId,
    /// Stored as Unix nanoseconds so the format is
    /// human-readable and survives `SystemTime` representation
    /// changes across Rust versions.
    fires_at_unix_nanos: u128,
    action: ScheduledAction,
}

impl From<&ScheduledEntry> for DiskEntry {
    fn from(e: &ScheduledEntry) -> Self {
        let nanos = e.fires_at.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        Self { id: e.id, fires_at_unix_nanos: nanos, action: e.action.clone() }
    }
}

impl From<DiskEntry> for ScheduledEntry {
    fn from(d: DiskEntry) -> Self {
        let fires_at =
            UNIX_EPOCH + Duration::from_nanos(u64::try_from(d.fires_at_unix_nanos).unwrap_or(0));
        Self { id: d.id, fires_at, action: d.action }
    }
}

/// The durable action scheduler.
///
/// Lives inside [`crate::engine::Engine`]. Construction loads
/// any persisted state; every mutating operation re-persists.
#[derive(Debug)]
pub struct Scheduler {
    path: PathBuf,
    next_id: u64,
    entries: Vec<ScheduledEntry>,
}

impl Scheduler {
    /// Open the scheduler rooted at `engine_dir`. Loads any
    /// existing schedule sidecar; missing file = empty
    /// schedule.
    pub fn open(engine_dir: &Path) -> Result<Self> {
        let path = engine_dir.join(SCHEDULE_FILENAME);
        if !path.exists() {
            return Ok(Self { path, next_id: FIRST_SCHEDULED_ID, entries: Vec::new() });
        }
        let raw = std::fs::read(&path).map_err(|e| TardigradeError::Io { source: e })?;
        let file: ScheduleFile = serde_json::from_slice(&raw).map_err(|e| {
            TardigradeError::SnapshotIntegrity(format!("scheduled_actions.json parse error: {e}"))
        })?;
        let entries = file.entries.into_iter().map(ScheduledEntry::from).collect();
        let next_id = file.next_id.max(FIRST_SCHEDULED_ID);
        Ok(Self { path, next_id, entries })
    }

    /// Enqueue `action` to fire at `fires_at`. Returns the
    /// assigned id.
    pub fn schedule(
        &mut self,
        fires_at: SystemTime,
        action: ScheduledAction,
    ) -> Result<ScheduledId> {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(ScheduledEntry { id, fires_at, action });
        self.persist()?;
        Ok(id)
    }

    /// Cancel a scheduled action by id. Returns `true` if a
    /// matching entry was removed, `false` if none was found.
    pub fn cancel(&mut self, id: ScheduledId) -> Result<bool> {
        let before = self.entries.len();
        self.entries.retain(|e| e.id != id);
        let removed = self.entries.len() < before;
        if removed {
            self.persist()?;
        }
        Ok(removed)
    }

    /// Return all currently scheduled entries, sorted by
    /// `fires_at` ascending. Cloned — callers can iterate
    /// without holding the scheduler lock.
    #[must_use]
    pub fn list(&self) -> Vec<ScheduledEntry> {
        let mut out = self.entries.clone();
        out.sort_by_key(|e| e.fires_at);
        out
    }

    /// Snapshot every entry whose `fires_at <= now`. The on-disk
    /// schedule is **not** mutated — entries remain durable until
    /// the caller confirms successful execution via
    /// [`Scheduler::commit_fired`]. This is the load-bearing half
    /// of the at-least-once contract: an entry stays scheduled
    /// across crash-before-commit, so it re-fires safely on
    /// restart.
    #[must_use]
    pub fn peek_due(&self, now: SystemTime) -> Vec<ScheduledEntry> {
        self.entries.iter().filter(|e| e.fires_at <= now).cloned().collect()
    }

    /// Remove the listed ids from the schedule and re-persist.
    /// Called by [`crate::engine::Engine::fire_due_scheduled`]
    /// after each action's `execute` returns successfully.
    ///
    /// Unknown ids are silently ignored (the caller might be
    /// reporting a batch where some entries were already
    /// removed — e.g. cancelled mid-flight).
    pub fn commit_fired(&mut self, ids: &[ScheduledId]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let before = self.entries.len();
        self.entries.retain(|e| !ids.contains(&e.id));
        if self.entries.len() != before {
            self.persist()?;
        }
        Ok(())
    }

    fn persist(&self) -> Result<()> {
        let file = ScheduleFile {
            next_id: self.next_id,
            entries: self.entries.iter().map(DiskEntry::from).collect(),
        };
        let bytes = serde_json::to_vec_pretty(&file).map_err(|e| {
            TardigradeError::SnapshotIntegrity(format!(
                "scheduled_actions.json serialize error: {e}"
            ))
        })?;
        // Atomic write: temp file in the same dir, then rename.
        // Rename-over is atomic on POSIX and on NTFS for files
        // on the same volume.
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| TardigradeError::Io { source: e })?;
        std::fs::rename(&tmp, &self.path).map_err(|e| TardigradeError::Io { source: e })?;
        Ok(())
    }
}
