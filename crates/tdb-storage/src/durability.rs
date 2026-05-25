//! Durability boundary tracking — the storage-layer contract that
//! lets engine consumers reason about "what is durable" without
//! coupling to the concrete tracker implementation.
//!
//! # Why this lives in `tdb-storage`
//!
//! Durability is fundamentally a storage concern: the offset advances
//! when a segment fsync returns. Putting the abstraction in
//! `tdb-storage` lets the storage layer own its own contract instead
//! of expressing it as a concrete struct nested inside the engine.
//! Future replicated / async-batched durability impls can replace
//! [`DurabilityTracker`] under the trait without touching the engine.
//!
//! # The contract
//!
//! Two monotonic counters express the durability boundary:
//!
//! - `issued`: bumps every time a write is accepted (regardless of
//!   whether the underlying storage has fsynced it). A write goes
//!   from "issued" to "durable" via an fsync somewhere downstream.
//! - `durable`: bumps every time an fsync completes successfully.
//!   Always `<= issued`. Catches up to `issued` after each flush.
//!
//! Confirmed reads snapshot `issued` at request entry, run the
//! retrieval pipeline, then wait until `durable` reaches the
//! snapshot — every concurrent in-flight write that existed at
//! request time is now durable when the read returns.
//!
//! # Implementation
//!
//! [`DurabilityTracker`] uses `std::sync::Condvar` + `AtomicU64`, no
//! async runtime. The project's existing concurrency model is
//! `Arc<RwLock<>>` from std accessed under `py.detach()`; staying
//! inside std avoids dragging tokio into a synchronous codebase
//! just for one watch channel.
//!
//! # Invariants
//!
//! - `issued` and `durable` are both monotonically non-decreasing.
//! - `durable <= issued` at all times.
//! - `publish_durable(n)` is idempotent: republishing an offset at
//!   or below the current value is a no-op.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Error returned by [`Durability::wait_durable`] when the caller's
/// deadline passes before `durable` reaches the target.
#[derive(Debug, Clone, Copy)]
pub struct WaitTimedOut {
    /// Elapsed wall-clock time the caller spent waiting before
    /// giving up. Useful for surfacing in error messages.
    pub waited: Duration,
}

/// The durability boundary contract.
///
/// Every concrete durability impl provides the same four observable
/// facts: the current durable offset, the current issued offset, a
/// way to bump issued, and a way to block until durable catches up
/// to a target. Engine consumers depend on this trait so a future
/// replicated / async-batched / metric-instrumented impl can plug in
/// without touching the engine's write or read paths.
///
/// Object-safe by construction — methods are `&self`, no generics,
/// no `Self` return types. Stored in the engine as
/// `Arc<dyn Durability + Send + Sync>`.
pub trait Durability: std::fmt::Debug + Send + Sync {
    /// Current durable offset — every write at or below this offset
    /// has reached stable storage.
    fn current_durable(&self) -> u64;

    /// Current issued offset — every accepted write up to and
    /// including this offset is in flight or durable.
    fn current_issued(&self) -> u64;

    /// Atomically bump `issued` and return the new value.
    ///
    /// Called from every write path the moment the engine accepts
    /// the write. Distinct from durability — the write may sit in a
    /// buffer for an arbitrary time before its fsync lands.
    fn issue_one(&self) -> u64;

    /// Mark every issued offset up to and including `offset` as
    /// durable. Idempotent: a second call with the same or lower
    /// value is a no-op.
    ///
    /// Implementations **must** clamp `offset` to `current_issued()`
    /// so `durable <= issued` is an unconditional invariant.
    fn publish_durable(&self, offset: u64);

    /// Block until `current_durable() >= target` or `deadline`
    /// elapses.
    ///
    /// Callers should drop the engine lock before invoking this so
    /// writers can advance the offset concurrently.
    ///
    /// # Errors
    ///
    /// Returns [`WaitTimedOut`] if `current_durable()` does not reach
    /// `target` within `deadline`.
    fn wait_durable(&self, target: u64, deadline: Duration) -> Result<(), WaitTimedOut>;
}

/// In-process durability tracker — the default [`Durability`]
/// implementation backing every embedded engine instance.
///
/// One instance per engine. Cloneable references via [`std::sync::Arc`]
/// let readers wait without holding the engine lock — the Python
/// binding's confirmed-read path borrows an `Arc<dyn Durability>`,
/// drops the engine read lock, then calls [`Self::wait_durable`]
/// outside the lock so writers can advance the offset concurrently.
#[derive(Debug, Default)]
pub struct DurabilityTracker {
    issued: AtomicU64,
    durable: AtomicU64,
    /// Companion to [`Self::notify`]. The Condvar protocol requires
    /// holding this mutex when both signalling and waiting.
    notify_lock: Mutex<()>,
    notify: Condvar,
}

impl DurabilityTracker {
    /// Construct an empty tracker. Both offsets start at 0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            issued: AtomicU64::new(0),
            durable: AtomicU64::new(0),
            notify_lock: Mutex::new(()),
            notify: Condvar::new(),
        }
    }
}

impl Durability for DurabilityTracker {
    fn current_durable(&self) -> u64 {
        self.durable.load(Ordering::Acquire)
    }

    fn current_issued(&self) -> u64 {
        self.issued.load(Ordering::Acquire)
    }

    fn issue_one(&self) -> u64 {
        self.issued.fetch_add(1, Ordering::AcqRel) + 1
    }

    fn publish_durable(&self, offset: u64) {
        // Clamp to issued — the durability boundary cannot lead
        // acceptance. Caught by a property test in
        // `tests/durability_proptest.rs::durable_never_exceeds_issued`.
        let effective = offset.min(self.issued.load(Ordering::Acquire));
        // `fetch_max` semantics via CAS — keep the maximum so
        // out-of-order publishes don't regress the offset.
        let mut current = self.durable.load(Ordering::Acquire);
        loop {
            if effective <= current {
                return;
            }
            match self.durable.compare_exchange_weak(
                current,
                effective,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
        // Acquiring the notify mutex here means waiters that
        // observed `durable < target` and are about to wait have
        // either already entered `wait_timeout` (and will be woken
        // by `notify_all`) or are still holding the mutex (in which
        // case we serialise behind them and they see the updated
        // value on their next check).
        let _guard = self.notify_lock.lock().expect("durability notify mutex poisoned");
        self.notify.notify_all();
    }

    fn wait_durable(&self, target: u64, deadline: Duration) -> Result<(), WaitTimedOut> {
        // Fast path — already durable, skip the mutex acquire.
        if self.current_durable() >= target {
            return Ok(());
        }
        let start = Instant::now();
        let mut guard = self.notify_lock.lock().expect("durability notify mutex poisoned");
        loop {
            if self.current_durable() >= target {
                return Ok(());
            }
            let elapsed = start.elapsed();
            // `checked_sub` rather than `-`: clippy's strict mode
            // forbids the unguarded operator on `Duration`.
            let Some(remaining) = deadline.checked_sub(elapsed) else {
                return Err(WaitTimedOut { waited: elapsed });
            };
            let (next_guard, timeout_result) = self
                .notify
                .wait_timeout(guard, remaining)
                .expect("durability notify mutex poisoned");
            guard = next_guard;
            if timeout_result.timed_out() && self.current_durable() < target {
                return Err(WaitTimedOut { waited: start.elapsed() });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn new_tracker_has_zero_offsets() {
        let t = DurabilityTracker::new();
        assert_eq!(t.current_durable(), 0);
        assert_eq!(t.current_issued(), 0);
    }

    #[test]
    fn issue_one_returns_strictly_increasing_offsets() {
        let t = DurabilityTracker::new();
        assert_eq!(t.issue_one(), 1);
        assert_eq!(t.issue_one(), 2);
        assert_eq!(t.issue_one(), 3);
        assert_eq!(t.current_issued(), 3);
    }

    /// Helper: bump `issued` to a target value so the tracker's
    /// invariant `durable <= issued` permits a publish at that
    /// level. Mirrors the engine's own write-then-fsync pattern.
    fn issue_to(t: &DurabilityTracker, target: u64) {
        while t.current_issued() < target {
            t.issue_one();
        }
    }

    #[test]
    fn publish_durable_is_monotonic() {
        let t = DurabilityTracker::new();
        issue_to(&t, 10);
        t.publish_durable(5);
        assert_eq!(t.current_durable(), 5);
        t.publish_durable(3); // Lower — must not regress.
        assert_eq!(t.current_durable(), 5);
        t.publish_durable(10);
        assert_eq!(t.current_durable(), 10);
    }

    #[test]
    fn wait_durable_returns_immediately_when_already_at_target() {
        let t = DurabilityTracker::new();
        issue_to(&t, 5);
        t.publish_durable(5);
        let start = Instant::now();
        assert!(t.wait_durable(3, Duration::from_mins(1)).is_ok());
        assert!(start.elapsed() < Duration::from_millis(10));
    }

    #[test]
    fn publish_durable_clamps_to_issued() {
        // Documented invariant: publishing past `issued` advances
        // durable only as far as `issued` permits. Caught by the
        // proptest gate; pinned here as an example.
        let t = DurabilityTracker::new();
        t.publish_durable(100);
        assert_eq!(t.current_durable(), 0, "no writes issued — durable must stay at 0");
        t.issue_one();
        t.publish_durable(100);
        assert_eq!(t.current_durable(), 1, "one write issued — durable clamps to 1");
    }

    #[test]
    fn wait_durable_times_out_when_no_publish_occurs() {
        let t = DurabilityTracker::new();
        let err = t.wait_durable(1, Duration::from_millis(50)).unwrap_err();
        assert!(err.waited >= Duration::from_millis(50));
        assert!(err.waited < Duration::from_millis(500));
    }

    #[test]
    fn wait_durable_unblocks_when_publish_arrives() {
        let t = Arc::new(DurabilityTracker::new());
        // Issue ahead so publish can advance durable to 5.
        issue_to(&t, 5);
        let waiter = {
            let t = Arc::clone(&t);
            thread::spawn(move || t.wait_durable(5, Duration::from_secs(2)))
        };
        // Give the waiter a moment to enter its wait.
        thread::sleep(Duration::from_millis(50));
        t.publish_durable(5);
        let result = waiter.join().unwrap();
        assert!(result.is_ok());
    }

    #[test]
    fn multiple_waiters_unblock_on_single_publish() {
        let t = Arc::new(DurabilityTracker::new());
        issue_to(&t, 7);
        let waiters: Vec<_> = (0..3)
            .map(|_| {
                let t = Arc::clone(&t);
                thread::spawn(move || t.wait_durable(7, Duration::from_secs(2)))
            })
            .collect();
        thread::sleep(Duration::from_millis(50));
        t.publish_durable(7);
        for waiter in waiters {
            assert!(waiter.join().unwrap().is_ok());
        }
    }

    /// Verify the trait is dyn-compatible — `Arc<dyn Durability>`
    /// must compile. If this stops working, callers that store the
    /// tracker as `Arc<dyn Durability + Send + Sync>` will break.
    #[test]
    fn durability_trait_is_object_safe() {
        let tracker: Arc<dyn Durability + Send + Sync> = Arc::new(DurabilityTracker::new());
        assert_eq!(tracker.current_durable(), 0);
        let issued = tracker.issue_one();
        tracker.publish_durable(issued);
        assert_eq!(tracker.current_durable(), issued);
    }
}
