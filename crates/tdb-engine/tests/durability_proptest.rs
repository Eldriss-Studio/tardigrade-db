//! Property-based tests for `DurabilityTracker`.
//!
//! The example-based unit tests inside `durability.rs` cover the
//! shapes I had in mind when writing them — single waiter unblocks,
//! multi-waiter coalesces, monotonicity, timeout-on-no-publish.
//! These properties cover the shapes I didn't think of:
//!
//! 1. The `durable <= issued` invariant holds across *any*
//!    interleaving of `issue_one` / `publish_durable(n)` calls.
//! 2. Both counters are monotonically non-decreasing under any
//!    operation sequence.
//! 3. `publish_durable(n)` is idempotent — replaying a sequence
//!    with arbitrarily duplicated `publish` calls produces the same
//!    final state.
//! 4. The fast path of `wait_durable` (target already reached)
//!    returns within a tight time budget regardless of inputs.
//!
//! The operation-sequence model uses `OpKind` enum values generated
//! by proptest, replayed against a real tracker. The model state is
//! `(issued, durable)` as a pair of `u64`s; we assert the real
//! tracker's state matches the model after every operation.

use proptest::prelude::*;
use std::time::Duration;
use tdb_engine::durability::{Durability, DurabilityTracker};

/// Operations the model supports — same surface as the real tracker.
#[derive(Debug, Clone, Copy)]
enum Op {
    /// Bump issued by one.
    Issue,
    /// Advance durable toward the target. `target_offset_from_issued`
    /// is added to current issued to get the publish target; if
    /// negative, the publish targets a value below the current
    /// durable, which must be a no-op.
    Publish { target_offset_from_issued: i32 },
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        Just(Op::Issue),
        (-4i32..=4i32).prop_map(|d| Op::Publish { target_offset_from_issued: d }),
    ]
}

fn op_sequence_strategy() -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(op_strategy(), 0..=64)
}

/// Apply an op sequence to a fresh tracker, returning the final
/// `(issued, durable)` pair. Asserts the invariant after every step.
fn run_sequence(ops: &[Op]) -> Result<(u64, u64), TestCaseError> {
    let tracker = DurabilityTracker::new();
    for op in ops {
        let issued_before = tracker.current_issued();
        let durable_before = tracker.current_durable();
        match *op {
            Op::Issue => {
                let new_issued = tracker.issue_one();
                prop_assert!(
                    new_issued >= issued_before,
                    "issue_one regressed issued from {issued_before} to {new_issued}"
                );
            }
            Op::Publish { target_offset_from_issued } => {
                // Compute a target that may be below or above the
                // current durable. Saturating math avoids u64
                // underflow when the offset is negative.
                #[allow(
                    clippy::cast_sign_loss,
                    // Reason: branch guards on `>= 0` / `< 0` so each
                    // cast lane receives a non-negative i32 — sign loss
                    // is impossible. Verified by the property test
                    // itself: any sign-loss-induced wraparound would
                    // make the durable monotonicity assertion fail.
                )]
                let target = if target_offset_from_issued >= 0 {
                    issued_before.saturating_add(target_offset_from_issued as u64)
                } else {
                    issued_before.saturating_sub((-target_offset_from_issued) as u64)
                };
                tracker.publish_durable(target);
            }
        }
        let issued_after = tracker.current_issued();
        let durable_after = tracker.current_durable();
        // Core invariant: durable always <= issued.
        prop_assert!(
            durable_after <= issued_after,
            "invariant violated: durable {durable_after} > issued {issued_after}"
        );
        // Monotonicity: neither counter regressed across the op.
        prop_assert!(
            issued_after >= issued_before,
            "issued regressed: {issued_before} → {issued_after}"
        );
        prop_assert!(
            durable_after >= durable_before,
            "durable regressed: {durable_before} → {durable_after}"
        );
    }
    Ok((tracker.current_issued(), tracker.current_durable()))
}

proptest! {
    /// Core invariant: under any interleaving of operations,
    /// `durable <= issued` holds after every step.
    #[test]
    fn durable_never_exceeds_issued(ops in op_sequence_strategy()) {
        run_sequence(&ops)?;
    }

    /// Monotonicity: both counters are non-decreasing across any op.
    /// Checked inline by `run_sequence`; this test just exercises
    /// longer sequences to make the assertion budget bite if
    /// monotonicity fails.
    #[test]
    fn counters_are_monotonic(
        ops in prop::collection::vec(op_strategy(), 0..=256),
    ) {
        run_sequence(&ops)?;
    }

    /// Idempotence: duplicating every publish in the sequence
    /// produces the same final `(issued, durable)` pair. Catches
    /// any accidental "publish twice = double effect" bug.
    #[test]
    fn publish_is_idempotent(ops in op_sequence_strategy()) {
        let original = run_sequence(&ops)?;

        // Build a sequence where every `Publish` appears twice.
        let mut doubled = Vec::with_capacity(ops.len() * 2);
        for op in &ops {
            doubled.push(*op);
            if matches!(op, Op::Publish { .. }) {
                doubled.push(*op);
            }
        }
        let doubled_result = run_sequence(&doubled)?;
        prop_assert_eq!(original.1, doubled_result.1,
            "doubling every publish changed the final durable offset");
    }

    /// Fast-path latency: when `wait_durable` is called with a target
    /// at or below the current durable, it returns in well under the
    /// deadline. We can't measure latency directly (the assertion
    /// would flake under load), but we *can* assert success with a
    /// short deadline — a slow path here would time out instead.
    ///
    /// Bumps `issued` to the publish target first so the clamp lets
    /// `durable` advance there. Mirrors the engine's pattern of
    /// issuing before publishing.
    #[test]
    fn wait_durable_returns_immediately_at_or_below_current(
        publish_target in 0u64..=1000,
        wait_target_offset in -1000i64..=0,
    ) {
        let tracker = DurabilityTracker::new();
        for _ in 0..publish_target {
            tracker.issue_one();
        }
        tracker.publish_durable(publish_target);
        #[allow(
            clippy::cast_sign_loss,
            // Reason: same as the `Op::Publish` cast lane — branches
            // guard on `>= 0` / `< 0` before each `as u64`.
        )]
        let wait_target = if wait_target_offset >= 0 {
            publish_target.saturating_add(wait_target_offset as u64)
        } else {
            publish_target.saturating_sub((-wait_target_offset) as u64)
        };
        // 50ms is generous for a fast-path return that shouldn't
        // hit the condvar at all. A real timeout would mean the
        // fast-path check is broken.
        let result = tracker.wait_durable(wait_target, Duration::from_millis(50));
        prop_assert!(
            result.is_ok(),
            "wait_durable({wait_target}) timed out when durable was already {publish_target}"
        );
    }
}
