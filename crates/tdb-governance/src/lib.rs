//! Adaptive Knowledge Lifecycle (AKL) — autonomous memory management for `TardigradeDB`.
//!
//! `tdb-governance` implements the self-curating layer of the Aeon architecture.
//! It answers the question: *given unlimited memory cells accumulating over time,
//! which ones should be kept, promoted, demoted, or evicted — and when?*
//!
//! No application code manages cell lifecycle. The AKL does it automatically based
//! on observed access patterns and elapsed time.
//!
//! # Importance Score (ι)
//!
//! Every cell carries an importance score ι ∈ \[0.0, 100.0\] managed by [`ImportanceScorer`].
//!
//! ```text
//! Event        │ Effect on ι
//! ─────────────┼──────────────────────────────────
//! Read access  │ +3.0  (capped at 100.0)
//! Write/update │ +5.0  (capped at 100.0)
//! Each day     │ × 0.995  (≈ 0.5% decay per day)
//! ```
//!
//! The daily decay factor of 0.995 produces a **half-life of approximately 138 days**
//! for a cell that is never accessed (ln(0.5) / ln(0.995) ≈ 138). A cell accessed
//! daily stabilizes around ι ≈ 60 (3 / (1 - 0.995)).
//!
//! # Tier State Machine
//!
//! [`TierStateMachine`] maps ι to one of three maturity tiers, using **hysteresis
//! gaps** to prevent oscillation when scores hover near a boundary:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                                                                 │
//! │  DRAFT ──────(ι ≥ 65)──────► VALIDATED ──────(ι ≥ 85)──────► CORE
//! │         ◄────(ι < 35)────                ◄────(ι < 60)────      │
//! │                                                                 │
//! │  Hysteresis gaps:  Draft↔Validated = 30 pts                    │
//! │                    Validated↔Core  = 25 pts                    │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Skipping tiers is supported: a new cell with ι = 90 will transition
//! `Draft → Validated → Core` in a single [`TierStateMachine::evaluate`] call.
//!
//! # Recency Decay
//!
//! [`recency_decay`] computes an exponential multiplier applied during retrieval
//! to down-weight cells that haven't been touched recently:
//!
//! ```text
//! r = exp(-Δt / τ),  τ = 30 days  (half-life ≈ 20.8 days)
//!
//! Δt = 0 days  →  r = 1.00  (fresh)
//! Δt ≈ 21 days →  r ≈ 0.50  (half-weight)
//! Δt = 30 days →  r ≈ 0.37  (one time-constant)
//! Δt = 90 days →  r ≈ 0.05  (near-zero)
//! ```
//!
//! This is applied as a **score multiplier** at query time, not as a modification
//! to the stored ι. It means a stale cell with a high importance score will still
//! rank lower than a recently-accessed cell with a moderate score.
//!
//! # Usage
//!
//! ```rust
//! use tdb_governance::scoring::ImportanceScorer;
//! use tdb_governance::tiers::TierStateMachine;
//! use tdb_governance::decay::recency_decay;
//! use tdb_core::Tier;
//!
//! let mut scorer = ImportanceScorer::new(50.0);
//! scorer.on_access();   // +3 → 53.0
//! scorer.on_update();   // +5 → 58.0
//! scorer.apply_daily_decay(30); // ×0.995^30 ≈ ×0.861 → ~49.9
//!
//! let mut tier = TierStateMachine::new(); // starts at Draft
//! tier.evaluate(scorer.importance());    // still Draft (< 65)
//!
//! // Recency factor for a cell last updated 21 days ago.
//! let r = recency_decay(21.0);
//! assert!(r > 0.49 && r < 0.51); // ≈ half-life
//!
//! // Final adjusted retrieval score:
//! let adjusted = scorer.importance() * r;
//! ```
//!
//! [`ImportanceScorer`]: scoring::ImportanceScorer
//! [`TierStateMachine`]: tiers::TierStateMachine
//! [`TierStateMachine::evaluate`]: tiers::TierStateMachine::evaluate
//! [`recency_decay`]: decay::recency_decay

#![deny(unsafe_code)]

pub mod decay;
pub mod scoring;
pub mod tiers;
