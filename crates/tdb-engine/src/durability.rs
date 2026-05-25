//! Back-compat re-exports for the durability layer.
//!
//! The durability contract — the [`Durability`] trait and its default
//! [`DurabilityTracker`] implementation — moved to `tdb-storage` in
//! v0.8.1 so the storage layer owns its own contract. This shim
//! keeps the previous `tdb_engine::durability::*` import path
//! working for the property tests and any in-tree caller that hasn't
//! migrated yet.
//!
//! New code should import from `tdb_storage::durability` directly.

pub use tdb_storage::durability::{Durability, DurabilityTracker, WaitTimedOut};
