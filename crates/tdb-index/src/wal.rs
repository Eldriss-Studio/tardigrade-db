//! Write-Ahead Log for crash-recoverable graph mutations.
//!
//! Append-only log file. Every mutation is logged before execution.
//! Checkpoint = flush all in-memory state + truncate WAL. Target: <1% write overhead.
