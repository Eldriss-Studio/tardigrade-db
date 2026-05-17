//! Engine-wide error type.
//!
//! `TardigradeError` is the single error type returned by every
//! public engine API. It derives both:
//!
//! - [`thiserror::Error`] — for ergonomic `?` propagation and
//!   `Display` / `source()` chains.
//! - [`miette::Diagnostic`] — for diagnostic-quality output:
//!   stable error codes (`tdb::storage::cell_not_found`,
//!   `tdb::retrieval::empty_corpus`, ...), inline help text
//!   with actionable next steps, and source spans (for variants
//!   that carry one).
//!
//! ## Backward compatibility
//!
//! Existing variants keep their `Display` shape verbatim — code
//! that grep'd "cell not found: 42" or pattern-matched on
//! variants pre-0.3 continues to work. New context-rich variants
//! were added alongside the old ones rather than replacing them,
//! and consumers can migrate at their own pace.
//!
//! ## Adding a variant
//!
//! Every new variant should:
//!
//! - Carry structured fields (not just `String`) for anything a
//!   consumer might want to programmatically match on.
//! - Have a `#[diagnostic(code = "tdb::<area>::<name>")]`
//!   annotation so consumers get a stable code.
//! - Have a `#[diagnostic(help("..."))]` line with a one-sentence
//!   hint on how to fix the underlying problem. The help text
//!   shows up in `cargo doc` and in miette's pretty-printed
//!   output.

use miette::Diagnostic;
use thiserror::Error;

use crate::kv_pack::PackId;
use crate::types::CellId;

#[derive(Debug, Error, Diagnostic)]
pub enum TardigradeError {
    #[error("cell not found: {0}")]
    #[diagnostic(
        code(tdb::storage::cell_not_found),
        help(
            "the cell id was never written or has been evicted; check engine.cell_count() and the deletion log"
        )
    )]
    CellNotFound(CellId),

    #[error("pack not found: {0}")]
    #[diagnostic(
        code(tdb::storage::pack_not_found),
        help(
            "the pack id was never written or has been deleted; check engine.list_packs() for the current set"
        )
    )]
    PackNotFound(PackId),

    #[error("segment full: cannot append to segment at {path}")]
    #[diagnostic(
        code(tdb::storage::segment_full),
        help(
            "the current segment hit its size cap; the engine should roll a new segment automatically — if you see this, the rollover path didn't fire"
        )
    )]
    SegmentFull { path: String },

    #[error("storage I/O error: {source}")]
    #[diagnostic(code(tdb::storage::io))]
    Io {
        #[from]
        source: std::io::Error,
    },

    #[error("quantization error: {0}")]
    #[diagnostic(code(tdb::storage::quantization))]
    Quantization(String),

    #[error("dimension mismatch: expected {expected}, got {actual} for {context}")]
    #[diagnostic(
        code(tdb::retrieval::dimension_mismatch),
        help(
            "the query key dimension must match the dimension of stored cells; verify your capture function emits the same hidden-state width on every write and read"
        )
    )]
    DimensionMismatch { expected: usize, actual: usize, context: String },

    #[error("WAL recovery failed: {0}")]
    #[diagnostic(
        code(tdb::index::wal_recovery),
        help(
            "the WAL on disk is corrupted or partially written; if this followed a hard crash, the segment may be salvageable with manual inspection of the WAL file"
        )
    )]
    WalRecovery(String),

    #[error("index error: {0}")]
    #[diagnostic(code(tdb::index::generic))]
    Index(String),

    #[error("governance error: {0}")]
    #[diagnostic(code(tdb::governance::generic))]
    Governance(String),

    #[error("capacity exceeded: {0}")]
    #[diagnostic(code(tdb::storage::capacity))]
    CapacityExceeded(String),

    // ── Snapshot / restore errors ────────────────────────────────
    #[error("not a TardigradeDB snapshot: {reason}")]
    #[diagnostic(
        code(tdb::snapshot::not_a_snapshot),
        help(
            "the file does not begin with the expected `tdb!` magic — verify it was produced by Engine::snapshot()"
        )
    )]
    NotATardigradeSnapshot { reason: String },

    #[error("unsupported snapshot format version {found}, supported: {supported}")]
    #[diagnostic(
        code(tdb::snapshot::unsupported_format_version),
        help(
            "the snapshot was created by a different (newer or much older) engine version; export it back to a portable format via the source engine"
        )
    )]
    UnsupportedFormatVersion { found: u8, supported: u8 },

    #[error("snapshot codec mismatch for {field}: snapshot={snapshot}, engine={engine}")]
    #[diagnostic(
        code(tdb::snapshot::codec_mismatch),
        help(
            "the snapshot was written by an engine using a different codec; build an explicit codec-conversion tool, do not silently reinterpret"
        )
    )]
    SnapshotCodecMismatch { field: String, snapshot: String, engine: String },

    #[error("snapshot integrity check failed: {0}")]
    #[diagnostic(
        code(tdb::snapshot::integrity),
        help(
            "the SHA-256 of the snapshot payload does not match the manifest; the archive is corrupted in transit or storage"
        )
    )]
    SnapshotIntegrity(String),

    // ── Retrieval errors (new in v0.3) ───────────────────────────
    /// Returned when a query targets an owner that has no packs
    /// stored. Distinguishes "no results found" (empty Ok vec)
    /// from "you queried something that never had data" — the
    /// latter is almost always a consumer bug.
    #[error("empty corpus for owner {owner}: no packs have ever been stored for this owner")]
    #[diagnostic(
        code(tdb::retrieval::empty_corpus),
        help(
            "verify the owner id matches the one used at write time; for multi-agent setups, the same engine handle must be shared across clients (use TardigradeClient.builder().with_engine(engine))"
        )
    )]
    EmptyCorpus { owner: u64 },

    /// Returned by query paths when no usable retrieval key
    /// could be derived for the query. The `hint` field carries
    /// a context-specific reason (e.g. `kv_capture_fn` returned
    /// `None`, or query text was empty after tokenization).
    #[error("no query key produced: {hint}")]
    #[diagnostic(
        code(tdb::retrieval::no_query_key),
        help(
            "a query needs a non-empty key to score against; for TardigradeClient, configure kv_capture_fn — for direct Engine use, ensure the passed key has length > 0"
        )
    )]
    NoQueryKey { hint: String },

    /// Returned when the streaming write buffer's coalesced
    /// flush fails. The cause is preserved as the diagnostic
    /// source so consumers can drill in.
    #[error("write buffer flush failed at offset {failed_offset}: {detail}")]
    #[diagnostic(
        code(tdb::storage::flush_failed),
        help(
            "the engine could not durably persist the buffered writes; check disk space, fsync errors, and underlying filesystem health"
        )
    )]
    FlushFailed { failed_offset: u64, detail: String },
}

pub type Result<T> = std::result::Result<T, TardigradeError>;
