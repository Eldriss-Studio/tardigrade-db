//! Prometheus-format metrics layer — closes the last item on the
//! April `SpacetimeDB` audit's §5.1 "Adopt Now" list.
//!
//! # Design
//!
//! The `metrics` facade crate stays cheap when no recorder is
//! installed: the macros expand to no-ops. When the engine boots, it
//! installs a `metrics-exporter-prometheus` recorder *once per
//! process* (via [`std::sync::OnceLock`]) so every engine
//! instance in the same process feeds the same global registry. The
//! [`Engine::metrics_prometheus_text`] accessor calls the handle's
//! `render()` method to produce the Prometheus text exposition
//! format.
//!
//! [`Engine::metrics_prometheus_text`]: crate::engine::Engine::metrics_prometheus_text
//!
//! # Why render-only
//!
//! `metrics-exporter-prometheus`'s default mode spins up its own
//! HTTP server on a configurable port. We don't want that — the
//! `FastAPI` bridge already runs an HTTP server, and adding a second
//! one complicates deployment. Render-only mode (the handle's
//! `render()` method) returns the Prometheus text without any
//! HTTP machinery; the bridge serves it from a `/metrics` route.
//!
//! # Metric inventory
//!
//! - `tdb_durable_offset` (gauge) — monotonic durability boundary.
//! - `tdb_issued_offset` (gauge) — monotonic acceptance counter.
//! - `tdb_confirmed_read_total{outcome}` (counter) — confirmed-read
//!   outcomes by category (`ok`, `timeout`).
//! - `tdb_confirmed_read_wait_seconds` (histogram) — wall-clock
//!   wait latency for confirmed reads that succeeded.
//! - `tdb_engine_open_seconds` (counter) — time spent in
//!   [`Engine::open`] / variants (replay + state rebuild).
//! - `tdb_snapshot_write_seconds` (histogram) — wall-clock duration
//!   of [`Engine::snapshot`] calls.
//!
//! [`Engine::open`]: crate::engine::Engine::open
//! [`Engine::snapshot`]: crate::engine::Engine::snapshot

use std::sync::OnceLock;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

static METRICS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Install the process-wide Prometheus recorder if it has not been
/// installed yet, and return a [`PrometheusHandle`] capable of
/// rendering the registry to text.
///
/// Idempotent within a process: the first caller installs, every
/// subsequent caller gets a clone of the same handle. Designed for
/// the engine constructor — every `Engine::open*` call invokes
/// this so multi-engine processes share one registry.
///
/// # Panics
///
/// Panics if `metrics-exporter-prometheus` cannot install its
/// recorder because another `metrics` recorder has already been
/// installed by a foreign consumer. That's a configuration error
/// the engine cannot recover from at runtime.
pub fn install_or_get_prometheus_handle() -> PrometheusHandle {
    METRICS_HANDLE
        .get_or_init(|| {
            PrometheusBuilder::new().install_recorder().expect(
                "another `metrics` recorder is already installed in this process; \
                     remove it before opening a TardigradeDB engine, or use \
                     a fresh process",
            )
        })
        .clone()
}

/// Wait for `snapshot` to become durable, emitting metrics for the
/// outcome. Wraps [`crate::durability::Durability::wait_durable`]
/// so the confirmed-read call site doesn't need to duplicate the
/// counter+histogram boilerplate.
///
/// Returns the elapsed wait duration on success (also recorded in
/// the wait-seconds histogram) and propagates
/// [`crate::durability::WaitTimedOut`] on timeout (recorded as the
/// `outcome="timeout"` variant of the counter).
///
/// # Errors
///
/// Returns [`crate::durability::WaitTimedOut`] when the underlying
/// tracker's deadline passes before durability catches up.
pub fn wait_durable_with_metrics(
    tracker: &dyn crate::durability::Durability,
    snapshot: u64,
    timeout: std::time::Duration,
) -> Result<std::time::Duration, crate::durability::WaitTimedOut> {
    let start = std::time::Instant::now();
    let result = tracker.wait_durable(snapshot, timeout);
    let elapsed = start.elapsed();
    match &result {
        Ok(()) => {
            metrics::counter!(
                names::CONFIRMED_READ_TOTAL,
                "outcome" => "ok",
            )
            .increment(1);
            metrics::histogram!(names::CONFIRMED_READ_WAIT_SECONDS).record(elapsed.as_secs_f64());
            Ok(elapsed)
        }
        Err(err) => {
            metrics::counter!(
                names::CONFIRMED_READ_TOTAL,
                "outcome" => "timeout",
            )
            .increment(1);
            Err(*err)
        }
    }
}

/// Metric family names — kept in one place so call sites and
/// dashboards stay in sync.
pub mod names {
    /// Monotonic durability boundary, in arbitrary units (currently
    /// "writes acknowledged as durable since process start").
    pub const DURABLE_OFFSET: &str = "tdb_durable_offset";

    /// Monotonic acceptance counter, in the same unit as
    /// [`DURABLE_OFFSET`]; always `>= DURABLE_OFFSET`.
    pub const ISSUED_OFFSET: &str = "tdb_issued_offset";

    /// Counter of confirmed-read outcomes. Label `outcome` is one of
    /// `ok` (the wait succeeded) or `timeout` (the wait exceeded the
    /// caller's deadline).
    pub const CONFIRMED_READ_TOTAL: &str = "tdb_confirmed_read_total";

    /// Histogram of successful confirmed-read wait durations, in
    /// seconds. Only the `ok` outcome contributes; timeouts have
    /// their elapsed time recorded but aren't separated into a
    /// distinct histogram (yet).
    pub const CONFIRMED_READ_WAIT_SECONDS: &str = "tdb_confirmed_read_wait_seconds";

    /// Histogram of `Engine::open` durations, in seconds. Covers
    /// segment scan, WAL replay, and derived state rebuild. Useful
    /// for noticing replay regressions across releases.
    pub const ENGINE_OPEN_SECONDS: &str = "tdb_engine_open_seconds";

    /// Histogram of `Engine::snapshot` wall-clock durations, in
    /// seconds. Useful for sizing maintenance windows.
    pub const SNAPSHOT_WRITE_SECONDS: &str = "tdb_snapshot_write_seconds";
}
