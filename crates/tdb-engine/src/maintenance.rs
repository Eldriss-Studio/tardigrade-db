//! Background maintenance worker (Active Object pattern).
//!
//! Periodically runs governance sweep (decay + eviction) and segment
//! compaction in a background thread. The worker shares an
//! `Arc<Mutex<Engine>>` with the foreground — lock is held only during
//! operations, never during sleep.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::engine::Engine;

const DEFAULT_SWEEP_INTERVAL_SECS: u64 = 3600;
const DEFAULT_COMPACTION_INTERVAL_SECS: u64 = 21600;
/// Default importance threshold below which Draft packs are evicted.
pub const DEFAULT_EVICTION_THRESHOLD: f32 = 15.0;
const DEFAULT_HOURS_PER_TICK: f32 = 1.0;
const SHUTDOWN_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Configuration for the background maintenance worker.
#[derive(Debug, Clone)]
pub struct MaintenanceConfig {
    pub sweep_interval: Duration,
    pub compaction_interval: Duration,
    pub eviction_threshold: f32,
    pub hours_per_tick: f32,
    pub enabled: bool,
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            sweep_interval: Duration::from_secs(DEFAULT_SWEEP_INTERVAL_SECS),
            compaction_interval: Duration::from_secs(DEFAULT_COMPACTION_INTERVAL_SECS),
            eviction_threshold: DEFAULT_EVICTION_THRESHOLD,
            hours_per_tick: DEFAULT_HOURS_PER_TICK,
            enabled: true,
        }
    }
}

/// Observable statistics from the maintenance worker.
#[derive(Debug, Clone, Default)]
pub struct MaintenanceStatus {
    pub sweep_count: u64,
    pub compaction_count: u64,
    pub total_packs_evicted: u64,
    pub total_bytes_reclaimed: u64,
    pub last_sweep_epoch_secs: Option<u64>,
    pub last_compaction_epoch_secs: Option<u64>,
}

/// Background maintenance worker that periodically runs governance
/// sweep and segment compaction.
#[derive(Debug)]
pub struct MaintenanceWorker {
    config: MaintenanceConfig,
    status: Arc<Mutex<MaintenanceStatus>>,
    stop_flag: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl MaintenanceWorker {
    pub fn new(config: MaintenanceConfig) -> Self {
        Self {
            config,
            status: Arc::new(Mutex::new(MaintenanceStatus::default())),
            stop_flag: Arc::new(AtomicBool::new(false)),
            handle: None,
        }
    }

    pub fn start(&mut self, engine: Arc<Mutex<Engine>>) {
        if !self.config.enabled || self.handle.is_some() {
            return;
        }

        let stop_flag = Arc::clone(&self.stop_flag);
        let status = Arc::clone(&self.status);
        let config = self.config.clone();

        let handle = thread::Builder::new()
            .name("tdb-maintenance".into())
            .spawn(move || run_loop(engine, config, status, stop_flag))
            .expect("failed to spawn maintenance thread");

        self.handle = Some(handle);
    }

    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.stop_flag.store(false, Ordering::Release);
    }

    pub fn status(&self) -> MaintenanceStatus {
        self.status.lock().map_or_else(|_| MaintenanceStatus::default(), |s| s.clone())
    }

    pub fn is_running(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }
}

impl Drop for MaintenanceWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

fn now_epoch_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

#[expect(clippy::needless_pass_by_value)]
fn run_loop(
    engine: Arc<Mutex<Engine>>,
    config: MaintenanceConfig,
    status: Arc<Mutex<MaintenanceStatus>>,
    stop_flag: Arc<AtomicBool>,
) {
    let mut last_sweep = Instant::now();
    let mut last_compaction = Instant::now();

    loop {
        let time_to_sweep = config.sweep_interval.saturating_sub(last_sweep.elapsed());
        let time_to_compact = config.compaction_interval.saturating_sub(last_compaction.elapsed());
        let sleep_duration = time_to_sweep.min(time_to_compact).min(SHUTDOWN_POLL_INTERVAL);

        thread::sleep(sleep_duration);

        if stop_flag.load(Ordering::Acquire) {
            return;
        }

        if last_sweep.elapsed() >= config.sweep_interval
            && let Ok(mut eng) = engine.lock()
        {
            let days = config.hours_per_tick / 24.0;
            eng.advance_days(days);
            let evicted = eng.evict_draft_packs(config.eviction_threshold, None).unwrap_or(0);
            drop(eng);

            if let Ok(mut s) = status.lock() {
                s.sweep_count += 1;
                s.total_packs_evicted += evicted as u64;
                s.last_sweep_epoch_secs = Some(now_epoch_secs());
            }
            last_sweep = Instant::now();
        }

        if last_compaction.elapsed() >= config.compaction_interval
            && let Ok(mut eng) = engine.lock()
        {
            if let Ok(result) = eng.compact() {
                drop(eng);
                if let Ok(mut s) = status.lock() {
                    s.compaction_count += 1;
                    s.total_bytes_reclaimed += result.bytes_reclaimed;
                    s.last_compaction_epoch_secs = Some(now_epoch_secs());
                }
            }
            last_compaction = Instant::now();
        }
    }
}
