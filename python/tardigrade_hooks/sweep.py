"""GovernanceSweepThread — DEPRECATED.

Use the Rust-native MaintenanceWorker instead::

    engine.start_maintenance(
        sweep_interval_secs=3600,
        compaction_interval_secs=21600,
        eviction_threshold=15.0,
    )

The Rust implementation runs governance sweep (decay + eviction)
AND segment compaction in a single background thread with no GIL
contention. This Python thread only runs decay.
"""

import threading
import warnings


class GovernanceSweepThread:
    """Background thread that periodically applies AKL governance decay.

    Active Object pattern: owns a timer loop, invokes operations on a passive Engine.
    The thread is a daemon — it will not prevent process exit.

    Args:
        engine: A `tardigrade_db.Engine` instance.
        interval_secs: Seconds between sweep cycles (default: 3600 = 1 hour).
        hours_per_tick: How many hours of simulated time per tick (default: 1.0).
    """

    def __init__(self, engine, interval_secs=3600, hours_per_tick=1.0):
        warnings.warn(
            "GovernanceSweepThread is deprecated. "
            "Use engine.start_maintenance() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.engine = engine
        self.interval = interval_secs
        self.hours_per_tick = hours_per_tick
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tdb-governance-sweep")
        self._tick_count = 0

    def start(self):
        """Start the background sweep thread."""
        self._stop_event.clear()
        self._thread.start()

    def stop(self, timeout=5.0):
        """Stop the sweep thread and wait for it to finish.

        Args:
            timeout: Max seconds to wait for the thread to join.
        """
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    @property
    def is_running(self):
        """Whether the sweep thread is currently alive."""
        return self._thread.is_alive()

    @property
    def tick_count(self):
        """Number of sweep cycles completed."""
        return self._tick_count

    def _run(self):
        """Timer loop: sleep for interval, then apply governance decay."""
        while not self._stop_event.wait(self.interval):
            days = self.hours_per_tick / 24.0
            self.engine.advance_days(days)
            self._tick_count += 1
