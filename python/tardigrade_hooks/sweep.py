"""Background governance sweep — Scheduler (Active Object) pattern.

A daemon thread that periodically calls `engine.advance_days()` to apply
AKL decay, evaluate tier transitions, and prepare stale cells for eviction.

The Engine itself remains single-threaded and passive — this thread is the
scheduling concern, not the business logic. Same pattern as SQLite
(single-writer, application-level concurrency).

Usage::

    from tardigrade_hooks.sweep import GovernanceSweepThread

    engine = tardigrade_db.Engine("/tmp/tdb")
    sweep = GovernanceSweepThread(engine, interval_secs=3600)
    sweep.start()
    # ... use engine normally ...
    sweep.stop()  # or let it die with the process (daemon thread)
"""

import threading


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
