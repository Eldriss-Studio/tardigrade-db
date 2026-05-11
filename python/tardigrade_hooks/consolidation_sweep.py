"""Background consolidation sweep for multi-view memory creation.

Active Object pattern (matches ``GovernanceSweepThread``): owns a timer
loop in a daemon thread, periodically runs ``MemoryConsolidator`` on
all eligible packs.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from .consolidator import MemoryConsolidator

if TYPE_CHECKING:
    from .consolidator import ConsolidationPolicy
    from .view_generator import ViewGenerator


class ConsolidationSweepThread:
    """Daemon thread that periodically consolidates eligible memories.

    Args:
        engine: A ``tardigrade_db.Engine`` instance.
        owner: Agent/user owner ID to consolidate for.
        interval_secs: Seconds between sweep cycles.
        view_generator: Optional custom ``ViewGenerator``.
        policy: Optional custom ``ConsolidationPolicy``.
    """

    def __init__(
        self,
        engine,
        *,
        owner: int = 1,
        interval_secs: float = 3600.0,
        view_generator: ViewGenerator | None = None,
        policy: ConsolidationPolicy | None = None,
    ):
        self._consolidator = MemoryConsolidator(
            engine,
            owner=owner,
            view_generator=view_generator,
            policy=policy,
        )
        self._owner = owner
        self._interval = interval_secs
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="tdb-consolidation-sweep",
        )
        self._packs_consolidated = 0
        self._views_attached = 0
        self._last_run_epoch: float = 0.0

    def start(self):
        """Start the background consolidation thread."""
        self._stop_event.clear()
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    @property
    def status(self) -> dict:
        return {
            "packs_consolidated": self._packs_consolidated,
            "views_attached": self._views_attached,
            "last_run_epoch": self._last_run_epoch,
        }

    def _run(self):
        while not self._stop_event.wait(self._interval):
            result = self._consolidator.consolidate_all(owner=self._owner)
            self._packs_consolidated += len(result)
            self._views_attached += sum(result.values())
            self._last_run_epoch = time.time()
