"""ATDD: M3.8a — Generic multi-agent demo (foundation acceptance gate).

The demo at ``examples/multi_agent_demo.py`` exercises every
foundation milestone end-to-end:

- **M0** — uses the public Python facade, not engine internals.
- **M1.1** — three agents (distinct owner ids), enumerated via
  ``Engine.list_owners``.
- **M1.2** — snapshots and restores the engine.
- **M3.1** — uses ``CheckpointRepository`` to label saves.
- **M3.2** — calls ``Engine.sweep_now`` to force tier transitions.
- **M3.3** — uses ``TardigradeClient.encode_query``.
- **M3.4** — constructs clients via ``TardigradeClient.builder``.

If any of those slices regresses, this test fails and the demo
breaks. It's the canary the foundation-completion plan asked for.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

DEMO_SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "multi_agent_demo.py"
)


class TestMultiAgentDemo:
    def test_script_exists(self):
        assert DEMO_SCRIPT.is_file(), f"missing demo script at {DEMO_SCRIPT}"

    def test_script_runs_to_completion(self, tmp_path):
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1",
            "TARDIGRADE_DEMO_DIR": str(tmp_path),
        }
        result = subprocess.run(
            [sys.executable, str(DEMO_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            check=False,
        )
        assert result.returncode == 0, (
            f"demo exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        # Sanity-check the milestone-coverage marker the demo
        # prints — if anyone removes a milestone-touching section,
        # this fails before silent-regression hides it.
        for marker in (
            "M1.1 owners:",
            "M3.1 checkpoint:",
            "M3.2 sweep_now",
            "M3.3 encode_query",
            "M3.4 builder",
            "restored ok",
        ):
            assert marker in result.stdout, (
                f"demo output missing milestone marker {marker!r}\n"
                f"stdout:\n{result.stdout}"
            )
