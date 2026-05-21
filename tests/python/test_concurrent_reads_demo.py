"""Smoke test for the concurrent-reads demo.

The demo at ``examples/concurrent_reads_demo.py`` runs N reader
threads against a shared engine and asserts every thread retrieves
the same cell set. It also reports single-thread and N-thread
throughput so the engine's read-path concurrency work can be
measured by re-running the same script before and after.

This test verifies the demo exits clean and that each of its
narrative sections is present. A silent removal of any section
would otherwise pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEMO_SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "concurrent_reads_demo.py"
)


class TestConcurrentReadsDemo:
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
            timeout=120,
            env=env,
            check=False,
        )
        assert result.returncode == 0, (
            f"demo exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        for marker in (
            "seeded:",
            "single-thread:",
            "8-thread:",
            "consistency:",
            "speedup vs single-thread:",
        ):
            assert marker in result.stdout, (
                f"demo did not print expected marker '{marker}'\n"
                f"stdout:\n{result.stdout}"
            )
