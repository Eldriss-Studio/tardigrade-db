"""Smoke test for the generic multi-agent demo.

The demo at ``examples/multi_agent_demo.py`` exercises every
foundation primitive end-to-end: the public Python facade, owner
registry, snapshot/restore via the checkpoint repository, the
synchronous sweep trigger, the encode_query convenience, and the
builder pattern. If any of those regresses the demo breaks and
this test fails — the canary the foundation phase asked for.

Markers below match the print lines in the demo. Renaming a line
there means updating it here too.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
        # Each marker corresponds to one primitive the demo
        # exercises. A silent removal of any of those sections
        # would otherwise pass the test.
        for marker in (
            "owners:",
            "checkpoint:",
            "sweep_now",
            "encode_query",
            "builder:",
            "restored ok",
        ):
            assert marker in result.stdout, (
                f"demo output missing expected marker {marker!r}\n"
                f"stdout:\n{result.stdout}"
            )
