"""Smoke test for the interactive-fiction NPC memory demo.

The demo at ``examples/if_demo.py`` shows the subjectivity pattern:
two NPCs observe the same scripted player events but store
different memories conditioned on persona. This test runs the demo
end-to-end and asserts the printed output reflects the per-NPC
divergence.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEMO_SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "if_demo.py"
)


class TestInteractiveFictionDemo:
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
        # The demo prints one summary line per NPC after the
        # scripted events. The line names the NPC and the count of
        # memories that NPC stored.
        for marker in (
            "blacksmith recalls",
            "tavern keeper recalls",
            "subjective divergence:",
        ):
            assert marker in result.stdout, (
                f"demo output missing expected marker {marker!r}\n"
                f"stdout:\n{result.stdout}"
            )
