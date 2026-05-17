"""ATDD: M2.2 — Node.js consumer round-trip.

Runs ``examples/nodejs_consumer/index.mjs`` against a real uvicorn
server (in a background thread) and asserts the script exits 0.
The script itself asserts all the round-trip invariants — this
test is just the orchestration shell.

Skips cleanly when:

- ``node`` isn't on PATH (CI lint hosts without Node).
- uvicorn isn't installed (CPU-only Python wheel).

The point is to prove the HTTP bridge is *actually* consumable
from outside Python — not to gate Python-only environments on
having Node.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest


def _node_available() -> bool:
    return shutil.which("node") is not None


def _uvicorn_available() -> bool:
    try:
        import uvicorn  # noqa: F401
        return True
    except ImportError:
        return False


def _pick_port() -> int:
    """Bind-then-close to grab a free port. There's a tiny race
    between this returning and uvicorn binding, but for a local
    test it's acceptable."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_ready(port: int, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket() as s:
            try:
                s.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.05)
    raise RuntimeError(f"uvicorn never bound 127.0.0.1:{port} within {timeout_s}s")


@pytest.fixture
def bridge(tmp_path):
    if not _uvicorn_available():
        pytest.skip("uvicorn not installed")
    import uvicorn

    import tardigrade_db
    from tardigrade_http.server import create_app

    port = _pick_port()
    engine = tardigrade_db.Engine(str(tmp_path / "engine"))
    app = create_app(engine)

    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait_until_ready(port)
        yield port, tmp_path
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


@pytest.mark.skipif(not _node_available(), reason="node not installed")
class TestNodeJsConsumerRoundTrip:
    def test_example_script_runs_to_completion(self, bridge):
        port, tmp_path = bridge
        script = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "nodejs_consumer"
            / "index.mjs"
        )
        assert script.is_file(), f"missing demo script at {script}"

        env = {
            "PATH": os.environ.get("PATH", ""),
            "BRIDGE_URL": f"http://127.0.0.1:{port}",
            "SNAPSHOT_PATH": str(tmp_path / "demo-snap.tar"),
            "RESTORE_DIR": str(tmp_path / "demo-restored"),
        }

        result = subprocess.run(
            ["node", str(script)],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            check=False,
        )

        assert result.returncode == 0, (
            f"node consumer exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        # Sanity-check at least one expected log line landed — guards
        # against a future regression where the script silently no-ops.
        assert "owners = [1] ✓" in result.stdout, (
            f"missing final-check line in stdout:\n{result.stdout}"
        )
