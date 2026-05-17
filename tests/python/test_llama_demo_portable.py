"""Portability checks for examples/llama_memory_test.py.

The demo loads a local GGUF model via llama-cpp-python. Neither the
model path nor the package can be assumed present on a fresh
checkout — they're consumer-provided. The script must fail-fast
with a useful message instead of crashing on a stale macOS path
or a missing import.

These tests just exercise the precondition surface — they don't
download a 2 GB model.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

DEMO_SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "llama_memory_test.py"
)


def _run_demo(args, env_extra=None):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1",
    }
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(DEMO_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )


class TestLlamaDemoPortability:
    def test_script_imports_without_llama_cpp(self):
        """No top-level ``from llama_cpp import Llama``.

        The import has to be deferred into a function so the
        ``info`` / ``clear`` / usage paths work without the package
        installed. (Real `store` / `query` of course still need
        it.)
        """
        result = _run_demo([])
        # No args → usage line. Must NOT die with ImportError.
        assert result.returncode != 0, "expected non-zero from no-arg usage"
        combined = result.stdout + result.stderr
        assert "ModuleNotFoundError" not in combined, (
            f"top-level llama_cpp import broke the script:\n{combined}"
        )
        assert "Usage:" in combined or "usage" in combined.lower()

    def test_store_without_env_var_fails_with_clear_message(self, tmp_path):
        """If neither ``TARDIGRADE_LLAMA_GGUF`` nor the discovery
        fallback finds a model, the script tells the user what to
        do — it doesn't crash with a FileNotFoundError on the
        original author's macOS path."""
        # Clear any prior cached DB so the test doesn't see one.
        env_extra = {"TARDIGRADE_LLAMA_GGUF": ""}
        result = _run_demo(["store", "anything"], env_extra=env_extra)
        assert result.returncode != 0
        combined = (result.stdout + result.stderr).lower()
        # Must mention the env var name so the user can fix it.
        assert "tardigrade_llama_gguf" in combined, (
            f"error message should name the env var:\n{result.stdout}\n{result.stderr}"
        )

    def test_store_with_nonexistent_path_fails_with_clear_message(self):
        env_extra = {"TARDIGRADE_LLAMA_GGUF": "/no/such/path.gguf"}
        result = _run_demo(["store", "anything"], env_extra=env_extra)
        assert result.returncode != 0
        combined = (result.stdout + result.stderr).lower()
        assert "/no/such/path.gguf" in combined or "not found" in combined, (
            f"error should reference the bad path:\n{result.stdout}\n{result.stderr}"
        )

    def test_info_works_without_llama_cpp_or_model(self):
        """``info`` doesn't need the LLM — it only reads the
        engine state. It should run fine even without
        ``llama_cpp`` installed or a configured model."""
        result = _run_demo(["info"])
        # info on a non-existent DB just prints "no database found".
        # The point: it doesn't blow up trying to import llama_cpp
        # or resolve the model path.
        combined = result.stdout + result.stderr
        assert "ModuleNotFoundError" not in combined, (
            f"info path tripped llama_cpp import:\n{combined}"
        )
