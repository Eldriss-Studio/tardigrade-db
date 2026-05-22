"""ATs for the `_do_gpu_cleanup` helper defined in conftest.py.

Pins two contracts:

1. The helper exists and does the right things (gc.collect first, then
   torch.cuda.empty_cache, with a guard against CUDA / torch absence).
2. The helper is NOT wired as a pytest autouse fixture. An earlier
   implementation tried that and over-cleaned — clearing the torch
   caching allocator between tests within a module broke tests sharing
   a module-scoped vLLM fixture. The right boundary is module
   teardown (per-fixture `try / finally`), not function teardown.

Background: test_pack_persists_across_vllm_restart passed in isolation
but failed in the full GPU suite because the test_vllm_integration.py::
llm module-scoped fixture had the literal comment "Cleanup handled by
garbage collection" — and GC doesn't release CUDA memory or vLLM
process-group state. The fix lives inside the fixture itself (its
`finally` block calls `_do_gpu_cleanup`), not in a cross-cutting
autouse hook.

CPU-runnable: monkeypatches torch.cuda.empty_cache and gc.collect so
the test does not require a real CUDA device.
"""

from __future__ import annotations

import gc
import importlib.util
from pathlib import Path

import pytest


def _load_conftest_module():
    """Import the conftest under test as a regular module so we can grab
    the cleanup fixture function out of it. The fixture is registered
    by pytest at collection time; for the AT we want the raw function."""
    spec = importlib.util.spec_from_file_location(
        "tests_python_conftest_under_test",
        Path(__file__).parent / "conftest.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conftest_exports_gpu_cleanup_helper():
    """Names the contract: conftest.py must expose `_do_gpu_cleanup` —
    the plain cleanup function module-scoped fixtures call in their
    `try / finally` teardown. Future renames break this AT loudly."""
    conftest = _load_conftest_module()
    assert hasattr(conftest, "_do_gpu_cleanup"), (
        "conftest.py must export `_do_gpu_cleanup` — the pure cleanup "
        "function module-scoped GPU fixtures invoke in their teardown. "
        "See feedback-gpu-fixtures-need-explicit-cleanup."
    )


def test_conftest_does_NOT_install_autouse_cuda_cleanup():
    """An earlier attempt wired the cleanup as a pytest autouse fixture
    that ran after EVERY test. That over-cleaned: it cleared the torch
    caching allocator between tests within a module, breaking tests that
    shared a module-scoped vLLM fixture. The cleanup must run at module
    teardown only, via the per-fixture `finally` block.

    This AT pins the negative: no autouse fixture invoking
    `_do_gpu_cleanup` may exist in conftest. If a future contributor
    re-adds one, this catches it."""
    conftest = _load_conftest_module()
    # Walk conftest top-level objects looking for a fixture-decorated
    # function whose body calls _do_gpu_cleanup. Pytest marks fixtures
    # with a `_pytestfixturefunction` attribute carrying the scope info.
    for name in dir(conftest):
        if name.startswith("__"):
            continue
        obj = getattr(conftest, name)
        marker = getattr(obj, "_pytestfixturefunction", None)
        if marker is None:
            continue
        # If a fixture exists, it must NOT be autouse (we're allowed to
        # have fixtures that opt-in for cleanup, but not unconditional).
        autouse = getattr(marker, "autouse", False)
        assert not autouse, (
            f"conftest fixture {name!r} is autouse — that re-introduces "
            "the over-aggressive cleanup that breaks intra-module tests. "
            "Module-scoped fixtures should call `_do_gpu_cleanup` in their "
            "own `try/finally` instead."
        )


def test_gpu_cleanup_invokes_cuda_empty_cache_and_gc_collect(monkeypatch):
    """GIVEN `_do_gpu_cleanup` runs on a host with CUDA available,
    WHEN it executes,
    THEN torch.cuda.empty_cache() is called AT LEAST once AND gc.collect()
    is called AT LEAST once. Order: gc.collect() first so Python objects
    holding CUDA tensor refs drop them BEFORE we try to free the cache —
    pinned in the assertion below."""
    pytest.importorskip("torch")
    import torch

    call_log: list[str] = []

    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: call_log.append("empty_cache")
    )
    monkeypatch.setattr(
        gc, "collect", lambda *args, **kwargs: call_log.append("collect") or 0
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    conftest = _load_conftest_module()
    conftest._do_gpu_cleanup()

    assert "collect" in call_log, (
        "gc.collect() must be called — Python GC must run before "
        "torch.cuda.empty_cache() to drop refs to CUDA tensors first."
    )
    assert "empty_cache" in call_log, (
        "torch.cuda.empty_cache() must be called — GC alone does not "
        "release blocks from the torch caching allocator."
    )
    assert call_log.index("collect") < call_log.index("empty_cache"), (
        f"gc.collect() must run BEFORE torch.cuda.empty_cache(); "
        f"call order was {call_log}"
    )


def test_gpu_cleanup_is_resilient_to_missing_cuda(monkeypatch):
    """GIVEN a host where torch is importable but CUDA is unavailable
    (CI without GPU, dev box without driver),
    WHEN _do_gpu_cleanup runs,
    THEN it must NOT raise AND must NOT invoke torch.cuda.empty_cache() —
    that call can throw on some torch builds when CUDA isn't initialised.
    gc.collect() is still expected to run unconditionally (it's free)."""
    pytest.importorskip("torch")
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    empty_cache_called: list[None] = []
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: empty_cache_called.append(None)
    )

    conftest = _load_conftest_module()
    conftest._do_gpu_cleanup()  # must complete without raising

    assert empty_cache_called == [], (
        "torch.cuda.empty_cache() must NOT be invoked when CUDA is "
        "unavailable — guard with torch.cuda.is_available() or equivalent."
    )


def test_gpu_cleanup_is_resilient_to_missing_torch(monkeypatch):
    """GIVEN a CPU-only venv where `import torch` raises ImportError,
    WHEN _do_gpu_cleanup runs,
    THEN it must NOT raise. The non-GPU `.venv-ft` (free-threaded Python)
    is exactly this case: it has the engine wheel but not torch."""
    import builtins

    real_import = builtins.__import__

    def _block_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_torch)

    conftest = _load_conftest_module()
    # Must complete without raising. gc.collect() still runs.
    conftest._do_gpu_cleanup()
