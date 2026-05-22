"""Helper module for GPU test fixtures — importable from any test file.

Lives outside conftest.py because pytest's conftest discovery doesn't
make its symbols importable via the regular `from conftest import …`
path. Module-scoped vLLM / torch / transformers fixtures import
[`do_gpu_cleanup`] from here in their `try / finally` teardown.

See feedback-gpu-fixtures-need-explicit-cleanup for why GC alone is
insufficient and why the cleanup boundary is module teardown
(per-fixture `finally`) rather than function teardown (autouse hook).
"""

from __future__ import annotations

import gc


def do_gpu_cleanup() -> None:
    """Pure cleanup helper — call from module-scoped fixture teardown.

    Module-scoped vLLM / torch / transformers fixtures must call this in
    their `try / yield / finally` block. Running it BETWEEN tests within
    a module (e.g. via an autouse fixture) is too aggressive — it clears
    the torch caching allocator that the still-live module fixture relies
    on, breaking subsequent tests in the same module. The right boundary
    is module teardown.

    Python's garbage collector does NOT release CUDA-allocated memory or
    vLLM process-group state on its own; only explicit
    `torch.cuda.empty_cache()` returns blocks to the torch caching
    allocator, and only `gc.collect()` actually triggers the dtors that
    drop refs to CUDA tensors. `gc.collect()` runs first so any Python
    objects holding CUDA tensor references actually drop them before we
    try to free the cache.

    Skipped when torch is not installed (CPU-only test runs in venvs
    that don't carry torch). Guarded by `torch.cuda.is_available()` so
    CI runners without a GPU don't hit `empty_cache()` errors.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
