"""Acceptance tests for ``tardigrade_vllm._vllm_bridge``.

The bridge is the Anti-Corruption Layer (Evans, DDD) between vLLM's
internal contracts (``KVCacheBlocks``, the ``Request`` duck-type) and
the connector's domain types. Each test pins one shape the bridge
promises to absorb, so a future vLLM bump touches the bridge module
alone and the connector keeps compiling.

These tests deliberately do NOT import vLLM — the bridge accepts
duck-typed inputs (``hasattr(blocks, "get_block_ids")``), so a fake
object is sufficient to prove the contract.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tardigrade_vllm._vllm_bridge import KVCacheBlocksAdapter, RequestAdapter


class _FakeKVCacheBlocks:
    """Duck-typed stand-in for ``vllm.v1.core.kv_cache_manager.KVCacheBlocks``.

    Returns ``tuple[list[int], ...]`` where the outer tuple is per
    kv-cache group, matching the real ``get_block_ids()`` contract.
    """

    def __init__(self, groups):
        self._groups = tuple(tuple(g) for g in groups)

    def get_block_ids(self):
        return self._groups


class _FakeRequest:
    """Duck-typed stand-in for vLLM's ``Request`` shape."""

    def __init__(self, *, request_id=None, prompt_token_ids=None):
        if request_id is not None:
            self.request_id = request_id
        if prompt_token_ids is not None:
            self.prompt_token_ids = prompt_token_ids


# ---------- KVCacheBlocksAdapter ----------


def test_kv_cache_blocks_adapter_first_block_id_from_vllm_0_19_dataclass():
    """vLLM 0.19+ ``KVCacheBlocks.get_block_ids()`` returns the per-group
    tuple; ``first_block_id`` extracts group 0's first block."""
    blocks = _FakeKVCacheBlocks([[42, 43, 44], [99]])
    assert KVCacheBlocksAdapter.first_block_id(blocks) == 42


def test_kv_cache_blocks_adapter_first_block_id_from_legacy_list_of_lists():
    """Older vLLM passed ``tuple[list[int], ...]`` directly (no
    dataclass). The adapter accepts both shapes through one entry point."""
    legacy = ([7, 8, 9],)
    assert KVCacheBlocksAdapter.first_block_id(legacy) == 7


def test_kv_cache_blocks_adapter_first_block_id_from_bare_list():
    """The oldest vLLM contract handed back a bare ``list[int]``."""
    bare = [13, 14, 15]
    assert KVCacheBlocksAdapter.first_block_id(bare) == 13


def test_kv_cache_blocks_adapter_first_block_id_returns_none_on_empty_allocation():
    """Precomputed empty ``KVCacheBlocks`` (e.g. cache-hit-only request)
    has no allocation — bridge must return ``None`` rather than raise."""
    empty = _FakeKVCacheBlocks([[]])
    assert KVCacheBlocksAdapter.first_block_id(empty) is None


def test_kv_cache_blocks_adapter_block_ids_returns_list_int_for_vllm_0_19():
    """``block_ids`` materializes group 0 as ``list[int]`` so downstream
    callers can ``list(...)`` without knowing about ``KVCacheBlocks``."""
    blocks = _FakeKVCacheBlocks([[1, 2, 3], [99]])
    result = KVCacheBlocksAdapter.block_ids(blocks)
    assert result == [1, 2, 3]
    assert all(isinstance(b, int) for b in result)


def test_kv_cache_blocks_adapter_block_ids_returns_empty_list_on_empty_allocation():
    """Empty allocation → empty list (NOT ``None``). Caller can iterate
    unconditionally."""
    empty = _FakeKVCacheBlocks([[]])
    assert KVCacheBlocksAdapter.block_ids(empty) == []


# ---------- RequestAdapter ----------


def test_request_adapter_request_id_returns_attribute_when_present():
    req = _FakeRequest(request_id="req-abc-123")
    assert RequestAdapter.request_id(req) == "req-abc-123"


def test_request_adapter_request_id_falls_back_to_object_identity():
    """When ``request_id`` is missing (older vLLM, or test fixtures),
    fall back to ``id(request)`` so we still get a stable key per request
    within the process. Matches the historical
    ``getattr(req, "request_id", id(req))`` pattern."""
    req = _FakeRequest()  # no request_id attribute
    assert RequestAdapter.request_id(req) == id(req)


def test_request_adapter_prompt_token_ids_returns_list_when_present():
    req = _FakeRequest(prompt_token_ids=[10, 20, 30])
    assert RequestAdapter.prompt_token_ids(req) == [10, 20, 30]


def test_request_adapter_prompt_token_ids_returns_none_when_missing():
    """Missing attribute → ``None``, not raise. Callers gate on this."""
    req = _FakeRequest()
    assert RequestAdapter.prompt_token_ids(req) is None
