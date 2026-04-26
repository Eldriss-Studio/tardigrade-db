# ATDD tests for start_load_kv() tensor copy logic.
#
# CPU-only: uses mock forward_context with numpy arrays standing in
# for GPU tensors. Tests the Adapter logic (flat → paged blocks → slots)
# independently of vLLM runtime.

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_vllm.format import flat_to_blocks


# -- Helpers ------------------------------------------------------------------

def _make_engine_with_pack(kv_dim=8, seq_len=4, num_layers=2):
    """Create an engine with one synthetic pack for testing load path."""
    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)

    key = np.ones(kv_dim, dtype=np.float32)
    layer_payloads = []
    for layer_idx in range(num_layers):
        # Deterministic data so we can verify what gets copied
        payload = np.arange(
            2 * seq_len * kv_dim, dtype=np.float32
        ) * (layer_idx + 1) * 0.01
        layer_payloads.append((layer_idx, payload))

    pack_id = engine.mem_write_pack(1, key, layer_payloads, 80.0)
    pack = engine.load_pack_by_id(pack_id)
    return engine, pack, kv_dim, seq_len, num_layers


def _make_mock_forward_context(num_layers, num_total_blocks, block_size,
                               num_kv_heads, head_dim):
    """Build a mock forward_context with numpy KV caches.

    Each layer's cache is a tuple (k_cache, v_cache), each shaped
    (num_total_blocks, block_size, num_kv_heads, head_dim).
    Initialized to zeros so we can detect writes.
    """
    ctx = MagicMock()
    shape = (num_total_blocks, block_size, num_kv_heads, head_dim)
    kv_caches = []
    for _ in range(num_layers):
        k_cache = np.zeros(shape, dtype=np.float32)
        v_cache = np.zeros(shape, dtype=np.float32)
        kv_caches.append((k_cache, v_cache))
    ctx.kv_caches = kv_caches
    return ctx


def _invoke_load_kv(connector_state, forward_context, num_kv_heads,
                    head_dim, block_size):
    """Execute the start_load_kv logic against mock objects.

    This mirrors the expected implementation: iterate _load_packs,
    convert flat → blocks, copy into allocated slots.

    When start_load_kv is implemented in the connector, this function
    should be replaced by calling the actual method.
    """
    from tardigrade_vllm.format import flat_to_blocks

    load_packs = connector_state["load_packs"]
    load_meta = connector_state["load_meta"]

    for req_id, load_info in list(load_packs.items()):
        meta = load_meta.get(req_id)
        if meta is None:
            continue

        pack = load_info["pack"]
        seq_len = load_info["seq_len"]
        block_ids = meta["block_ids"]

        for layer_entry in pack["layers"]:
            layer_idx = layer_entry["layer_idx"]
            layer_data = np.array(layer_entry["data"], dtype=np.float32)

            k_blocks, v_blocks = flat_to_blocks(
                layer_data, num_kv_heads, head_dim, block_size
            )

            kv_cache = forward_context.kv_caches[layer_idx]
            k_cache, v_cache = kv_cache[0], kv_cache[1]

            num_blocks = k_blocks.shape[0]
            for i in range(min(num_blocks, len(block_ids))):
                k_cache[block_ids[i]] = k_blocks[i]
                v_cache[block_ids[i]] = v_blocks[i]

        del load_packs[req_id]
        load_meta.pop(req_id, None)


# -- Tests --------------------------------------------------------------------

def test_load_kv_copies_blocks_to_cache_slots():
    """GIVEN a mock forward_context with numpy KV caches and a matched pack,
    WHEN start_load_kv logic is executed,
    THEN the correct block slots contain the converted pack data."""
    num_kv_heads = 2
    head_dim = 4
    kv_dim = num_kv_heads * head_dim
    block_size = 4
    seq_len = 4
    num_layers = 2
    num_total_blocks = 8

    engine, pack, _, _, _ = _make_engine_with_pack(
        kv_dim=kv_dim, seq_len=seq_len, num_layers=num_layers
    )

    # Allocate blocks 2 and 3 for this request (seq_len=4, block_size=4 → 1 block)
    block_ids = [2]

    ctx = _make_mock_forward_context(
        num_layers, num_total_blocks, block_size, num_kv_heads, head_dim
    )

    state = {
        "load_packs": {"req_1": {"pack": pack, "seq_len": seq_len}},
        "load_meta": {"req_1": {"block_ids": block_ids, "num_tokens": seq_len}},
    }

    _invoke_load_kv(state, ctx, num_kv_heads, head_dim, block_size)

    # Block 2 should be non-zero in both layers
    for layer_idx in range(num_layers):
        k_cache, v_cache = ctx.kv_caches[layer_idx]
        assert not np.allclose(k_cache[2], 0.0), (
            f"Layer {layer_idx}: block 2 K cache should be non-zero after load"
        )
        assert not np.allclose(v_cache[2], 0.0), (
            f"Layer {layer_idx}: block 2 V cache should be non-zero after load"
        )
        # Block 0 should still be zeros (untouched)
        assert np.allclose(k_cache[0], 0.0)


def test_load_kv_skips_requests_without_meta():
    """GIVEN a request in _load_packs but NOT in _load_meta,
    WHEN start_load_kv logic is executed,
    THEN no copy occurs and no error is raised."""
    num_kv_heads = 2
    head_dim = 4
    block_size = 4
    num_layers = 1
    num_total_blocks = 4

    engine, pack, _, seq_len, _ = _make_engine_with_pack(
        kv_dim=num_kv_heads * head_dim, seq_len=4, num_layers=1
    )

    ctx = _make_mock_forward_context(
        num_layers, num_total_blocks, block_size, num_kv_heads, head_dim
    )

    state = {
        "load_packs": {"req_orphan": {"pack": pack, "seq_len": seq_len}},
        "load_meta": {},  # No meta for this request
    }

    # Should not raise
    _invoke_load_kv(state, ctx, num_kv_heads, head_dim, block_size)

    # All blocks should remain zeros
    k_cache, v_cache = ctx.kv_caches[0]
    assert np.allclose(k_cache, 0.0)


def test_load_kv_handles_partial_blocks():
    """GIVEN a pack with seq_len=5 and block_size=4 (not evenly divisible),
    WHEN start_load_kv converts and copies,
    THEN partial blocks are padded and both blocks are written."""
    num_kv_heads = 2
    head_dim = 4
    kv_dim = num_kv_heads * head_dim
    block_size = 4
    seq_len = 5  # Needs 2 blocks (8 slots, 3 padded)
    num_layers = 1
    num_total_blocks = 8

    engine, pack, _, _, _ = _make_engine_with_pack(
        kv_dim=kv_dim, seq_len=seq_len, num_layers=num_layers
    )

    # 2 blocks needed for seq_len=5, block_size=4
    block_ids = [1, 3]

    ctx = _make_mock_forward_context(
        num_layers, num_total_blocks, block_size, num_kv_heads, head_dim
    )

    state = {
        "load_packs": {"req_partial": {"pack": pack, "seq_len": seq_len}},
        "load_meta": {"req_partial": {"block_ids": block_ids, "num_tokens": seq_len}},
    }

    _invoke_load_kv(state, ctx, num_kv_heads, head_dim, block_size)

    k_cache, v_cache = ctx.kv_caches[0]
    # Block 1 should have data (first 4 tokens)
    assert not np.allclose(k_cache[1], 0.0), "Block 1 should have data"
    # Block 3 should have partial data (1 real token + 3 padded)
    # At least the first slot should be non-zero
    assert not np.allclose(k_cache[3][0], 0.0), "Block 3 slot 0 should have data"
    # Block 0 untouched
    assert np.allclose(k_cache[0], 0.0)


def test_load_kv_cleans_up_state():
    """GIVEN a request that was loaded,
    WHEN start_load_kv completes,
    THEN the request is removed from load_packs and load_meta."""
    num_kv_heads = 2
    head_dim = 4
    kv_dim = num_kv_heads * head_dim
    block_size = 4

    engine, pack, _, seq_len, _ = _make_engine_with_pack(
        kv_dim=kv_dim, seq_len=4, num_layers=1
    )

    ctx = _make_mock_forward_context(1, 4, block_size, num_kv_heads, head_dim)

    state = {
        "load_packs": {"req_cleanup": {"pack": pack, "seq_len": seq_len}},
        "load_meta": {"req_cleanup": {"block_ids": [0], "num_tokens": seq_len}},
    }

    _invoke_load_kv(state, ctx, num_kv_heads, head_dim, block_size)

    assert "req_cleanup" not in state["load_packs"]
    assert "req_cleanup" not in state["load_meta"]


# -- Connector __init__ signature tests (Gap 6 / vLLM 0.19+) ------------------

def test_connector_init_accepts_kv_cache_config_kwarg():
    """GIVEN vLLM 0.19+ which inspects connector signatures,
    WHEN TardigradeConnector.__init__ is introspected,
    THEN it accepts (vllm_config, kv_cache_config, role) — three required
    positional args — to avoid the deprecation warning at vllm.factory:126."""
    import inspect

    vllm = pytest.importorskip("vllm", reason="vLLM not installed")
    from tardigrade_vllm.connector import TardigradeConnector

    sig = inspect.signature(TardigradeConnector.__init__)
    params = list(sig.parameters.values())
    # First param is self; vLLM counts the rest
    non_self = [p for p in params if p.name != "self"]
    names = [p.name for p in non_self]

    assert "kv_cache_config" in names, (
        f"vLLM 0.19+ requires kv_cache_config in __init__ signature, got {names}"
    )
    # Order matters: vllm_config, kv_cache_config, role
    assert names.index("vllm_config") < names.index("kv_cache_config") < names.index("role"), (
        f"Expected order (vllm_config, kv_cache_config, role), got {names}"
    )


def test_connector_init_back_compat_with_two_arg_call():
    """GIVEN older callers that may still pass (vllm_config, role),
    WHEN TardigradeConnector is instantiated with two positional args,
    THEN it does not crash on signature mismatch (kv_cache_config defaults to None).

    This protects against breaking any test/script that bypasses vLLM's factory
    and constructs the connector directly with the old signature.
    """
    import inspect

    vllm = pytest.importorskip("vllm", reason="vLLM not installed")
    from tardigrade_vllm.connector import TardigradeConnector

    sig = inspect.signature(TardigradeConnector.__init__)
    kv_cfg_param = sig.parameters.get("kv_cache_config")
    assert kv_cfg_param is not None
    # Must have a default — this is what enables back-compat with 2-arg callers
    assert kv_cfg_param.default is not inspect.Parameter.empty, (
        "kv_cache_config must have a default to preserve back-compat with "
        "callers using the old (vllm_config, role) signature"
    )
