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


# -- RequestSlotResolver tests (Step 1 / Strategy + Parameter Object) ---------

def _make_attn_metadata_mock(slot_mapping, query_start_loc):
    """Build a minimal mock that mimics vLLM 0.19 FlashAttentionMetadata."""
    import torch
    am = MagicMock()
    am.slot_mapping = torch.tensor(slot_mapping, dtype=torch.long)
    am.query_start_loc = torch.tensor(query_start_loc, dtype=torch.long)
    return am


def test_resolver_single_request_extracts_block_indices():
    """GIVEN attn_metadata with one request occupying slots 16..20 (block 1, block_size=16),
    WHEN resolve() runs,
    THEN it returns a single BatchSlice with block_indices=(1,) and slot_count=5."""
    pytest.importorskip("torch")
    from tardigrade_vllm.slot_resolver import RequestSlotResolver

    am = _make_attn_metadata_mock(
        slot_mapping=[16, 17, 18, 19, 20],
        query_start_loc=[0, 5],
    )
    slices = RequestSlotResolver().resolve(am, block_size=16)

    assert len(slices) == 1
    assert slices[0].batch_index == 0
    assert slices[0].block_indices == (1,)
    assert slices[0].slot_count == 5
    assert slices[0].first_slot == 16


def test_resolver_two_requests_in_one_step_yield_two_slices():
    """GIVEN slot_mapping with two requests interleaved (req0: 4 tokens in block 0,
    req1: 3 tokens in block 2),
    WHEN resolve() runs,
    THEN two distinct BatchSlices are returned with different blocks."""
    pytest.importorskip("torch")
    from tardigrade_vllm.slot_resolver import RequestSlotResolver

    am = _make_attn_metadata_mock(
        slot_mapping=[0, 1, 2, 3, 32, 33, 34],  # req0: block 0, req1: block 2
        query_start_loc=[0, 4, 7],
    )
    slices = RequestSlotResolver().resolve(am, block_size=16)

    assert len(slices) == 2
    assert slices[0].batch_index == 0
    assert slices[0].block_indices == (0,)
    assert slices[0].slot_count == 4
    assert slices[1].batch_index == 1
    assert slices[1].block_indices == (2,)
    assert slices[1].slot_count == 3


def test_resolver_request_spanning_two_blocks():
    """GIVEN one request with 20 tokens (more than block_size=16, so 2 blocks),
    WHEN resolve() runs,
    THEN block_indices contains both block IDs in sorted order."""
    pytest.importorskip("torch")
    from tardigrade_vllm.slot_resolver import RequestSlotResolver

    am = _make_attn_metadata_mock(
        slot_mapping=list(range(16, 36)),  # slots 16..35 → blocks 1 and 2
        query_start_loc=[0, 20],
    )
    slices = RequestSlotResolver().resolve(am, block_size=16)

    assert len(slices) == 1
    assert slices[0].block_indices == (1, 2)
    assert slices[0].slot_count == 20


def test_resolver_handles_decode_step_one_token_per_request():
    """GIVEN a typical decode step where each in-flight request adds one token,
    WHEN resolve() runs,
    THEN one BatchSlice per request, each with slot_count=1."""
    pytest.importorskip("torch")
    from tardigrade_vllm.slot_resolver import RequestSlotResolver

    # 3 requests each adding 1 token, slots scattered
    am = _make_attn_metadata_mock(
        slot_mapping=[21, 5, 67],  # blocks 1, 0, 4 for the three requests
        query_start_loc=[0, 1, 2, 3],
    )
    slices = RequestSlotResolver().resolve(am, block_size=16)

    assert len(slices) == 3
    assert all(s.slot_count == 1 for s in slices)
    assert [s.block_indices for s in slices] == [(1,), (0,), (4,)]


# -- Step 4: real start_load_kv writes to GPU buffer (Spy pattern) ------------
#
# Constructs TardigradeConnector via __new__ to skip the heavyweight vLLM
# init, then directly invokes .start_load_kv on a mock forward_context built
# from real torch tensors so .copy_() actually moves data. CPU-only.

def _build_bare_connector(num_kv_heads, head_dim, block_size, num_layers):
    """Construct a TardigradeConnector with the minimum attributes start_load_kv touches."""
    pytest.importorskip("vllm", reason="vLLM not installed")
    from tardigrade_vllm.connector import TardigradeConnector

    c = TardigradeConnector.__new__(TardigradeConnector)
    c.num_kv_heads = num_kv_heads
    c.head_dim = head_dim
    c.block_size = block_size
    c.num_layers = num_layers
    c.kv_dim = num_kv_heads * head_dim
    c._load_packs = {}
    c._load_meta = {}
    return c


def _mock_torch_forward_context(num_layers, num_total_blocks, block_size, num_kv_heads, head_dim):
    """Build a mock forward_context with torch CPU tensors for kv_caches."""
    torch = pytest.importorskip("torch")
    ctx = MagicMock()
    shape = (num_total_blocks, block_size, num_kv_heads, head_dim)
    kv_caches = []
    for _ in range(num_layers):
        k = torch.zeros(shape, dtype=torch.float32)
        v = torch.zeros(shape, dtype=torch.float32)
        kv_caches.append((k, v))
    ctx.kv_caches = kv_caches
    return ctx


def _bind_metadata_for_load(connector, requests):
    """Bind a _TardigradeConnectorMetadata DTO onto a bare connector.

    Simulates the IPC path: scheduler populates metadata, worker reads it.
    ``requests`` is a list of dicts with keys: request_id, pack, seq_len,
    block_ids, num_tokens.
    """
    from tardigrade_vllm.connector import _TardigradeConnectorMetadata, _LoadRequestMeta
    meta = _TardigradeConnectorMetadata(load_requests=[
        _LoadRequestMeta(**r) for r in requests
    ])
    connector._connector_metadata = meta


def test_real_start_load_kv_writes_block_slot():
    """GIVEN a manually-staged pack with all-ones K and known V data,
    WHEN TardigradeConnector.start_load_kv runs,
    THEN the allocated GPU block slot is non-zero in both K and V caches."""
    torch = pytest.importorskip("torch")
    num_kv_heads, head_dim, block_size = 2, 4, 4
    num_layers, num_total_blocks = 2, 8
    seq_len = 4  # exactly 1 block worth

    c = _build_bare_connector(num_kv_heads, head_dim, block_size, num_layers)
    ctx = _mock_torch_forward_context(num_layers, num_total_blocks, block_size,
                                       num_kv_heads, head_dim)

    kv_dim = num_kv_heads * head_dim
    k_flat = np.ones(seq_len * kv_dim, dtype=np.float32)
    v_flat = -np.ones(seq_len * kv_dim, dtype=np.float32)
    payload = np.concatenate([k_flat, v_flat])
    pack_layers = [{"layer_idx": li, "data": payload.tolist()}
                   for li in range(num_layers)]

    _bind_metadata_for_load(c, [{
        "request_id": "req_spy",
        "pack": {"pack_id": 99, "layers": pack_layers},
        "seq_len": seq_len,
        "block_ids": [3],
        "num_tokens": seq_len,
    }])

    for k_cache, v_cache in ctx.kv_caches:
        assert torch.all(k_cache == 0)
        assert torch.all(v_cache == 0)

    c.start_load_kv(ctx)

    for layer_idx in range(num_layers):
        k_cache, v_cache = ctx.kv_caches[layer_idx]
        assert torch.all(k_cache[3] == 1.0), (
            f"Layer {layer_idx}: K cache block 3 not written to all-ones; "
            f"got max={k_cache[3].max().item()}, min={k_cache[3].min().item()}"
        )
        assert torch.all(v_cache[3] == -1.0), (
            f"Layer {layer_idx}: V cache block 3 not written to -1.0"
        )
        assert torch.all(k_cache[0] == 0)


def test_real_start_load_kv_ignores_empty_metadata():
    """GIVEN metadata with no load_requests,
    WHEN start_load_kv runs,
    THEN no blocks are touched and no error is raised."""
    torch = pytest.importorskip("torch")
    num_kv_heads, head_dim, block_size = 2, 4, 4
    num_layers, num_total_blocks = 1, 4

    c = _build_bare_connector(num_kv_heads, head_dim, block_size, num_layers)
    ctx = _mock_torch_forward_context(num_layers, num_total_blocks, block_size,
                                       num_kv_heads, head_dim)

    _bind_metadata_for_load(c, [])

    c.start_load_kv(ctx)

    k_cache, v_cache = ctx.kv_caches[0]
    assert torch.all(k_cache == 0), "No blocks should be touched"


# -- Per-method ATDD for scheduler-side hooks (catches silent-early-return bugs)
#
# These tests construct TardigradeConnector via __new__, wire the minimum
# attributes the method-under-test references, and call the method with a
# stub request. The pre-existing `_get_embed_weights` / `vllm_config` bug
# was NOT caught by the end-to-end Layer-3 A/B test because of a silent
# `if query_key is None: return 0, False` path. These tests would have
# caught it before any GPU work.

def _build_bare_connector_for_scheduler(tmp_path, embed_dim=8):
    """Connector with engine + minimal model_config for scheduler-side methods."""
    import tardigrade_db
    from unittest.mock import MagicMock
    pytest.importorskip("vllm", reason="vLLM not installed")
    from tardigrade_vllm.connector import TardigradeConnector

    c = TardigradeConnector.__new__(TardigradeConnector)
    c.engine = tardigrade_db.Engine(str(tmp_path))
    c.owner = 1
    c.kv_dim = embed_dim
    c.num_kv_heads = 2
    c.head_dim = embed_dim // 2
    c.num_layers = 2
    c.block_size = 4
    c._load_packs = {}
    c._load_meta = {}
    c._save_buffers = {}
    c._pack_id_by_fingerprint = {}
    c._embed_weights = None
    c._match_threshold = 1.0  # low threshold so any positive match counts
    c.hidden_size = embed_dim

    # Stub vllm_config — must include attributes _get_embed_weights touches.
    # The pre-existing bug was that this attribute wasn't being stored in
    # __init__. This test will fail loudly if the attribute is missing.
    cfg = MagicMock()
    cfg.model_config.model = "Qwen/Qwen3-0.6B"
    c.vllm_config = cfg

    # Strategy for slot resolution (used in save path; harmless to pre-set)
    from tardigrade_vllm.slot_resolver import RequestSlotResolver
    c._slot_resolver = RequestSlotResolver()

    # Retrieval key strategy (normally set in __init__)
    from tardigrade_vllm.retrieval_key import LastTokenEmbeddingStrategy
    c._retrieval_key_strategy = LastTokenEmbeddingStrategy()

    return c


def test_get_num_new_matched_tokens_does_not_crash_on_empty_engine(tmp_path):
    """GIVEN a connector pointing at an empty engine,
    WHEN get_num_new_matched_tokens is called with a real-shaped request,
    THEN it returns (0, False) and does not raise.

    Regression for: silent early-returns can mask attribute errors. If
    this method raises AttributeError on an upstream attribute (the bug
    we just hit with `self.vllm_config`), the test will fail loudly.
    """
    from unittest.mock import MagicMock
    c = _build_bare_connector_for_scheduler(tmp_path)

    request = MagicMock()
    request.request_id = "req_test_1"
    request.prompt_token_ids = [1, 2, 3, 4, 5]

    # Must not raise. With pack_count == 0 the engine query path is
    # short-circuited, so embed_weights load isn't exercised — but we
    # still pass through enough of the method to catch obvious wiring bugs.
    result = c.get_num_new_matched_tokens(request, 0)
    assert result == (0, False)


def test_get_num_new_matched_tokens_propagates_attribute_errors(tmp_path):
    """GIVEN a connector that's missing a referenced attribute,
    WHEN get_num_new_matched_tokens is called and reaches _get_embed_weights,
    THEN the AttributeError propagates instead of being silently swallowed.

    Direct regression for the `self.vllm_config` bug (2026-04-26): the
    previous broad `except Exception` in _get_embed_weights would catch
    the AttributeError and silently return empty embeddings, masking
    a real programming error for days.
    """
    from unittest.mock import MagicMock
    c = _build_bare_connector_for_scheduler(tmp_path)

    # Intentionally remove vllm_config to simulate the historical bug.
    delattr(c, "vllm_config")

    # Need a pack so the early-return on pack_count==0 doesn't fire.
    key = np.ones(c.kv_dim, dtype=np.float32)
    payload = np.full(2 * 4 * c.kv_dim, 0.5, dtype=np.float32)
    c.engine.mem_write_pack(c.owner, key, [(0, payload)], 80.0)

    request = MagicMock()
    request.request_id = "req_propagate"
    request.prompt_token_ids = [10, 20, 30]

    # The AttributeError must escape — not be swallowed silently.
    with pytest.raises(AttributeError, match="vllm_config"):
        c.get_num_new_matched_tokens(request, 0)


def test_get_num_new_matched_tokens_loads_embed_table_when_packs_exist(tmp_path):
    """GIVEN a connector with at least one stored pack,
    WHEN get_num_new_matched_tokens is called,
    THEN the embedding table is requested AND _compute_retrieval_key
    returns a non-None vector — proves no AttributeError on vllm_config.

    This is the test that would have caught the `self.vllm_config` bug
    BEFORE the GPU integration test wasted hours on a misleading failure.
    """
    from unittest.mock import MagicMock, patch
    c = _build_bare_connector_for_scheduler(tmp_path)

    # Seed the engine so we don't short-circuit
    key = np.ones(c.kv_dim, dtype=np.float32)
    payload = np.full(2 * 4 * c.kv_dim, 0.5, dtype=np.float32)
    c.engine.mem_write_pack(c.owner, key, [(0, payload)], 80.0)

    request = MagicMock()
    request.request_id = "req_test_2"
    request.prompt_token_ids = [10, 20, 30]

    # Patch _get_embed_weights so we don't actually download the model
    # (that would take minutes and require network). The patch verifies
    # the method is REACHED — i.e., we got past every early-return check
    # without raising.
    fake_embeds = np.random.RandomState(0).randn(50000, c.kv_dim).astype(np.float32)
    with patch.object(
        type(c), "_get_embed_weights", return_value=fake_embeds
    ) as mock:
        result = c.get_num_new_matched_tokens(request, 0)
        assert mock.called, (
            "_get_embed_weights was never reached — likely an early-return "
            "before _compute_retrieval_key. This is the silent-failure mode "
            "that masked the `self.vllm_config` AttributeError."
        )

    # Result is (int|None, bool) per vLLM contract; method must not raise.
    assert isinstance(result, tuple) and len(result) == 2


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


# -- Phase 1: Bounded Fingerprint Lifecycle (Bounded Cache pattern) -----------

def _build_connector_for_save_path(tmp_path, max_fingerprints=4):
    """Connector with engine + save-path attributes for fingerprint tests.

    Does NOT require vLLM: uses __new__ to bypass __init__ and manually sets
    only the attributes wait_for_save touches.
    """
    import tardigrade_db
    from collections import OrderedDict

    pytest.importorskip("vllm", reason="vLLM not installed")
    from tardigrade_vllm.connector import TardigradeConnector

    c = TardigradeConnector.__new__(TardigradeConnector)
    c.engine = tardigrade_db.Engine(str(tmp_path))
    c.owner = 1
    c.kv_dim = 8
    c.num_kv_heads = 2
    c.head_dim = 4
    c.num_layers = 1
    c.block_size = 4
    c.hidden_size = 8
    c._load_packs = {}
    c._load_meta = {}
    c._save_buffers = {}
    c._pack_id_by_fingerprint = OrderedDict()
    c._max_fingerprints = max_fingerprints
    c._embed_weights = np.random.RandomState(42).randn(1000, 8).astype(np.float32)
    c._match_threshold = 1.0

    from tardigrade_vllm.slot_resolver import RequestSlotResolver
    c._slot_resolver = RequestSlotResolver()
    from tardigrade_vllm.retrieval_key import LastTokenEmbeddingStrategy
    c._retrieval_key_strategy = LastTokenEmbeddingStrategy()

    cfg = MagicMock()
    cfg.model_config.model = "Qwen/Qwen3-0.6B"
    c.vllm_config = cfg
    return c


def _simulate_save_cycle(connector, batch_idx, block_index, layer_data=None):
    """Simulate one complete save_kv_layer + wait_for_save cycle for a request.

    Manually populates _save_buffers with one layer's data keyed by batch_idx,
    using the given block_index as the fingerprint (block_indices[0]).
    """
    from tardigrade_vllm.slot_resolver import BatchSlice

    if layer_data is None:
        layer_data = np.arange(2 * 4 * connector.kv_dim, dtype=np.float32) * 0.01

    sl = BatchSlice(
        batch_index=batch_idx,
        block_indices=(block_index,),
        slot_count=4,
        first_slot=block_index * connector.block_size,
    )
    connector._save_buffers[batch_idx] = {0: (layer_data, sl)}
    connector.wait_for_save()


def test_fingerprint_map_bounded_after_many_requests(tmp_path):
    """GIVEN a connector with max_num_seqs=4
    AND 20 sequential requests each completing save_kv_layer + wait_for_save
    WHEN all 20 have completed
    THEN len(_pack_id_by_fingerprint) <= 4"""
    c = _build_connector_for_save_path(tmp_path, max_fingerprints=4)

    for i in range(20):
        _simulate_save_cycle(c, batch_idx=0, block_index=100 + i)

    assert len(c._pack_id_by_fingerprint) <= 4, (
        f"Fingerprint map should be bounded to max_num_seqs=4, "
        f"got {len(c._pack_id_by_fingerprint)}"
    )


def test_stale_fingerprint_eviction_prevents_wrong_pack_deletion(tmp_path):
    """GIVEN request A used block_indices[0]=5, wrote pack_id=P_A
    AND request A is evicted from the fingerprint map (capacity exceeded)
    AND request B is allocated the same block_indices[0]=5
    WHEN request B calls wait_for_save
    THEN P_A is NOT deleted (stale entry was evicted before B arrived)
    AND request B's new pack exists in the engine"""
    c = _build_connector_for_save_path(tmp_path, max_fingerprints=2)

    # Request A writes with fingerprint=5
    _simulate_save_cycle(c, batch_idx=0, block_index=5)
    pack_a = c._pack_id_by_fingerprint.get(5)
    assert pack_a is not None, "Request A should have written a pack"

    # Evict A by filling the map past capacity
    _simulate_save_cycle(c, batch_idx=0, block_index=100)
    _simulate_save_cycle(c, batch_idx=0, block_index=101)
    assert 5 not in c._pack_id_by_fingerprint, (
        "Fingerprint 5 should have been evicted"
    )

    # Verify pack A still exists in the engine (not deleted by eviction)
    pack_a_data = c.engine.load_pack_by_id(pack_a)
    assert pack_a_data is not None, (
        "Pack A should still exist — eviction removes map entry, not pack"
    )

    # Request B gets the same block_index=5 (block reuse after A finished)
    _simulate_save_cycle(c, batch_idx=0, block_index=5)
    pack_b = c._pack_id_by_fingerprint.get(5)
    assert pack_b is not None, "Request B should have written a pack"
    assert pack_b != pack_a, "B should have a different pack_id than A"

    # Pack A should still exist — B didn't delete it because the stale
    # fingerprint was already evicted
    pack_a_still = c.engine.load_pack_by_id(pack_a)
    assert pack_a_still is not None, (
        "Pack A must survive: stale fingerprint was evicted before B arrived, "
        "so B never saw it and never called delete_pack on it"
    )


def test_live_request_fingerprint_survives_eviction_of_others(tmp_path):
    """GIVEN requests A, B, C, D all active (max_num_seqs=4)
    AND request E arrives (would exceed capacity)
    WHEN wait_for_save processes E
    THEN request A's fingerprint is evicted (oldest)
    AND requests B, C, D fingerprints remain"""
    c = _build_connector_for_save_path(tmp_path, max_fingerprints=4)

    # Write A, B, C, D
    for i, fp in enumerate([10, 20, 30, 40]):
        _simulate_save_cycle(c, batch_idx=0, block_index=fp)

    assert len(c._pack_id_by_fingerprint) == 4
    assert 10 in c._pack_id_by_fingerprint  # A is oldest

    # Request E arrives — should evict A (oldest)
    _simulate_save_cycle(c, batch_idx=0, block_index=50)

    assert 10 not in c._pack_id_by_fingerprint, "A (oldest) should be evicted"
    assert 20 in c._pack_id_by_fingerprint, "B should survive"
    assert 30 in c._pack_id_by_fingerprint, "C should survive"
    assert 40 in c._pack_id_by_fingerprint, "D should survive"
    assert 50 in c._pack_id_by_fingerprint, "E should be present"


# -- Phase 2: Metadata Bridge (DTO pattern) -----------------------------------

def _build_connector_with_load_state(tmp_path):
    """Connector pre-populated with matched packs for metadata bridge tests."""
    pytest.importorskip("vllm", reason="vLLM not installed")
    from tardigrade_vllm.connector import TardigradeConnector

    c = _build_connector_for_save_path(tmp_path)

    kv_dim = c.kv_dim
    key = np.ones(kv_dim, dtype=np.float32)

    # Write two packs to the engine
    payload1 = np.arange(2 * 4 * kv_dim, dtype=np.float32) * 0.01
    payload2 = np.arange(2 * 4 * kv_dim, dtype=np.float32) * 0.02
    pack_id_1 = c.engine.mem_write_pack(c.owner, key, [(0, payload1)], 80.0)
    pack_id_2 = c.engine.mem_write_pack(c.owner, key * 2, [(0, payload2)], 80.0)

    pack1 = c.engine.load_pack_by_id(pack_id_1)
    pack2 = c.engine.load_pack_by_id(pack_id_2)

    # Simulate scheduler-side matching: populate _load_packs and _load_meta
    c._load_packs = {
        "req_a": {"pack": pack1, "seq_len": 4},
        "req_b": {"pack": pack2, "seq_len": 4},
    }
    c._load_meta = {
        "req_a": {"block_ids": [1, 2], "num_tokens": 4},
        "req_b": {"block_ids": [3], "num_tokens": 4},
    }
    return c


def test_build_connector_meta_packages_matched_packs(tmp_path):
    """GIVEN a connector with 2 matched packs in _load_packs/_load_meta
    WHEN build_connector_meta is called
    THEN returned metadata has 2 load_requests
    AND each contains correct request_id, pack data, seq_len, block_ids, num_tokens"""
    c = _build_connector_with_load_state(tmp_path)

    meta = c.build_connector_meta(MagicMock())

    assert hasattr(meta, "load_requests"), (
        "Metadata must have load_requests field (DTO pattern)"
    )
    assert len(meta.load_requests) == 2, (
        f"Expected 2 load_requests, got {len(meta.load_requests)}"
    )

    req_ids = {r.request_id for r in meta.load_requests}
    assert req_ids == {"req_a", "req_b"}

    for req_meta in meta.load_requests:
        assert req_meta.pack is not None
        assert req_meta.seq_len == 4
        assert isinstance(req_meta.block_ids, list)
        assert req_meta.num_tokens == 4


def test_build_connector_meta_clears_scheduler_state(tmp_path):
    """GIVEN a connector with matched packs
    WHEN build_connector_meta is called
    THEN _load_packs and _load_meta are empty dicts"""
    c = _build_connector_with_load_state(tmp_path)
    assert len(c._load_packs) == 2  # precondition

    c.build_connector_meta(MagicMock())

    assert len(c._load_packs) == 0, "_load_packs must be cleared after build"
    assert len(c._load_meta) == 0, "_load_meta must be cleared after build"


def test_start_load_kv_reads_from_bound_metadata_not_instance_state(tmp_path):
    """GIVEN metadata bound via _connector_metadata (simulating IPC)
    AND _load_packs is empty (proving scheduler state isn't used)
    WHEN start_load_kv is called with valid forward_context
    THEN blocks are written to the correct GPU slots from metadata"""
    torch = pytest.importorskip("torch")
    c = _build_connector_with_load_state(tmp_path)

    # Build metadata via the proper path
    meta = c.build_connector_meta(MagicMock())
    assert len(c._load_packs) == 0  # scheduler state cleared

    # Simulate IPC: bind the metadata on the worker side
    c._connector_metadata = meta

    # Build mock forward context with torch caches
    ctx = _mock_torch_forward_context(
        c.num_layers, 8, c.block_size, c.num_kv_heads, c.head_dim
    )

    c.start_load_kv(ctx)

    # Verify at least one block was written (non-zero)
    wrote_something = False
    for layer_idx in range(c.num_layers):
        k_cache, v_cache = ctx.kv_caches[layer_idx]
        for bid in [1, 2, 3]:  # block_ids from the two requests
            if not torch.allclose(k_cache[bid], torch.zeros_like(k_cache[bid])):
                wrote_something = True
                break
    assert wrote_something, (
        "start_load_kv must read from bound metadata, not from _load_packs"
    )


def test_metadata_survives_serialization_round_trip(tmp_path):
    """GIVEN _TardigradeConnectorMetadata with load request data
    WHEN serialized and deserialized (simulating IPC transport)
    THEN all fields are preserved and usable"""
    import copy
    c = _build_connector_with_load_state(tmp_path)
    meta = c.build_connector_meta(MagicMock())

    # deepcopy simulates serialization round-trip (same contract as IPC)
    meta_copy = copy.deepcopy(meta)

    assert len(meta_copy.load_requests) == len(meta.load_requests)
    for orig, copied in zip(meta.load_requests, meta_copy.load_requests):
        assert orig.request_id == copied.request_id
        assert orig.seq_len == copied.seq_len
        assert orig.block_ids == copied.block_ids
        assert orig.num_tokens == copied.num_tokens


def test_build_connector_meta_with_no_matches_returns_empty(tmp_path):
    """GIVEN a connector with no matched packs
    WHEN build_connector_meta is called
    THEN returned metadata has empty load_requests list
    AND is still a valid KVConnectorMetadata instance"""
    c = _build_connector_for_save_path(tmp_path)
    assert len(c._load_packs) == 0  # no matches

    meta = c.build_connector_meta(MagicMock())

    assert hasattr(meta, "load_requests")
    assert len(meta.load_requests) == 0
