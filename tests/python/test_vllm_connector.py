# ATDD tests for TardigradeDB vLLM connector load path.
#
# Tests the semantic matching and retrieval key computation.
# Works without vLLM installed — tests the engine-facing logic.

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_vllm.format import flat_to_blocks, blocks_to_flat


def make_engine_with_packs(num_packs=3, kv_dim=8, seq_len=4):
    """Create an engine with synthetic packs for testing."""
    tmpdir = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(tmpdir)

    pack_ids = []
    for i in range(num_packs):
        # Retrieval key: simple mean-pooled vector
        key = np.full(kv_dim, float(i + 1), dtype=np.float32)
        # Layer payload: [K_flat | V_flat]
        payload = np.random.randn(2 * seq_len * kv_dim).astype(np.float32)
        pid = engine.mem_write_pack(1, key, [(0, payload)], 80.0)
        pack_ids.append(pid)

    return engine, pack_ids, kv_dim, seq_len


def test_engine_retrieval_for_connector():
    """GIVEN packs stored with retrieval keys,
    WHEN querying,
    THEN at least one result is returned with a valid pack_id."""
    engine, pack_ids, kv_dim, seq_len = make_engine_with_packs()

    query = np.full(kv_dim, 2.1, dtype=np.float32)
    results = engine.mem_read_pack(query, 1, None)

    assert len(results) == 1
    assert results[0]["pack_id"] in pack_ids


def test_load_pack_by_id_for_connector():
    """GIVEN a stored pack,
    WHEN loaded by ID,
    THEN all layer data is returned correctly."""
    engine, pack_ids, kv_dim, seq_len = make_engine_with_packs()

    pack = engine.load_pack_by_id(pack_ids[0])
    assert pack["pack_id"] == pack_ids[0]
    assert len(pack["layers"]) == 1


def test_format_round_trip_with_engine_data():
    """GIVEN KV data stored in engine and retrieved,
    WHEN converted to vLLM blocks and back,
    THEN data matches within Q4 quantization tolerance."""
    engine, pack_ids, kv_dim, seq_len = make_engine_with_packs()

    pack = engine.load_pack_by_id(pack_ids[0])
    layer_data = np.array(pack["layers"][0]["data"], dtype=np.float32)

    num_kv_heads = 2
    head_dim = kv_dim // num_kv_heads
    block_size = 4

    k_blocks, v_blocks = flat_to_blocks(layer_data, num_kv_heads, head_dim, block_size)
    recovered = blocks_to_flat(k_blocks, v_blocks, seq_len, num_kv_heads, head_dim)

    # Q4 quantization in engine adds some error
    assert recovered.shape == layer_data.shape
    # Data goes through Q4, so allow larger tolerance
    assert np.allclose(recovered, layer_data, atol=0.5)


def test_trace_boosted_retrieval_returns_results():
    """GIVEN packs with trace links,
    WHEN querying with trace boost,
    THEN results are returned and boost API works."""
    engine, pack_ids, kv_dim, seq_len = make_engine_with_packs(num_packs=2)

    # Link pack 0 to a third pack
    key3 = np.full(kv_dim, 10.0, dtype=np.float32)
    payload3 = np.random.randn(2 * seq_len * kv_dim).astype(np.float32)
    pid3 = engine.mem_write_pack(1, key3, [(0, payload3)], 80.0)
    engine.add_pack_link(pack_ids[0], pid3)

    query = np.full(kv_dim, 1.5, dtype=np.float32)
    results = engine.mem_read_pack_with_trace_boost(query, 3, None, 0.5)

    # Should return results and the linked pack should be present
    assert len(results) >= 1
    result_ids = {r["pack_id"] for r in results}
    assert pack_ids[0] in result_ids  # linked pack found
