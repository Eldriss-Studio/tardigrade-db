"""ATDD: ``TardigradeClient.encode_query`` convenience.

A consumer that wants the raw retrieval key (e.g. to pass it
directly into ``engine.mem_read_pack``, or to log it, or to cache
it across queries) shouldn't have to reach into the private
``_kv_fn`` attribute. ``encode_query(text)`` returns just the
per-token key as a numpy array.

Pinned behavior:

- Returns a numpy float32 array.
- Deterministic — same text yields the same key under the
  default stub.
- Round-trips: encoding ``text`` and passing the result to
  ``engine.mem_read_pack`` retrieves a pack stored as ``text``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def client(tmp_path: Path):
    from tardigrade_hooks import TardigradeClient
    return TardigradeClient(tmp_path / "engine")


class TestEncodeQuery:
    def test_returns_float32_numpy_array(self, client):
        key = client.encode_query("anything")
        assert isinstance(key, np.ndarray)
        assert key.dtype == np.float32

    def test_deterministic_for_same_text(self, client):
        a = client.encode_query("the same words")
        b = client.encode_query("the same words")
        assert np.array_equal(a, b)

    def test_round_trips_via_engine_mem_read_pack(self, client):
        text = "the cat sat on the mat"
        pid = client.store(text)
        key = client.encode_query(text)
        results = client.engine.mem_read_pack(key, 5, 1)
        assert any(r["pack_id"] == pid for r in results)
