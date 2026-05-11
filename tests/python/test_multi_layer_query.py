"""ATDD tests for multi-layer RRF fusion — Composite pattern."""

import pytest
from tardigrade_hooks.multi_layer_query import rrf_fuse


class TestRRFFusion:
    def test_single_list_preserves_order(self):
        ranked = [{"pack_id": 1, "score": 0.9}, {"pack_id": 2, "score": 0.5}]
        fused = rrf_fuse([ranked], k=60)
        assert [r["pack_id"] for r in fused] == [1, 2]

    def test_shared_packs_rank_highest(self):
        a = [{"pack_id": 1, "score": 0.9}, {"pack_id": 2, "score": 0.5}]
        b = [{"pack_id": 2, "score": 0.8}, {"pack_id": 3, "score": 0.4}]
        fused = rrf_fuse([a, b], k=60)
        assert fused[0]["pack_id"] == 2

    def test_empty_input(self):
        assert rrf_fuse([], k=60) == []
        assert rrf_fuse([[]], k=60) == []

    def test_disjoint_lists(self):
        a = [{"pack_id": 1, "score": 0.9}]
        b = [{"pack_id": 2, "score": 0.9}]
        c = [{"pack_id": 3, "score": 0.9}]
        fused = rrf_fuse([a, b, c], k=60)
        assert len(fused) == 3
