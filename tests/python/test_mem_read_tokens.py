# ATDD tests for Engine.mem_read_tokens (Direct Token Query API).
#
# Verifies that the new direct-token API produces identical results to
# mem_read with a Python-encoded key, and that it integrates correctly
# with HuggingFaceKVHook.

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.encoding import encode_per_token


@pytest.fixture
def engine_with_tokens():
    """Engine pre-populated with 5 cells, each storing distinct per-token keys."""
    db = tempfile.mkdtemp()
    engine = tardigrade_db.Engine(db)
    dim = 8

    rng = np.random.default_rng(42)
    for cell_idx in range(5):
        # Three distinct token vectors per cell.
        tokens = rng.standard_normal((3, dim)).astype(np.float32)
        # Make one dimension dominant per cell so retrieval is deterministic.
        tokens[0, cell_idx] = 5.0
        encoded = encode_per_token(tokens, dim)
        value = np.zeros(dim, dtype=np.float32)
        engine.mem_write(1, 0, encoded, value, 50.0, None)

    return engine


class TestMemReadTokens:
    """Direct Token Query API parity with the encoded mem_read path."""

    def test_results_match_encoded_path(self, engine_with_tokens):
        """GIVEN an engine with 5 per-token cells
        AND a query as a 2D numpy array
        WHEN both mem_read_tokens and mem_read (with encoded key) are called
        THEN both return the same cell_ids in the same order
        AND the scores match within f32 epsilon."""
        engine = engine_with_tokens
        dim = 8

        query_2d = np.zeros((1, dim), dtype=np.float32)
        query_2d[0, 2] = 5.0  # Should match cell 2 strongest

        direct = engine.mem_read_tokens(query_2d, 5, None)
        encoded = encode_per_token(query_2d, dim)
        classic = engine.mem_read(encoded, 5, None)

        assert len(direct) == len(classic)
        for d, c in zip(direct, classic):
            assert d.cell_id == c.cell_id
            assert abs(d.score - c.score) < 1e-5

    def test_multi_token_query_works(self, engine_with_tokens):
        """GIVEN a 5-token query (typical hook output shape)
        WHEN mem_read_tokens is called
        THEN it returns valid results without panicking."""
        engine = engine_with_tokens
        dim = 8

        # 5 tokens, varied — matches realistic hook output.
        rng = np.random.default_rng(7)
        query = rng.standard_normal((5, dim)).astype(np.float32)

        results = engine.mem_read_tokens(query, 3, None)
        assert len(results) <= 3
        # All returned cell_ids are valid (0..4).
        for r in results:
            assert 0 <= r.cell_id < 5

    def test_handles_non_contiguous_array(self, engine_with_tokens):
        """GIVEN a non-contiguous numpy view (e.g. transposed slice)
        WHEN mem_read_tokens is called
        THEN it produces the same results as a contiguous copy
        (proving the binding handles non-C-order arrays correctly)."""
        engine = engine_with_tokens
        dim = 8

        # Build a 2D array, then take a non-contiguous slice.
        contig = np.zeros((2, dim), dtype=np.float32)
        contig[0, 1] = 5.0
        contig[1, 3] = 5.0

        # Non-contiguous: take every other column then transpose-back.
        # Simpler: take a strided slice that breaks C-contiguity.
        big = np.zeros((4, dim), dtype=np.float32)
        big[0::2] = contig
        non_contig = big[0::2]  # strided view

        assert not non_contig.flags["C_CONTIGUOUS"]

        contig_results = engine.mem_read_tokens(np.ascontiguousarray(non_contig), 5, None)
        non_contig_results = engine.mem_read_tokens(non_contig, 5, None)

        assert [r.cell_id for r in contig_results] == [r.cell_id for r in non_contig_results]
