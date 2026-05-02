# ATDD tests for the vague-query refinement API.
#
# Verifies that set_refinement_mode plumbs through to the Rust engine,
# that mem_read still works in every mode, and that "none" is the safe
# default which preserves the existing behavior on the existing test suite.

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.encoding import encode_per_token


@pytest.fixture
def engine():
    db = tempfile.mkdtemp()
    eng = tardigrade_db.Engine(db)
    dim = 8

    rng = np.random.default_rng(0)
    for cell_idx in range(5):
        tokens = rng.standard_normal((3, dim)).astype(np.float32)
        # Make one dimension dominant per cell so retrieval is deterministic.
        tokens[0, cell_idx] = 5.0
        encoded = encode_per_token(tokens, dim)
        value = np.zeros(dim, dtype=np.float32)
        eng.mem_write(1, 0, encoded, value, 50.0, None)
    return eng


class TestRefinementMode:
    def test_default_mode_is_none(self, engine):
        assert engine.refinement_mode() == "none"

    def test_set_centered_then_read_back(self, engine):
        engine.set_refinement_mode("centered")
        assert engine.refinement_mode() == "centered"

    def test_set_prf_with_params(self, engine):
        engine.set_refinement_mode("prf", alpha=0.5, beta=0.5, k_prime=5)
        assert engine.refinement_mode() == "prf"

    def test_set_prf_with_defaults(self, engine):
        engine.set_refinement_mode("prf")
        assert engine.refinement_mode() == "prf"

    def test_unknown_mode_raises(self, engine):
        with pytest.raises(RuntimeError, match="unknown refinement mode"):
            engine.set_refinement_mode("bogus")

    def test_alias_mean_centered_accepted(self, engine):
        engine.set_refinement_mode("mean_centered")
        assert engine.refinement_mode() == "centered"

    def test_alias_latent_prf_accepted(self, engine):
        engine.set_refinement_mode("latent_prf", alpha=0.7, beta=0.3, k_prime=3)
        assert engine.refinement_mode() == "prf"


class TestRefinementBehavior:
    """Each mode produces non-panicking results; default 'none' preserves first-stage."""

    def _query(self, engine, target_dim_idx, dim=8):
        query_2d = np.zeros((1, dim), dtype=np.float32)
        query_2d[0, target_dim_idx] = 5.0
        return engine.mem_read_tokens(query_2d, 5, None)

    def test_none_mode_returns_results(self, engine):
        engine.set_refinement_mode("none")
        results = self._query(engine, target_dim_idx=2)
        assert len(results) > 0

    def test_centered_mode_returns_results(self, engine):
        engine.set_refinement_mode("centered")
        results = self._query(engine, target_dim_idx=2)
        assert len(results) > 0

    def test_prf_mode_returns_results(self, engine):
        engine.set_refinement_mode("prf", alpha=0.7, beta=0.3, k_prime=2)
        results = self._query(engine, target_dim_idx=2)
        assert len(results) > 0

    def test_switching_modes_does_not_corrupt_state(self, engine):
        engine.set_refinement_mode("centered")
        first = [r.cell_id for r in self._query(engine, target_dim_idx=2)]
        engine.set_refinement_mode("none")
        baseline = [r.cell_id for r in self._query(engine, target_dim_idx=2)]
        engine.set_refinement_mode("centered")
        second = [r.cell_id for r in self._query(engine, target_dim_idx=2)]
        # Centered runs are deterministic given identical state.
        assert first == second
        # Baseline returned a non-empty list.
        assert baseline
