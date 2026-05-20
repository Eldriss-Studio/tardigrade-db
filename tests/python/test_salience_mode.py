"""Acceptance tests for the `salience_mode` parameter on the write APIs.

`salience_mode` lets the engine derive pack salience from the retrieval
key directly. When omitted (or `"none"`) the caller's explicit
`salience` is stored verbatim — existing contract unchanged.

The derivation lives in `tdb_core::salience::SalienceMode` and skips the
encoded key's header, so the value reflects token data magnitude rather
than header padding. `tardigrade_db.SALIENCE_SCALE` and
`tardigrade_db.SALIENCE_CAP` mirror the Rust constants so tests can
restate the formula without baking in magic literals.
"""

from __future__ import annotations

import numpy as np
import pytest

import tardigrade_db
from tardigrade_db import (
    ENCODING_HEADER_SIZE,
    SALIENCE_CAP,
    SALIENCE_SCALE,
)
from tardigrade_hooks.encoding import encode_per_token


DIM = 16
OWNER = 1
ON_UPDATE_BUMP = 5.0  # ImportanceScorer adds this on write; pinned by tdb-governance.


def _engine(path):
    return tardigrade_db.Engine(str(path), vamana_threshold=9999)


def _random_tokens(seed: int, n_tokens: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_tokens, DIM)).astype(np.float32)


def _layer(seed: int) -> tuple[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (0, rng.standard_normal(64).astype(np.float32))


def _expected_l2(tokens: np.ndarray) -> float:
    encoded = encode_per_token(tokens, DIM)
    data = encoded[ENCODING_HEADER_SIZE:]
    n = max(len(data), 1)
    return float(min(np.linalg.norm(data) / n * SALIENCE_SCALE, SALIENCE_CAP))


def _expected_max(tokens: np.ndarray) -> float:
    encoded = encode_per_token(tokens, DIM)
    data = encoded[ENCODING_HEADER_SIZE:]
    return float(min(np.max(np.abs(data)) * SALIENCE_SCALE, SALIENCE_CAP))


def test_default_salience_mode_preserves_caller_value(tmp_path):
    engine = _engine(tmp_path)
    pid = engine.mem_write_pack_tokens(OWNER, _random_tokens(1), [_layer(1)], 42.5)
    pi = engine.pack_importance(pid)
    assert pi is not None
    assert abs(pi - (42.5 + ON_UPDATE_BUMP)) < 1e-4


def test_salience_mode_l2_matches_rust_formula(tmp_path):
    engine = _engine(tmp_path)
    tokens = _random_tokens(2)
    expected = _expected_l2(tokens) + ON_UPDATE_BUMP

    pid = engine.mem_write_pack_tokens(
        OWNER, tokens, [_layer(2)], 0.0, salience_mode="l2"
    )
    pi = engine.pack_importance(pid)
    assert pi is not None
    assert abs(pi - expected) < 1e-3, f"l2 salience {pi} != {expected}"


def test_salience_mode_max_uses_inf_norm(tmp_path):
    engine = _engine(tmp_path)
    # Choose tokens whose max(abs) lands well under SALIENCE_CAP / SALIENCE_SCALE
    # so the comparison stays in the linear regime — the cap is exercised
    # separately by `test_salience_mode_clamps_at_cap`.
    safe_magnitude = SALIENCE_CAP / SALIENCE_SCALE / 4.0
    tokens = (np.ones((4, DIM), dtype=np.float32) * safe_magnitude)
    expected = _expected_max(tokens) + ON_UPDATE_BUMP

    pid = engine.mem_write_pack_tokens(
        OWNER, tokens, [_layer(3)], 0.0, salience_mode="max"
    )
    pi = engine.pack_importance(pid)
    assert pi is not None
    assert abs(pi - expected) < 1e-3


def test_salience_mode_none_preserves_explicit_value(tmp_path):
    engine = _engine(tmp_path)
    pid = engine.mem_write_pack_tokens(
        OWNER, _random_tokens(4), [_layer(4)], 33.7, salience_mode="none"
    )
    pi = engine.pack_importance(pid)
    assert pi is not None
    assert abs(pi - (33.7 + ON_UPDATE_BUMP)) < 1e-4


def test_unknown_salience_mode_raises(tmp_path):
    engine = _engine(tmp_path)
    with pytest.raises((RuntimeError, ValueError)) as exc:
        engine.mem_write_pack_tokens(
            OWNER, _random_tokens(5), [_layer(5)], 50.0, salience_mode="banana"
        )
    assert "salience" in str(exc.value).lower() or "mode" in str(exc.value).lower()


def test_salience_mode_clamps_at_cap(tmp_path):
    """Pathological large-magnitude key: derived salience must clamp at
    SALIENCE_CAP, and the resulting importance must respect the scorer's
    own cap as well."""
    engine = _engine(tmp_path)
    huge = np.ones((4, DIM), dtype=np.float32) * 1e6
    pid = engine.mem_write_pack_tokens(
        OWNER, huge, [_layer(6)], 0.0, salience_mode="l2"
    )
    pi = engine.pack_importance(pid)
    assert pi is not None
    # Salience clamps at SALIENCE_CAP (100); importance is then SALIENCE_CAP
    # plus the on-update bump, capped again at SALIENCE_CAP by the scorer.
    assert pi <= SALIENCE_CAP + ON_UPDATE_BUMP + 1e-3
