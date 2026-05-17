"""ATDD: Python binding parity for the owner registry.

The Rust-side acceptance suite at
``crates/tdb-engine/tests/owner_registry.rs`` already covers the
semantics; this file pins the binding layer exposes the three
``list_owners`` / ``owner_exists`` / ``delete_owner`` methods at
parity.

Tests use ``tardigrade_db.Engine`` directly so they target the
PyO3 surface, not the higher-level ``TardigradeClient`` facade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def engine(tmp_path: Path):
    import tardigrade_db
    return tardigrade_db.Engine(str(tmp_path / "engine"))


def _write_pack(engine, owner: int, marker: float = 1.0) -> int:
    """Helper: write a minimal pack under ``owner`` and return its id."""
    # mem_write_pack expects 1-D float32 ndarrays.
    key = np.array([marker, 0.0, 0.0, 0.0], dtype=np.float32)
    layers = [
        (0, np.array([marker] * 64, dtype=np.float32)),
        (1, np.array([marker] * 64, dtype=np.float32)),
    ]
    return engine.mem_write_pack(owner, key, layers, 80.0, None)


class TestListOwners:
    def test_empty_engine_returns_empty_list(self, engine):
        assert engine.list_owners() == []

    def test_returns_sorted_unique_owners(self, engine):
        _write_pack(engine, 42, marker=1.0)
        _write_pack(engine, 7, marker=1.1)
        _write_pack(engine, 42, marker=1.2)  # duplicate
        _write_pack(engine, 1, marker=1.3)

        assert engine.list_owners() == [1, 7, 42]


class TestOwnerExists:
    def test_false_on_empty(self, engine):
        assert engine.owner_exists(1) is False

    def test_true_after_write(self, engine):
        _write_pack(engine, 99, marker=1.0)
        assert engine.owner_exists(99) is True
        assert engine.owner_exists(98) is False


class TestDeleteOwner:
    def test_removes_only_target_packs(self, engine):
        _write_pack(engine, 1, marker=1.0)
        _write_pack(engine, 7, marker=1.1)
        _write_pack(engine, 7, marker=1.2)
        _write_pack(engine, 42, marker=1.3)

        removed = engine.delete_owner(7)
        assert removed == 2
        assert engine.list_owners() == [1, 42]
        assert engine.owner_exists(7) is False

    def test_returns_zero_for_unknown_owner(self, engine):
        _write_pack(engine, 1, marker=1.0)
        assert engine.delete_owner(999) == 0
        assert engine.owner_exists(1) is True

    def test_is_idempotent(self, engine):
        _write_pack(engine, 5, marker=1.0)
        assert engine.delete_owner(5) == 1
        assert engine.delete_owner(5) == 0
        assert engine.owner_exists(5) is False
