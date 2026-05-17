"""ATDD: Python binding parity for the streaming write buffer.

The Rust acceptance suite at
``crates/tdb-engine/tests/write_buffer_acceptance.rs`` covers the
semantics. This file pins the binding layer exposes the new
constructor and flush method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_pack(engine, owner: int, marker: float = 1.0) -> int:
    key = np.array([marker, 0.0, 0.0, 0.0], dtype=np.float32)
    layers = [
        (0, np.array([marker] * 64, dtype=np.float32)),
        (1, np.array([marker] * 64, dtype=np.float32)),
    ]
    return engine.mem_write_pack(owner, key, layers, 80.0, None)


class TestBufferedConstructor:
    def test_open_with_write_buffer_returns_engine(self, tmp_path: Path):
        import tardigrade_db
        engine = tardigrade_db.Engine.open_with_write_buffer(
            str(tmp_path / "engine"), 100,
        )
        # Constructor returns a usable Engine.
        assert engine.pack_count() == 0


class TestBufferedSemantics:
    def test_buffered_writes_invisible_until_flush(self, tmp_path: Path):
        import tardigrade_db
        engine = tardigrade_db.Engine.open_with_write_buffer(
            str(tmp_path / "engine"), 100,
        )
        for i in range(5):
            _write_pack(engine, 1, marker=1.0 + i * 0.1)
        assert engine.pack_count() == 0

        engine.flush_buffer()
        assert engine.pack_count() == 5

    def test_buffer_auto_flushes_at_max_batch_size(self, tmp_path: Path):
        import tardigrade_db
        engine = tardigrade_db.Engine.open_with_write_buffer(
            str(tmp_path / "engine"), 3,
        )
        for i in range(3):
            _write_pack(engine, 1, marker=1.0 + i * 0.1)
        # Hit threshold mid-loop → auto-flushed.
        assert engine.pack_count() == 3

    def test_flush_drains_buffer(self, tmp_path: Path):
        import tardigrade_db
        engine = tardigrade_db.Engine.open_with_write_buffer(
            str(tmp_path / "engine"), 100,
        )
        _write_pack(engine, 1, marker=1.0)
        _write_pack(engine, 7, marker=1.1)
        engine.flush()
        assert engine.pack_count() == 2


class TestUnbufferedDefault:
    def test_default_open_is_unbuffered(self, tmp_path: Path):
        import tardigrade_db
        engine = tardigrade_db.Engine(str(tmp_path / "engine"))
        _write_pack(engine, 1, marker=1.0)
        # Unbuffered writes are immediately visible.
        assert engine.pack_count() == 1


class TestIdempotentFlush:
    def test_flush_buffer_on_disabled_engine_is_noop(self, tmp_path: Path):
        import tardigrade_db
        engine = tardigrade_db.Engine(str(tmp_path / "engine"))
        # Should not raise; just a no-op.
        engine.flush_buffer()
        assert engine.pack_count() == 0
