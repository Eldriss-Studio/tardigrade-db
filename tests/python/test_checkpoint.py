"""ATDD: Python binding parity for ``CheckpointRepository`` (M3.1).

Mirrors the Rust acceptance suite at
``crates/tdb-engine/tests/checkpoint_acceptance.rs``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PACK_SALIENCE = 80.0
LAYER_DIM = 64


def _write_pack(engine, owner: int, marker: float) -> int:
    key = np.array([marker, 0.0, 0.0, 0.0], dtype=np.float32)
    layers = [
        (0, np.array([marker] * LAYER_DIM, dtype=np.float32)),
        (1, np.array([marker] * LAYER_DIM, dtype=np.float32)),
    ]
    return engine.mem_write_pack(owner, key, layers, PACK_SALIENCE, None)


@pytest.fixture
def populated(tmp_path: Path):
    import tardigrade_db
    engine = tardigrade_db.Engine(str(tmp_path / "engine"))
    _write_pack(engine, 1, 1.0)
    _write_pack(engine, 7, 1.1)
    _write_pack(engine, 42, 1.2)
    repo = tardigrade_db.CheckpointRepository(str(tmp_path / "checkpoints"))
    return engine, repo, tmp_path


class TestCheckpointRepository:
    def test_save_returns_entry_with_seq_one(self, populated):
        engine, repo, _ = populated
        entry = repo.save(engine, "autosave")
        assert entry["label"] == "autosave"
        assert entry["seq"] == 1
        assert entry["manifest"]["pack_count"] == 3
        assert Path(entry["path"]).is_file()

    def test_save_assigns_monotonic_sequence_per_label(self, populated):
        engine, repo, _ = populated
        a = repo.save(engine, "autosave")
        b = repo.save(engine, "autosave")
        c = repo.save(engine, "manual")
        assert (a["seq"], b["seq"], c["seq"]) == (1, 2, 1)

    def test_list_returns_sorted_entries(self, populated):
        engine, repo, _ = populated
        repo.save(engine, "manual")
        repo.save(engine, "autosave")
        repo.save(engine, "autosave")

        entries = repo.list()
        summary = [(e["label"], e["seq"]) for e in entries]
        assert summary == [
            ("autosave", 1),
            ("autosave", 2),
            ("manual", 1),
        ]

    def test_latest_returns_highest_seq(self, populated):
        engine, repo, _ = populated
        repo.save(engine, "autosave")
        repo.save(engine, "autosave")
        third = repo.save(engine, "autosave")
        latest = repo.latest("autosave")
        assert latest is not None
        assert latest["seq"] == third["seq"]

    def test_latest_unknown_label_returns_none(self, populated):
        _, repo, _ = populated
        assert repo.latest("nope") is None

    def test_restore_latest_yields_engine_with_same_packs(self, populated):
        engine, repo, tmp_path = populated
        repo.save(engine, "autosave")
        target = tmp_path / "restored"
        restored = repo.restore_latest("autosave", str(target))
        assert restored.list_owners() == [1, 7, 42]
        assert restored.pack_count() == 3

    def test_restore_latest_unknown_label_raises(self, populated):
        _, repo, tmp_path = populated
        with pytest.raises(RuntimeError):
            repo.restore_latest("missing", str(tmp_path / "restored"))
