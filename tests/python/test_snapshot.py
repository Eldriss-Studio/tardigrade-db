"""ATDD: Python binding parity for snapshot/restore.

The Rust acceptance suite at
``crates/tdb-engine/tests/snapshot_acceptance.rs`` covers semantics
end-to-end. This file pins the binding layer exposes the snapshot +
restore_from surface and returns a manifest dict the consumer can
inspect.
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


@pytest.fixture
def engine(tmp_path: Path):
    import tardigrade_db
    eng = tardigrade_db.Engine(str(tmp_path / "engine"))
    _write_pack(eng, 1, marker=1.0)
    _write_pack(eng, 7, marker=1.1)
    _write_pack(eng, 42, marker=1.2)
    return eng


class TestSnapshotManifest:
    def test_returns_manifest_with_expected_keys(self, engine, tmp_path):
        snap_dir = tmp_path / "snaps"
        snap_dir.mkdir()
        snap_file = snap_dir / "snap.tar"
        manifest = engine.snapshot(str(snap_file))

        # The manifest is exposed as a Python dict for ergonomic
        # consumption — no opaque Rust types leaking through.
        for key in (
            "magic",
            "format_version",
            "created_at",
            "pack_count",
            "owner_count",
            "sha256",
            "quantization_codec",
            "key_codec",
        ):
            assert key in manifest, f"missing key {key} in manifest {manifest}"

        assert manifest["magic"] == "tdb!"
        assert manifest["format_version"] == 1
        assert manifest["pack_count"] == 3
        assert manifest["owner_count"] == 3
        assert manifest["quantization_codec"] == "q4"
        assert manifest["key_codec"] == "top5avg"

    def test_writes_archive_file(self, engine, tmp_path):
        snap_dir = tmp_path / "snaps"
        snap_dir.mkdir()
        snap_file = snap_dir / "snap.tar"
        engine.snapshot(str(snap_file))
        assert snap_file.is_file()
        assert snap_file.stat().st_size > 0


class TestRestoreFrom:
    def test_restore_yields_engine_with_same_owners(self, engine, tmp_path):
        import tardigrade_db

        snap_dir = tmp_path / "snaps"
        snap_dir.mkdir()
        snap_file = snap_dir / "snap.tar"
        engine.snapshot(str(snap_file))

        target = tmp_path / "restored"
        restored = tardigrade_db.Engine.restore_from(str(snap_file), str(target))
        assert restored.list_owners() == [1, 7, 42]
        assert restored.pack_count() == 3

    def test_restore_rejects_non_snapshot(self, tmp_path):
        import tardigrade_db

        # Write a plain text file disguised as a tar; restore_from
        # should raise.
        bogus = tmp_path / "not-a-snapshot.tar"
        bogus.write_bytes(b"definitely not a tar")
        target = tmp_path / "restored"

        with pytest.raises(RuntimeError):
            tardigrade_db.Engine.restore_from(str(bogus), str(target))


class TestSnapshotFootGunGuard:
    def test_snapshot_refuses_out_path_inside_engine_dir(
        self, engine, tmp_path,
    ):
        # The engine directory was created by the fixture at
        # tmp_path / "engine". Writing the snapshot inside that
        # directory would let the tar walker read its own output.
        bad = tmp_path / "engine" / "snap.tar"
        with pytest.raises(RuntimeError) as exc:
            engine.snapshot(str(bad))
        assert "engine" in str(exc.value).lower()
