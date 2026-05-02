"""ATDD acceptance tests for PyPI packaging."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db


def test_version_exists():
    """ATDD 1: Module exposes __version__ attribute."""
    assert hasattr(tardigrade_db, "__version__")
    assert isinstance(tardigrade_db.__version__, str)
    assert len(tardigrade_db.__version__) > 0


def test_version_matches_cargo():
    """ATDD 2: __version__ tracks the workspace Cargo.toml version, not a hardcoded value."""
    cargo_path = Path(__file__).resolve().parents[2] / "Cargo.toml"
    cargo_text = cargo_path.read_text()
    # Workspace `[workspace.package] version = "x.y.z"` is the source of truth.
    expected = next(
        line.split('"')[1]
        for line in cargo_text.splitlines()
        if line.startswith("version") and '"' in line
    )
    assert tardigrade_db.__version__ == expected


def test_module_has_engine():
    """ATDD 3: Module exposes Engine class."""
    assert hasattr(tardigrade_db, "Engine")


def test_module_has_read_result():
    """ATDD 4: Module exposes ReadResult class."""
    assert hasattr(tardigrade_db, "ReadResult")


def test_installed_api_smoke(tmp_path):
    """ATDD 5: Create engine, write, read — basic smoke test."""
    import numpy as np

    engine = tardigrade_db.Engine(str(tmp_path))
    key = np.ones(32, dtype=np.float32)
    val = np.zeros(32, dtype=np.float32)

    cell_id = engine.mem_write(1, 0, key, val, 50.0, None)
    assert cell_id == 0
    assert engine.cell_count() == 1

    results = engine.mem_read(key, 1, None)
    assert len(results) == 1
    assert results[0].cell_id == 0
