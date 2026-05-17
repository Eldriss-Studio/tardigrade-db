"""ATDD: `tardigrade init` scaffolds a new project directory.

The init subcommand is the first-touch experience — it must:

* Create the target directory if it doesn't exist.
* Lay down a starter file based on the chosen template
  (python-basic or rust-basic).
* Place a README that explains the scaffold.
* Refuse to overwrite an existing non-empty directory unless
  --force is passed (gives a clear hint message).
* Default template is python-basic when none specified.

The CLI entry point is invoked in-process via ``main(argv=[...])``
rather than ``subprocess.run`` — that way the harness is fast,
deterministic, and avoids depending on PATH or install state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tardigrade_cli.main import main  # noqa: E402


class TestInitCommandScaffolding:
    def test_init_creates_target_directory(self, tmp_path: Path):
        target = tmp_path / "my-project"
        rc = main(["init", "--dir", str(target)])
        assert rc == 0
        assert target.is_dir()

    def test_init_lays_down_python_basic_starter_by_default(self, tmp_path: Path):
        target = tmp_path / "my-project"
        main(["init", "--dir", str(target)])
        # Default template is python-basic.
        assert (target / "main.py").exists()
        assert (target / "README.md").exists()

    def test_init_python_basic_starter_imports_tardigrade_client(
        self, tmp_path: Path
    ):
        target = tmp_path / "my-project"
        main(["init", "--dir", str(target)])
        starter = (target / "main.py").read_text(encoding="utf-8")
        # The starter must point new users at the right API surface.
        assert "TardigradeClient" in starter or "tardigrade_hooks" in starter

    def test_init_with_rust_basic_lays_down_rust_starter(self, tmp_path: Path):
        target = tmp_path / "my-rust-project"
        main(["init", "--dir", str(target), "--template", "rust-basic"])
        assert (target / "Cargo.toml").exists()
        assert (target / "src" / "main.rs").exists()

    def test_init_creates_engine_subdirectory(self, tmp_path: Path):
        target = tmp_path / "my-project"
        main(["init", "--dir", str(target)])
        # The engine_dir/ marker (or similar) — the starter expects a
        # known location for the engine state.
        assert (target / "engine_dir").is_dir()


class TestInitCommandRefusesOverwrite:
    def test_init_refuses_existing_non_empty_dir_without_force(
        self, tmp_path: Path, capsys
    ):
        target = tmp_path / "occupied"
        target.mkdir()
        (target / "existing.txt").write_text("don't clobber me", encoding="utf-8")

        rc = main(["init", "--dir", str(target)])
        assert rc != 0
        out = capsys.readouterr().err + capsys.readouterr().out
        # Error message must hint at --force.
        assert "--force" in (capsys.readouterr().err + out)

    def test_init_force_overwrites(self, tmp_path: Path):
        target = tmp_path / "occupied"
        target.mkdir()
        (target / "leftover.txt").write_text("stale", encoding="utf-8")

        rc = main(["init", "--dir", str(target), "--force"])
        assert rc == 0
        assert (target / "main.py").exists()

    def test_init_accepts_empty_existing_dir(self, tmp_path: Path):
        target = tmp_path / "empty"
        target.mkdir()
        rc = main(["init", "--dir", str(target)])
        assert rc == 0
        assert (target / "main.py").exists()


class TestInitCommandHelp:
    def test_init_help_is_non_empty_and_mentions_template(self, capsys):
        # Forensic UX check: --help is informative.
        with pytest.raises(SystemExit):
            main(["init", "--help"])
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "template" in out.lower()
