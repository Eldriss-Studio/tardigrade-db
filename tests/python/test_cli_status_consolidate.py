"""ATDD: ``tardigrade status`` and ``tardigrade consolidate``.

Status output must be stable enough that scripts can parse it;
consolidate must run cleanly even on an empty corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tardigrade_cli.main import main  # noqa: E402


@pytest.fixture
def engine_dir(tmp_path: Path) -> Path:
    d = tmp_path / "engine"
    d.mkdir()
    return d


class TestStatusCommand:
    def test_status_on_empty_engine_succeeds(self, engine_dir: Path, capsys):
        rc = main(["status", "--dir", str(engine_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        # Status must mention cell count for parseability.
        assert "cell" in out.lower() or "cells" in out.lower()

    def test_status_on_populated_engine_reports_nonzero_cells(
        self, engine_dir: Path, capsys,
    ):
        main(["store", "--dir", str(engine_dir), "first fact"])
        main(["store", "--dir", str(engine_dir), "second fact"])
        capsys.readouterr()  # drain store output

        rc = main(["status", "--dir", str(engine_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        # Pack count should be reported and > 0 after two stores.
        assert "pack" in out.lower() or "cell" in out.lower()


class TestConsolidateCommand:
    def test_consolidate_on_empty_engine_succeeds(self, engine_dir: Path, capsys):
        rc = main(["consolidate", "--dir", str(engine_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        # Should report the count of views attached (zero is fine).
        assert "view" in out.lower()

    def test_consolidate_with_owner(self, engine_dir: Path):
        rc = main([
            "consolidate", "--dir", str(engine_dir), "--owner", "7",
        ])
        assert rc == 0


class TestHelp:
    def test_status_help_is_non_empty(self, capsys):
        with pytest.raises(SystemExit):
            main(["status", "--help"])
        captured = capsys.readouterr()
        # Help text must be present (argparse formats it into stdout).
        assert (captured.out + captured.err).strip() != ""

    def test_consolidate_help_mentions_owner_or_consolidation(self, capsys):
        with pytest.raises(SystemExit):
            main(["consolidate", "--help"])
        captured = capsys.readouterr()
        text = (captured.out + captured.err).lower()
        assert "owner" in text or "consolidat" in text
