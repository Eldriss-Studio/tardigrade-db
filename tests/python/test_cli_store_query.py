"""ATDD: ``tardigrade store`` + ``tardigrade query`` roundtrip.

Pins M0.2 — the two commands work together. Tests use a fresh tmp
engine directory per test (no shared state, no GPU dependency:
TardigradeClient's default ``kv_capture_fn`` is the random-key
stub, which is deterministic enough for substring-level assertions
across the same process).
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


class TestStoreCommand:
    def test_store_returns_zero_exit_code(self, engine_dir: Path, capsys):
        rc = main([
            "store",
            "--dir", str(engine_dir),
            "Alice moved to Berlin in 2021.",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        # Output must mention the assigned pack id so users see what
        # happened.
        assert "pack" in out.lower() or "stored" in out.lower()

    def test_store_with_explicit_owner(self, engine_dir: Path):
        rc = main([
            "store",
            "--dir", str(engine_dir),
            "--owner", "7",
            "fact under owner 7",
        ])
        assert rc == 0

    def test_store_requires_dir(self, capsys):
        # Missing --dir should be an argparse error → exit 2.
        with pytest.raises(SystemExit) as exc:
            main(["store", "some fact"])
        assert exc.value.code != 0


class TestQueryCommand:
    def test_query_against_empty_engine_returns_zero_with_no_results(
        self, engine_dir: Path, capsys,
    ):
        rc = main([
            "query",
            "--dir", str(engine_dir),
            "anything",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        # Empty engine should produce a "no results" line, not crash.
        assert "no results" in out.lower() or "0 results" in out.lower() \
            or out.strip() == ""

    def test_query_with_top_k_flag(self, engine_dir: Path):
        # First, populate one fact so we have something to query.
        main(["store", "--dir", str(engine_dir), "alpha beta gamma"])
        rc = main([
            "query",
            "--dir", str(engine_dir),
            "--top-k", "3",
            "alpha",
        ])
        assert rc == 0


class TestStoreQueryRoundtrip:
    def test_stored_fact_is_retrievable(self, engine_dir: Path, capsys):
        # The marker text must show up in the query output.
        MARKER = "UNIQUE_MARKER_alpha_beta_gamma"
        main(["store", "--dir", str(engine_dir), f"Fact about {MARKER}"])
        capsys.readouterr()  # drain store output

        main([
            "query",
            "--dir", str(engine_dir),
            "--top-k", "5",
            MARKER,
        ])
        out = capsys.readouterr().out
        assert MARKER in out, f"marker {MARKER!r} not in query output:\n{out}"


class TestHelp:
    def test_store_help_mentions_owner(self, capsys):
        with pytest.raises(SystemExit):
            main(["store", "--help"])
        captured = capsys.readouterr()
        assert "owner" in (captured.out + captured.err).lower()

    def test_query_help_mentions_top_k(self, capsys):
        with pytest.raises(SystemExit):
            main(["query", "--help"])
        captured = capsys.readouterr()
        assert "top" in (captured.out + captured.err).lower()
