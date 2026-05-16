"""ATDD for prepare_phase1_datasets._locomo_rows session-date injection.

Why this test exists
====================

The 2026-05-15 LLM-gated bench smoke surfaced a class of LoCoMo
failures the retrieval pipeline cannot fix: temporal questions whose
ground-truth answers are absolute dates (e.g. *"7 May 2023"*) while
the per-item evidence text only mentions relative references like
*"yesterday"*. The LLM has no way to resolve the relative reference
without the **session timestamp** the prep script previously stripped.

This file pins the contract that ``_locomo_rows()`` injects each
turn's session date into the evidence text, so downstream
retrieve-then-read adapters can answer temporal questions correctly.

Run with::

    python -m pytest benchmarks/scripts/test_prepare_phase1_datasets.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# Make the prep script importable as a module without changing its
# script-shaped layout.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import prepare_phase1_datasets as prep  # noqa: E402


_DATE_S1 = "1:56 pm on 8 May, 2023"
_DATE_S2 = "9:15 am on 25 May, 2023"


def _synthetic_locomo(tmp_path: Path) -> Path:
    """Two-session conversation with one evidence-mode QA per session."""
    payload = [
        {
            "sample_id": "conv-test",
            "conversation": {
                "speaker_a": "Caroline",
                "speaker_b": "Melanie",
                "session_1_date_time": _DATE_S1,
                "session_1": [
                    {
                        "dia_id": "D1:1",
                        "speaker": "Caroline",
                        "text": "I went to a LGBTQ support group yesterday and it was powerful.",
                    },
                    {
                        "dia_id": "D1:2",
                        "speaker": "Melanie",
                        "text": "Wow, glad to hear.",
                    },
                ],
                "session_2_date_time": _DATE_S2,
                "session_2": [
                    {
                        "dia_id": "D2:1",
                        "speaker": "Caroline",
                        "text": "Just got back from running a charity race.",
                    },
                ],
            },
            "qa": [
                {
                    "question": "When did Caroline go to the LGBTQ support group?",
                    "answer": "7 May 2023",
                    "evidence": ["D1:1"],
                },
                {
                    "question": "When did Caroline run a charity race?",
                    "answer": "25 May 2023",
                    "evidence": ["D2:1"],
                },
            ],
        }
    ]
    src = tmp_path / "locomo_synth.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    return src


class TestEvidenceCarriesSessionDate:
    def test_each_evidence_line_prefixed_with_session_date(self, tmp_path: Path):
        src = _synthetic_locomo(tmp_path)
        rows = prep._locomo_rows(src, context_mode="evidence")

        assert len(rows) == 2
        # Session 1 evidence carries session_1's date string.
        assert _DATE_S1 in rows[0]["context"]
        assert "yesterday" in rows[0]["context"]
        # Session 2 evidence carries session_2's date string.
        assert _DATE_S2 in rows[1]["context"]
        assert "charity race" in rows[1]["context"]
        # Cross-session leakage check.
        assert _DATE_S2 not in rows[0]["context"]
        assert _DATE_S1 not in rows[1]["context"]

    def test_evidence_text_still_present(self, tmp_path: Path):
        # Date injection must augment, not replace, the original text.
        src = _synthetic_locomo(tmp_path)
        rows = prep._locomo_rows(src, context_mode="evidence")
        assert "LGBTQ support group" in rows[0]["context"]
        assert "charity race" in rows[1]["context"]


class TestEvidenceOrderAndDedupePreserved:
    def test_dedupe_within_a_session_keeps_one_dated_line(self, tmp_path: Path):
        # If the same dia_id appears twice in `evidence`, the dedupe at
        # context-build time keeps a single dated line (no duplicate
        # date prefixes either).
        payload = [
            {
                "sample_id": "dup",
                "conversation": {
                    "speaker_a": "A",
                    "speaker_b": "B",
                    "session_1_date_time": _DATE_S1,
                    "session_1": [
                        {"dia_id": "D1:1", "speaker": "A", "text": "alpha"},
                    ],
                },
                "qa": [
                    {
                        "question": "Q?",
                        "answer": "alpha",
                        "evidence": ["D1:1", "D1:1"],
                    }
                ],
            }
        ]
        src = tmp_path / "dup.json"
        src.write_text(json.dumps(payload), encoding="utf-8")

        rows = prep._locomo_rows(src, context_mode="evidence")
        assert len(rows) == 1
        # One dated line, no repetition.
        assert rows[0]["context"].count(_DATE_S1) == 1
        assert rows[0]["context"].count("alpha") == 1


class TestFullModeUnaffected:
    def test_full_context_mode_does_not_change_format(self, tmp_path: Path):
        # The fix only targets evidence mode; --locomo-context=full keeps
        # its existing speaker-prefixed joined-turns format.
        src = _synthetic_locomo(tmp_path)
        rows = prep._locomo_rows(src, context_mode="full")

        # All speaker lines appear in the joined transcript.
        for row in rows:
            assert "Caroline:" in row["context"]
            assert "Melanie:" in row["context"]


class TestMalformedDiaIdHandledGracefully:
    def test_evidence_with_unknown_dia_id_is_silently_skipped(self, tmp_path: Path):
        payload = [
            {
                "sample_id": "miss",
                "conversation": {
                    "speaker_a": "A",
                    "speaker_b": "B",
                    "session_1_date_time": _DATE_S1,
                    "session_1": [
                        {"dia_id": "D1:1", "speaker": "A", "text": "real evidence"},
                    ],
                },
                "qa": [
                    {
                        "question": "Q?",
                        "answer": "real evidence",
                        # D9:99 doesn't exist — should be skipped, not crash.
                        "evidence": ["D1:1", "D9:99"],
                    }
                ],
            }
        ]
        src = tmp_path / "missing.json"
        src.write_text(json.dumps(payload), encoding="utf-8")

        rows = prep._locomo_rows(src, context_mode="evidence")
        assert len(rows) == 1
        assert _DATE_S1 in rows[0]["context"]
        assert "real evidence" in rows[0]["context"]


class TestSessionWithoutDateStillBuilds:
    def test_missing_session_date_falls_back_to_undated_evidence(self, tmp_path: Path):
        # Defensive: if a session_N_date_time key is absent for some
        # reason, the item must still ingest (without date prefix) rather
        # than disappearing or crashing.
        payload = [
            {
                "sample_id": "nodate",
                "conversation": {
                    "speaker_a": "A",
                    "speaker_b": "B",
                    # session_1_date_time intentionally omitted.
                    "session_1": [
                        {"dia_id": "D1:1", "speaker": "A", "text": "undated line"},
                    ],
                },
                "qa": [
                    {
                        "question": "Q?",
                        "answer": "undated line",
                        "evidence": ["D1:1"],
                    }
                ],
            }
        ]
        src = tmp_path / "nodate.json"
        src.write_text(json.dumps(payload), encoding="utf-8")

        rows = prep._locomo_rows(src, context_mode="evidence")
        assert len(rows) == 1
        assert "undated line" in rows[0]["context"]
