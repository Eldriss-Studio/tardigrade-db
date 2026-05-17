"""ATDD: full-conversation LoCoMo prep.

`--locomo-context=full` mode produces full ~62 K-char conversation
contexts (vs ~500 chars in evidence mode). This pins the shape so
the challenger pipeline's ingest path can rely on it.

One slice — the GPU smoke (B2.2 / B2.3) is gated on B1
(capture-model swap) and lives in
``tests/python/test_capture_model_swap.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import prepare_phase1_datasets as prep  # noqa: E402


_DATE_S1 = "1:56 pm on 8 May, 2023"
_DATE_S2 = "9:15 am on 25 May, 2023"


def _synthetic_locomo_two_sessions(tmp_path: Path) -> Path:
    """Two-session conv with three QAs — one needs both sessions."""
    payload = [
        {
            "sample_id": "conv-test",
            "conversation": {
                "speaker_a": "Caroline",
                "speaker_b": "Melanie",
                "session_1_date_time": _DATE_S1,
                "session_1": [
                    {"dia_id": "D1:1", "speaker": "Caroline", "text": "Joined the LGBTQ support group yesterday."},
                    {"dia_id": "D1:2", "speaker": "Melanie", "text": "That's wonderful, Caroline."},
                ],
                "session_2_date_time": _DATE_S2,
                "session_2": [
                    {"dia_id": "D2:1", "speaker": "Caroline", "text": "Just got back from running a charity race."},
                    {"dia_id": "D2:2", "speaker": "Melanie", "text": "Was it fun?"},
                ],
            },
            "qa": [
                {"question": "What did Caroline join?", "answer": "LGBTQ support group", "evidence": ["D1:1"]},
                {"question": "When did Caroline run the charity race?", "answer": "25 May 2023", "evidence": ["D2:1"]},
                {"question": "Who did Caroline talk to?", "answer": "Melanie", "evidence": ["D1:2", "D2:2"]},
            ],
        }
    ]
    src = tmp_path / "locomo_full_test.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    return src


class TestFullContextMode:
    def test_each_row_carries_full_conversation_not_just_evidence(self, tmp_path: Path):
        src = _synthetic_locomo_two_sessions(tmp_path)
        rows = prep._locomo_rows(src, context_mode="full")

        assert len(rows) == 3
        # Every QA row sees the full transcript regardless of which
        # session its evidence sits in — that's the whole point.
        for row in rows:
            assert "LGBTQ support group" in row["context"]
            assert "charity race" in row["context"]

    def test_speaker_prefixes_preserved(self, tmp_path: Path):
        src = _synthetic_locomo_two_sessions(tmp_path)
        rows = prep._locomo_rows(src, context_mode="full")

        for row in rows:
            assert "Caroline:" in row["context"]
            assert "Melanie:" in row["context"]

    def test_full_mode_context_is_longer_than_evidence_mode(self, tmp_path: Path):
        src = _synthetic_locomo_two_sessions(tmp_path)
        full_rows = prep._locomo_rows(src, context_mode="full")
        ev_rows = prep._locomo_rows(src, context_mode="evidence")

        for full, ev in zip(full_rows, ev_rows):
            # Full mode strictly includes more than the per-item slice.
            assert len(full["context"]) > len(ev["context"])

    def test_full_mode_keeps_all_qas(self, tmp_path: Path):
        # Evidence mode drops items with empty evidence arrays; full
        # mode keeps everything because every item has full context
        # by definition.
        payload = [
            {
                "sample_id": "edge",
                "conversation": {
                    "speaker_a": "A",
                    "speaker_b": "B",
                    "session_1_date_time": _DATE_S1,
                    "session_1": [
                        {"dia_id": "D1:1", "speaker": "A", "text": "real evidence"}
                    ],
                },
                "qa": [
                    {"question": "Q1?", "answer": "yes", "evidence": []},  # no evidence
                    {"question": "Q2?", "answer": "no", "evidence": ["D1:1"]},
                ],
            }
        ]
        src = tmp_path / "edge.json"
        src.write_text(json.dumps(payload), encoding="utf-8")

        full = prep._locomo_rows(src, context_mode="full")
        ev = prep._locomo_rows(src, context_mode="evidence")

        # Evidence mode skips the empty-evidence item (Stage 2 fix).
        assert len(ev) == 1
        # Full mode keeps both.
        assert len(full) == 2
