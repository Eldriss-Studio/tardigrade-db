# ATDD tests for the cross-encoder reranker (Stage-2 over text-bearing
# first-stage candidates). The cross-encoder model is small (~22M
# params) and downloads automatically; tests are skipped if the model
# can't be loaded (e.g. offline CI).

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


@pytest.fixture(scope="module")
def reranker():
    pytest.importorskip("sentence_transformers")
    from tardigrade_hooks.reranker import CrossEncoderReranker

    try:
        return CrossEncoderReranker()
    except Exception as exc:
        pytest.skip(f"cross-encoder model unavailable: {exc}")


class TestCrossEncoderReranker:
    def test_promotes_lexically_relevant_candidate_above_distractor(self, reranker):
        # GIVEN two candidates, only one of which actually answers the query
        candidates = [
            {"id": "distractor", "text": "The weather in Berlin is cloudy."},
            {"id": "answer", "text": "Alice moved to Berlin in 2021."},
        ]
        query = "Where did Alice move?"

        # WHEN the reranker is applied
        ordered = reranker.rerank(
            query_text=query,
            candidates=candidates,
            get_text=lambda c: c["text"],
        )

        # THEN the answer ranks first
        assert ordered[0]["id"] == "answer"
        assert ordered[1]["id"] == "distractor"

    def test_textless_candidates_are_preserved_not_dropped(self, reranker):
        candidates = [
            {"id": "with_text", "text": "Berlin is the capital of Germany."},
            {"id": "no_text", "text": None},
            {"id": "another_text", "text": "Paris is the capital of France."},
        ]
        ordered = reranker.rerank(
            query_text="Where is Berlin?",
            candidates=candidates,
            get_text=lambda c: c["text"],
        )
        ids = [c["id"] for c in ordered]
        assert set(ids) == {"with_text", "no_text", "another_text"}
        # The text-less candidate ends up after the text-scored ones.
        assert ids.index("no_text") == 2

    def test_empty_candidates_returns_empty(self, reranker):
        ordered = reranker.rerank(
            query_text="anything",
            candidates=[],
            get_text=lambda c: "",
        )
        assert ordered == []

    def test_blank_text_treated_as_missing(self, reranker):
        candidates = [
            {"id": "blank", "text": "   "},
            {"id": "real", "text": "Cats are mammals."},
        ]
        ordered = reranker.rerank(
            query_text="What are cats?",
            candidates=candidates,
            get_text=lambda c: c["text"],
        )
        assert ordered[0]["id"] == "real"

    def test_metadata_reports_model(self, reranker):
        meta = reranker.metadata()
        assert meta["reranker"] == "cross_encoder"
        assert "MiniLM" in meta["model"] or "bge" in meta["model"].lower()
