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


class TestCrossEncoderRerankerThreadSafety:
    """Concurrent `rerank` calls must produce the same ordering each
    thread would have seen in isolation. PyTorch model.forward is not
    thread-safe — without an internal lock, parallel `predict` calls
    race on shared state and yield non-deterministic scores. Observed
    as a 0.42 → 0.38 LoCoMo headline drift under `--workers 4`
    (bench audit 2026-05-16 task #96)."""

    _QUERIES = [
        ("Where did Alice move?", "alice"),
        ("What is the capital of France?", "paris"),
        ("Who married Aaron?", "sonia"),
        ("What animal is Bowie?", "bowie"),
    ]
    _CANDS = [
        {"id": "alice", "text": "Alice moved to Berlin in 2021."},
        {"id": "paris", "text": "Paris is the capital of France."},
        {"id": "sonia", "text": "Sonia married Aaron in 2019."},
        {"id": "bowie", "text": "Aaron and Sonia have a dog named Bowie."},
        {"id": "distractor1", "text": "The weather in Berlin is cloudy."},
        {"id": "distractor2", "text": "Cats are mammals."},
    ]

    def _baseline_orderings(self, reranker):
        return {
            label: [
                c["id"]
                for c in reranker.rerank(
                    query_text=q,
                    candidates=list(self._CANDS),
                    get_text=lambda c: c["text"],
                )
            ]
            for q, label in self._QUERIES
        }

    def test_concurrent_rerank_matches_sequential_baseline(self, reranker):
        import threading

        baseline = self._baseline_orderings(reranker)

        # Run each query in many threads simultaneously; the ordering
        # observed by every thread must match the single-threaded
        # baseline for the same query.
        threads_per_query = 4
        results: dict[tuple[str, int], list[str]] = {}
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker(query: str, label: str, idx: int) -> None:
            try:
                ordered = reranker.rerank(
                    query_text=query,
                    candidates=list(self._CANDS),
                    get_text=lambda c: c["text"],
                )
                ids = [c["id"] for c in ordered]
                with lock:
                    results[(label, idx)] = ids
            except BaseException as exc:  # noqa: BLE001 — surface any race-induced error
                with lock:
                    errors.append(exc)

        threads: list[threading.Thread] = []
        for query, label in self._QUERIES:
            for idx in range(threads_per_query):
                threads.append(
                    threading.Thread(target=worker, args=(query, label, idx))
                )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"races raised: {errors}"
        for query, label in self._QUERIES:
            expected = baseline[label]
            for idx in range(threads_per_query):
                assert results[(label, idx)] == expected, (
                    f"thread {idx} for {label!r} drifted: "
                    f"{results[(label, idx)]} != {expected}"
                )
