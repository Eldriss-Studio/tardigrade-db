"""ATDD: per-(dataset, category) aggregation in the bench runner.

LoCoMo single_hop, multi_hop, temporal, open_domain, adversarial are
different abilities — system A may beat system B overall while losing
the temporal slice. The runner must surface that split so the audit
doc can attribute movement to a real cause. Same shape applies to
LongMemEval's ``question_type``.

Pin the contract on the static aggregation helper directly so the
test is hermetic — no adapter or evaluator wiring needed. the bench audit
audit 2026-05-16 #89.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tdb_bench.runner import BenchmarkRunner  # noqa: E402


_SYS = "sys-under-test"


def _row(
    *,
    dataset: str,
    category: str,
    score: float,
    status: str = "ok",
    system: str = _SYS,
) -> dict:
    return {
        "item_id": f"{dataset}-{category}-{score}",
        "dataset": dataset,
        "category": category,
        "system": system,
        "status": status,
        "score": score,
    }


class TestPerCategoryBreakdownGrouping:
    def test_groups_keyed_by_dataset_slash_category(self):
        items = [
            _row(dataset="locomo", category="single_hop", score=1.0),
            _row(dataset="locomo", category="single_hop", score=0.0),
            _row(dataset="locomo", category="temporal", score=1.0),
        ]
        out = BenchmarkRunner._per_category_breakdown(items, _SYS)
        assert set(out.keys()) == {"locomo/single_hop", "locomo/temporal"}

    def test_avg_score_only_counts_ok_status(self):
        items = [
            _row(dataset="locomo", category="single_hop", score=1.0),
            _row(dataset="locomo", category="single_hop", score=0.0, status="failed"),
        ]
        out = BenchmarkRunner._per_category_breakdown(items, _SYS)
        # n counts everything, ok counts ok-status only, avg averages ok.
        assert out["locomo/single_hop"]["n"] == 2
        assert out["locomo/single_hop"]["ok"] == 1
        assert out["locomo/single_hop"]["avg_score"] == 1.0

    def test_locomo_temporal_and_longmemeval_temporal_reasoning_are_separate(self):
        # Bug guard: the prefix must keep different-dataset slices
        # apart even when they share a category name suffix.
        items = [
            _row(dataset="locomo", category="temporal", score=1.0),
            _row(dataset="longmemeval", category="temporal-reasoning", score=0.0),
        ]
        out = BenchmarkRunner._per_category_breakdown(items, _SYS)
        assert "locomo/temporal" in out
        assert "longmemeval/temporal-reasoning" in out
        assert out["locomo/temporal"]["avg_score"] == 1.0
        assert out["longmemeval/temporal-reasoning"]["avg_score"] == 0.0

    def test_filters_out_other_systems(self):
        items = [
            _row(dataset="locomo", category="single_hop", score=1.0, system=_SYS),
            _row(dataset="locomo", category="single_hop", score=0.0, system="other"),
        ]
        out = BenchmarkRunner._per_category_breakdown(items, _SYS)
        # Only the focus system's rows count.
        assert out["locomo/single_hop"]["n"] == 1
        assert out["locomo/single_hop"]["avg_score"] == 1.0

    def test_missing_category_falls_back_to_unknown(self):
        items = [
            {
                "item_id": "x",
                "dataset": "locomo",
                "system": _SYS,
                "status": "ok",
                "score": 0.5,
                # No "category" key.
            },
        ]
        out = BenchmarkRunner._per_category_breakdown(items, _SYS)
        assert "locomo/unknown" in out

    def test_empty_input_returns_empty_dict(self):
        assert BenchmarkRunner._per_category_breakdown([], _SYS) == {}


class TestRetrievalAggregateAttachedToSystemsPayload:
    """Pin the audit-resistant retrieval headline (#88) plumbing:
    the system payload carries a ``retrieval`` block with
    mean(recall@k) and mean(ndcg@k), skipping gold-less rows."""

    @staticmethod
    def _row_with_metrics(score: float, metrics: dict | None) -> dict:
        return {
            "item_id": f"r-{score}",
            "dataset": "locomo",
            "category": "single_hop",
            "system": _SYS,
            "status": "ok",
            "score": score,
            "retrieval_metrics": metrics,
        }

    def test_retrieval_aggregate_excludes_nan_rows(self):
        from tdb_bench.runner import BenchmarkRunner
        items = [
            self._row_with_metrics(
                1.0, {"recall@5": 1.0, "ndcg@5": 1.0},
            ),
            self._row_with_metrics(
                1.0, {"recall@5": 0.0, "ndcg@5": 0.0},
            ),
            self._row_with_metrics(
                1.0, {"recall@5": float("nan"), "ndcg@5": float("nan")},
            ),
        ]
        agg = BenchmarkRunner._retrieval_aggregate(items, _SYS)
        # Two scored rows; NaN one skipped.
        assert agg["n"] == 2
        assert agg["recall@5"] == 0.5
        assert agg["ndcg@5"] == 0.5

    def test_retrieval_aggregate_handles_no_scored_rows(self):
        from tdb_bench.runner import BenchmarkRunner
        items = [
            self._row_with_metrics(
                1.0, {"recall@5": float("nan"), "ndcg@5": float("nan")},
            ),
        ]
        agg = BenchmarkRunner._retrieval_aggregate(items, _SYS)
        assert agg["n"] == 0

    def test_retrieval_aggregate_filters_other_systems(self):
        from tdb_bench.runner import BenchmarkRunner
        items = [
            self._row_with_metrics(1.0, {"recall@5": 1.0, "ndcg@5": 1.0}),
            {**self._row_with_metrics(0.0, {"recall@5": 0.0, "ndcg@5": 0.0}),
             "system": "other"},
        ]
        agg = BenchmarkRunner._retrieval_aggregate(items, _SYS)
        assert agg["n"] == 1
        assert agg["recall@5"] == 1.0


class TestAnswerTextRetrievalAggregate:
    """research recommendation #101: a parallel
    answer-text retrieval metric that substring-matches the
    BenchmarkItem.ground_truth against retrieved chunks, rather
    than evidence text. Predicts LLM-Judge better than the
    evidence-text metric because the answer text is what the
    LLM needs in its window."""

    @staticmethod
    def _row_with_answer(score: float, metrics: dict | None) -> dict:
        return {
            "item_id": f"ar-{score}",
            "dataset": "locomo",
            "category": "single_hop",
            "system": _SYS,
            "status": "ok",
            "score": score,
            "answer_text_metrics": metrics,
        }

    def test_answer_text_aggregate_averages_present_metrics(self):
        from tdb_bench.runner import BenchmarkRunner
        items = [
            self._row_with_answer(1.0, {"recall@5": 1.0, "ndcg@5": 1.0}),
            self._row_with_answer(0.0, {"recall@5": 0.0, "ndcg@5": 0.0}),
        ]
        agg = BenchmarkRunner._answer_text_aggregate(items, _SYS)
        assert agg["n"] == 2
        assert agg["recall@5"] == 0.5
        assert agg["ndcg@5"] == 0.5

    def test_answer_text_aggregate_excludes_nan(self):
        from tdb_bench.runner import BenchmarkRunner
        items = [
            self._row_with_answer(1.0, {"recall@5": 1.0, "ndcg@5": 1.0}),
            self._row_with_answer(
                1.0, {"recall@5": float("nan"), "ndcg@5": float("nan")},
            ),
        ]
        agg = BenchmarkRunner._answer_text_aggregate(items, _SYS)
        assert agg["n"] == 1
        assert agg["recall@5"] == 1.0

    def test_answer_text_aggregate_filters_other_systems(self):
        from tdb_bench.runner import BenchmarkRunner
        row_us = self._row_with_answer(1.0, {"recall@5": 1.0, "ndcg@5": 1.0})
        row_other = {**row_us, "system": "other"}
        agg = BenchmarkRunner._answer_text_aggregate([row_us, row_other], _SYS)
        assert agg["n"] == 1


class TestPerCategoryBreakdownAttachedToSystemsPayload:
    """End-to-end check that `_aggregates` wires `by_category` into
    each system's payload."""

    def test_systems_payload_carries_by_category(self):
        items = [
            _row(dataset="locomo", category="single_hop", score=1.0),
            _row(dataset="locomo", category="temporal", score=0.0),
        ]
        # _aggregates is an instance method but doesn't touch self
        # state; construct a runner only enough to call it.
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        agg = runner._aggregates(items)
        assert "by_category" in agg["systems"][_SYS]
        breakdown = agg["systems"][_SYS]["by_category"]
        assert breakdown["locomo/single_hop"]["avg_score"] == 1.0
        assert breakdown["locomo/temporal"]["avg_score"] == 0.0
