"""ATDD: retrieval metrics — Recall@k, NDCG@k.

Pins the metric math used by the audit-resistant retrieval-only
headline (#88). The runner attaches the output of
``compute_retrieval_metrics`` to every row whose item has
``gold_evidence`` set, so the LLM-Judge score and the retrieval-only
score can be compared side-by-side.

Relevance is binary text-overlap: a retrieved chunk is "relevant"
when at least one gold-evidence snippet appears as a substring.
The denominator for recall is the number of distinct gold snippets
*covered* — each gold snippet counts at most once across all
retrieved chunks.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tdb_bench.metrics.retrieval import compute_retrieval_metrics  # noqa: E402


class TestRecallAtK:
    def test_perfect_recall_when_all_gold_in_top1(self):
        m = compute_retrieval_metrics(
            gold=["alice moved to berlin"],
            retrieved=["alice moved to berlin in 2021"],
            ks=(1, 5, 10),
        )
        assert m["recall@1"] == 1.0
        assert m["recall@5"] == 1.0
        assert m["recall@10"] == 1.0

    def test_zero_recall_when_no_overlap(self):
        m = compute_retrieval_metrics(
            gold=["alice moved to berlin"],
            retrieved=["the weather in paris is cloudy"],
            ks=(1, 5),
        )
        assert m["recall@1"] == 0.0
        assert m["recall@5"] == 0.0

    def test_partial_recall_counts_distinct_gold(self):
        # Two gold snippets; only one matched → recall = 0.5.
        m = compute_retrieval_metrics(
            gold=["alice moved", "bob is in paris"],
            retrieved=["alice moved to berlin"],
            ks=(5,),
        )
        assert m["recall@5"] == 0.5

    def test_recall_caps_at_k(self):
        # 3 gold, but only top-1 considered; only the first retrieved
        # chunk can cover gold.
        m = compute_retrieval_metrics(
            gold=["a", "b", "c"],
            retrieved=["text a here", "text b here", "text c here"],
            ks=(1, 3),
        )
        assert m["recall@1"] == pytest.approx(1 / 3)
        assert m["recall@3"] == 1.0

    def test_duplicate_gold_match_counts_once(self):
        # Two retrieved chunks both cover the same gold → still 1/1.
        m = compute_retrieval_metrics(
            gold=["alice moved"],
            retrieved=["alice moved here", "alice moved there"],
            ks=(5,),
        )
        assert m["recall@5"] == 1.0


class TestNDCGAtK:
    def test_ndcg_perfect_when_relevant_at_top(self):
        m = compute_retrieval_metrics(
            gold=["alice moved"],
            retrieved=["alice moved to berlin", "irrelevant filler"],
            ks=(5,),
        )
        assert m["ndcg@5"] == 1.0

    def test_ndcg_penalizes_lower_ranked_relevant(self):
        # Relevant at position 2: DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
        m = compute_retrieval_metrics(
            gold=["alice moved"],
            retrieved=["irrelevant", "alice moved to berlin"],
            ks=(5,),
        )
        expected = (1.0 / math.log2(3)) / 1.0
        assert m["ndcg@5"] == pytest.approx(expected, abs=1e-6)

    def test_ndcg_is_zero_when_no_relevant(self):
        m = compute_retrieval_metrics(
            gold=["alice moved"],
            retrieved=["paris", "berlin", "rome"],
            ks=(5,),
        )
        assert m["ndcg@5"] == 0.0


class TestEdgeCases:
    def test_empty_gold_returns_nan_metrics(self):
        # No gold = no signal; report NaN so the row aggregator can skip
        # rather than counting these as perfect-zero (which would be
        # misleading after averaging).
        m = compute_retrieval_metrics(gold=[], retrieved=["a"], ks=(5,))
        assert math.isnan(m["recall@5"])
        assert math.isnan(m["ndcg@5"])

    def test_empty_retrieved_returns_zero(self):
        # Adapter returned nothing — that's a real-zero, not undefined.
        m = compute_retrieval_metrics(
            gold=["alice"], retrieved=[], ks=(5,),
        )
        assert m["recall@5"] == 0.0
        assert m["ndcg@5"] == 0.0

    def test_blank_gold_snippets_ignored(self):
        # Whitespace-only gold contributes nothing to the denominator.
        m = compute_retrieval_metrics(
            gold=["alice moved", "   "],
            retrieved=["alice moved to berlin"],
            ks=(5,),
        )
        assert m["recall@5"] == 1.0

    def test_case_insensitive_matching(self):
        # LoCoMo turn text capitalization is inconsistent — matching
        # must be case-insensitive to be useful as a retrieval metric.
        m = compute_retrieval_metrics(
            gold=["Alice Moved"],
            retrieved=["alice moved to berlin"],
            ks=(5,),
        )
        assert m["recall@5"] == 1.0


class TestReturnShape:
    def test_returns_one_recall_and_ndcg_per_k(self):
        m = compute_retrieval_metrics(
            gold=["a"], retrieved=["a"], ks=(1, 5, 10),
        )
        assert set(m.keys()) == {
            "recall@1", "recall@5", "recall@10",
            "ndcg@1", "ndcg@5", "ndcg@10",
        }
