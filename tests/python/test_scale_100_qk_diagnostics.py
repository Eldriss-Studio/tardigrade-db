import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from scale_100_qk_diagnostics import (
    DecisionReport,
    HeadSweepResult,
    LayerSweepResult,
    Metrics,
    RankingEntry,
    TensorItem,
    best_pair_trace,
    build_decision_report,
    compare_projection_metrics,
    compute_metrics,
    compute_oracle_result,
    compute_rank_depth,
    cosine_sum_max_score,
    expected_rank,
    gravity_stats,
    layer_fraction_to_index,
    order_heads_by_metric,
    projection_plan,
    random_recall_at_k,
    select_best_layer,
)


def test_random_recall_at_k_matches_topk_probability():
    assert math.isclose(random_recall_at_k(100, 1, 5), 0.05)

    expected_two_targets = 1.0 - math.comb(98, 5) / math.comb(100, 5)
    assert math.isclose(random_recall_at_k(100, 2, 5), expected_two_targets)


def test_gravity_stats_flags_repeated_top1_collapse():
    unique, worst_cell, worst_count = gravity_stats([87, 87, 6, 87, 10, 10])

    assert unique == 3
    assert worst_cell == 87
    assert worst_count == 3


def test_best_pair_trace_returns_responsible_tokens():
    scores = np.array([[0.1, 0.2], [0.3, 0.9]], dtype=np.float32)

    trace = best_pair_trace(scores, ["where", "flat"], ["car", "tire"])

    assert trace.query_token == "flat"
    assert trace.memory_token == "tire"
    assert trace.query_index == 1
    assert trace.memory_index == 1
    assert math.isclose(trace.score, 0.9, rel_tol=1e-6)


def test_cosine_sum_max_ignores_vector_magnitude():
    query = TensorItem(
        0,
        "query",
        ["aligned"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    high_magnitude_wrong = TensorItem(
        1,
        "wrong",
        ["loud"],
        np.array([[10.0, 100.0]], dtype=np.float32),
    )
    aligned = TensorItem(
        2,
        "aligned",
        ["quiet"],
        np.array([[0.5, 0.0]], dtype=np.float32),
    )

    wrong_score = cosine_sum_max_score(query, high_magnitude_wrong)
    aligned_score = cosine_sum_max_score(query, aligned)

    assert aligned_score.score > wrong_score.score


def test_cosine_sum_max_prefers_broad_matches():
    query = TensorItem(
        0,
        "query",
        ["alpha", "beta", "gamma", "delta"],
        np.eye(4, dtype=np.float32),
    )
    broad_match = TensorItem(
        1,
        "broad",
        ["a", "b", "c", "d"],
        np.eye(4, dtype=np.float32),
    )
    single_spike = TensorItem(
        2,
        "spike",
        ["a"],
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

    broad_score = cosine_sum_max_score(query, broad_match)
    spike_score = cosine_sum_max_score(query, single_spike)

    assert broad_score.score > spike_score.score


def test_cosine_sum_max_trace_reports_best_pair():
    query = TensorItem(
        0,
        "query",
        ["horizontal", "vertical"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    memory = TensorItem(
        1,
        "memory",
        ["vertical_memory", "mixed"],
        np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    )

    result = cosine_sum_max_score(query, memory)

    assert result.trace is not None
    assert result.trace.query_token == "vertical"
    assert result.trace.memory_token == "vertical_memory"
    assert result.trace.query_index == 1
    assert result.trace.memory_index == 0
    assert math.isclose(result.trace.score, 1.0, rel_tol=1e-6)


def test_compute_metrics_on_deterministic_rankings():
    queries = [
        TensorItem(0, "q0", [], np.zeros((1, 2), dtype=np.float32), expected=(2,), qtype="cross"),
        TensorItem(1, "q1", [], np.zeros((1, 2), dtype=np.float32), expected=(3,), qtype="within"),
        TensorItem(2, "q2", [], np.zeros((1, 2), dtype=np.float32), expected=(), qtype="negative"),
    ]
    rankings = {
        0: [
            RankingEntry(1, 1.0),
            RankingEntry(2, 0.8),
            RankingEntry(3, 0.1),
        ],
        1: [
            RankingEntry(3, 0.9),
            RankingEntry(4, 0.2),
            RankingEntry(5, 0.1),
        ],
        2: [
            RankingEntry(4, 0.5),
            RankingEntry(2, 0.4),
            RankingEntry(3, 0.3),
        ],
    }

    metrics = compute_metrics(rankings, queries, num_memories=6, topks=(1, 3, 5, 10))

    assert math.isclose(metrics.recall_at[1], 0.5)
    assert math.isclose(metrics.recall_at[3], 1.0)
    assert math.isclose(metrics.recall_at[5], 1.0)
    assert math.isclose(metrics.mrr, 0.75)
    assert metrics.unique_top1 == 2
    assert metrics.worst_top1_count == 1
    assert metrics.negative_false_positive_rate == 0.0
    assert metrics.score_gap > 0.0


def test_expected_rank_finds_first_matching_memory():
    ranking = [RankingEntry(5, 1.0), RankingEntry(8, 0.8), RankingEntry(13, 0.4)]

    assert expected_rank(ranking, (8, 13)) == 2


def test_depth_recall_counts_hits_at_multiple_cutoffs():
    queries = [
        TensorItem(0, "q0", [], np.zeros((1, 2), dtype=np.float32), expected=(10,), qtype="cross"),
        TensorItem(1, "q1", [], np.zeros((1, 2), dtype=np.float32), expected=(20,), qtype="within"),
    ]
    rankings = {
        0: [RankingEntry(i, float(100 - i)) for i in range(60)],
        1: [RankingEntry(i, float(100 - i)) for i in range(60)],
    }

    depth = compute_rank_depth(rankings, queries, cutoffs=(10, 20, 50))

    assert math.isclose(depth.recall_at[10], 0.0)
    assert math.isclose(depth.recall_at[20], 0.5)
    assert math.isclose(depth.recall_at[50], 1.0)
    assert depth.median_rank == 16.0
    assert depth.worst_rank == 21


def test_missing_expected_memory_reports_none():
    ranking = [RankingEntry(1, 1.0), RankingEntry(2, 0.8)]

    assert expected_rank(ranking, (99,)) is None


def test_oracle_recall_uses_best_available_scorer():
    queries = [
        TensorItem(0, "q0", [], np.zeros((1, 2), dtype=np.float32), expected=(1,)),
        TensorItem(1, "q1", [], np.zeros((1, 2), dtype=np.float32), expected=(2,)),
    ]
    rankings = {
        "a": {
            0: [RankingEntry(1, 1.0), RankingEntry(9, 0.1)],
            1: [RankingEntry(9, 1.0), RankingEntry(8, 0.1)],
        },
        "b": {
            0: [RankingEntry(9, 1.0), RankingEntry(8, 0.1)],
            1: [RankingEntry(2, 1.0), RankingEntry(9, 0.1)],
        },
    }

    oracle = compute_oracle_result(rankings, queries, topks=(1, 5))

    assert math.isclose(oracle.recall_at[1], 1.0)
    assert math.isclose(oracle.recall_at[5], 1.0)


def test_oracle_records_rescuing_scorer():
    query = TensorItem(0, "q0", [], np.zeros((1, 2), dtype=np.float32), expected=(7,))
    rankings = {
        "miss": {0: [RankingEntry(1, 1.0), RankingEntry(2, 0.8)]},
        "hit": {0: [RankingEntry(7, 0.9), RankingEntry(3, 0.1)]},
    }

    oracle = compute_oracle_result(rankings, [query])

    assert oracle.rescued_by_query[0] == "hit"


def test_oracle_unrescued_queries_are_reported():
    query = TensorItem(0, "q0", [], np.zeros((1, 2), dtype=np.float32), expected=(7,))
    rankings = {
        "miss": {0: [RankingEntry(1, 1.0), RankingEntry(2, 0.8)]},
    }

    oracle = compute_oracle_result(rankings, [query])

    assert oracle.rescued_by_query[0] is None
    assert oracle.unrescued_query_ids == [0]


def test_layer_fraction_to_index_is_stable():
    assert layer_fraction_to_index(28, 0.67) == 18
    assert layer_fraction_to_index(28, 0.0) == 0
    assert layer_fraction_to_index(28, 1.0) == 27


def test_layer_sweep_selects_best_result_by_recall():
    weaker = LayerSweepResult(
        layer_fraction=0.5,
        layer_idx=14,
        best_scorer="a",
        metrics=Metrics({5: 0.6}, 0.7, 10, 1, 2, 0.0, 1.0, 0.0),
    )
    stronger = LayerSweepResult(
        layer_fraction=0.67,
        layer_idx=18,
        best_scorer="b",
        metrics=Metrics({5: 0.6}, 0.8, 11, 2, 1, 0.0, 1.0, 0.0),
    )

    assert select_best_layer([weaker, stronger]) == stronger


def test_head_report_orders_heads_by_metric():
    heads = [
        HeadSweepResult(0, Metrics({5: 0.2}, 0.4, 2, 1, 3, 0.0, 1.0, 0.0)),
        HeadSweepResult(1, Metrics({5: 0.8}, 0.7, 4, 2, 1, 0.0, 1.0, 0.0)),
    ]

    assert [h.head_index for h in order_heads_by_metric(heads)] == [1, 0]


def test_projection_mode_qk_uses_query_q_and_memory_k():
    plan = projection_plan("qk")

    assert plan.query_projection == "q"
    assert plan.memory_projection == "k"


def test_projection_mode_hidden_uses_same_representation_for_both():
    plan = projection_plan("hidden")

    assert plan.query_projection == "hidden"
    assert plan.memory_projection == "hidden"


def test_projection_comparison_reports_delta():
    qk = Metrics({5: 0.4}, 0.5, 10, 1, 4, 0.8, 1.0, 0.0)
    hidden = Metrics({5: 0.7}, 0.6, 12, 2, 2, 0.2, 1.0, 0.0)

    comparison = compare_projection_metrics("qk", qk, "hidden", hidden)

    assert math.isclose(comparison.recall_at_5_delta, 0.3)
    assert comparison.gravity_delta == -2
    assert math.isclose(comparison.negative_fp_delta, -0.6)


def test_decision_report_flags_scoring_problem():
    report = build_decision_report(
        current=Metrics({5: 0.45}, 0.4, 10, 1, 5, 0.5, 1.0, 0.0),
        rank_depth_recall_at_50=0.9,
        oracle=Metrics({5: 0.75}, 0.7, 20, 1, 2, 0.1, 1.0, 0.0),
    )

    assert report.verdict == "SCORING_PROBLEM"


def test_decision_report_flags_layer_problem():
    report = build_decision_report(
        current=Metrics({5: 0.45}, 0.4, 10, 1, 5, 0.5, 1.0, 0.0),
        best_layer=Metrics({5: 0.62}, 0.6, 16, 1, 4, 0.4, 1.0, 0.0),
    )

    assert report.verdict == "LAYER_OR_HEAD_PROBLEM"


def test_decision_report_flags_qk_specific_problem():
    report = build_decision_report(
        current=Metrics({5: 0.4}, 0.4, 10, 1, 5, 0.5, 1.0, 0.0),
        hidden=Metrics({5: 0.75}, 0.7, 20, 1, 2, 0.1, 1.0, 0.0),
    )

    assert report.verdict == "QK_SPECIFIC_PROBLEM"


def test_decision_report_flags_latent_signal_weak():
    report = build_decision_report(
        current=Metrics({5: 0.4}, 0.4, 10, 1, 5, 0.5, 1.0, 0.0),
        rank_depth_recall_at_50=0.45,
        oracle=Metrics({5: 0.48}, 0.45, 11, 1, 5, 0.5, 1.0, 0.0),
        hidden=Metrics({5: 0.5}, 0.45, 11, 1, 5, 0.5, 1.0, 0.0),
    )

    assert report.verdict == "LATENT_SIGNAL_WEAK"


def test_decision_report_flags_ready_for_rust():
    report = build_decision_report(
        current=Metrics({5: 0.72}, 0.7, 20, 1, 3, 0.05, 1.0, 0.0),
        current_baseline_recall_at_5=0.633,
        deterministic_tensors=True,
    )

    assert report.verdict == "READY_FOR_RUST_EXPERIMENT"
    assert isinstance(report, DecisionReport)
