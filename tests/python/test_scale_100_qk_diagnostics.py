import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from scale_100_qk_diagnostics import (
    RankingEntry,
    TensorItem,
    best_pair_trace,
    compute_metrics,
    gravity_stats,
    random_recall_at_k,
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
