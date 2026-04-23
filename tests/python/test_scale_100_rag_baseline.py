import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from corpus_100 import ALL_QUERIES, MEMORIES
from scale_100_rag_baseline import (
    CosineRankingStrategy,
    QueryCase,
    RankedResult,
    compute_metrics,
    l2_normalize,
    load_query_cases,
    mean_pool,
)


def test_embedding_mean_pooling_uses_attention_mask():
    hidden = torch.tensor(
        [
            [
                [1.0, 1.0],
                [3.0, 3.0],
                [100.0, 100.0],
            ]
        ]
    )
    mask = torch.tensor([[1, 1, 0]])

    pooled = mean_pool(hidden, mask)

    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_cosine_ranking_orders_by_similarity():
    memories = l2_normalize(np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.2],
    ], dtype=np.float32))
    ranker = CosineRankingStrategy(memories)

    ranked = ranker.rank(np.array([1.0, 0.0], dtype=np.float32))

    assert [r.cell_id for r in ranked] == [0, 2, 1]


def test_rag_metrics_match_known_rankings():
    queries = [
        QueryCase(0, "q0", (2,), "cross"),
        QueryCase(1, "q1", (3,), "within"),
        QueryCase(2, "q2", (), "negative"),
    ]
    rankings = {
        0: [
            RankedResult(1, 0.9),
            RankedResult(2, 0.8),
            RankedResult(3, 0.1),
        ],
        1: [
            RankedResult(3, 0.95),
            RankedResult(4, 0.2),
            RankedResult(5, 0.1),
        ],
        2: [
            RankedResult(4, 0.3),
            RankedResult(2, 0.2),
            RankedResult(3, 0.1),
        ],
    }

    metrics = compute_metrics(rankings, queries)

    assert math.isclose(metrics.recall_at[1], 0.5)
    assert math.isclose(metrics.recall_at[3], 1.0)
    assert math.isclose(metrics.recall_at[5], 1.0)
    assert math.isclose(metrics.recall_at[10], 1.0)
    assert math.isclose(metrics.mrr, 0.75)
    assert metrics.unique_top1 == 2
    assert metrics.worst_top1_count == 1
    assert metrics.score_gap > 0


def test_rag_baseline_uses_same_corpus_shape():
    queries = load_query_cases()

    assert len(MEMORIES) == 100
    assert len(ALL_QUERIES) == 40
    assert len([q for q in queries if q.qtype != "negative"]) == 30
    assert len([q for q in queries if q.qtype == "negative"]) == 10
