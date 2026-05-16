"""Recall@k and NDCG@k over retrieved evidence.

The bench runner computes the LLM-Judge score for every row. That
score conflates retrieval quality with answerer quality — a perfect
retriever paired with a refusing LLM scores 0; a hallucinating LLM
fed garbage retrieval can score 1. To break the dependency the
runner also computes pure retrieval metrics whenever an item carries
``gold_evidence``. Those metrics are audit-resistant: they depend
only on the retriever and the test data, not the answerer.

Relevance is binary text-overlap. A retrieved chunk is "relevant"
for a gold snippet ``g`` iff ``g`` (after whitespace strip,
case-folded) appears as a substring of the chunk. Each gold snippet
contributes to the recall numerator at most once across all
retrieved chunks (set-cover semantics).

This is intentionally coarser than dia_id matching — we don't have
chunk→dia_id provenance plumbed end-to-end, and the text-overlap
proxy is what ENGRAM (Lewis et al., 2023) and dial481/locomo-audit
recommend as a chunk-friendly stand-in.

The function is pure: no IO, no model state, deterministic for a
given input. The bench runner calls it once per (item, system) row
and inlines the result as ``row["retrieval_metrics"]``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


_DEFAULT_KS: tuple[int, ...] = (1, 5, 10)


def compute_retrieval_metrics(
    *,
    gold: Sequence[str],
    retrieved: Sequence[str],
    ks: Iterable[int] = _DEFAULT_KS,
) -> dict[str, float]:
    """Return Recall@k and NDCG@k for binary-relevance text overlap.

    Args:
        gold: Gold evidence snippets. Whitespace-only entries are
            silently dropped — they contribute no signal.
        retrieved: Retrieved chunk texts in descending-rank order.
        ks: k-values to score. Defaults to (1, 5, 10).

    Returns:
        ``{"recall@k": ..., "ndcg@k": ...}`` for each ``k``.

        When ``gold`` (after the blank-strip) is empty, every metric
        is ``float("nan")`` so callers can skip the row in averages
        instead of counting it as perfect-zero.

        When ``gold`` is non-empty but ``retrieved`` is empty, every
        metric is 0.0 — the retriever genuinely missed.
    """
    ks_tuple = tuple(sorted(set(int(k) for k in ks)))
    gold_clean = [g.strip().lower() for g in gold if g and g.strip()]
    retrieved_lc = [r.lower() for r in retrieved]

    if not gold_clean:
        return {
            **{f"recall@{k}": float("nan") for k in ks_tuple},
            **{f"ndcg@{k}": float("nan") for k in ks_tuple},
        }

    if not retrieved_lc:
        return {
            **{f"recall@{k}": 0.0 for k in ks_tuple},
            **{f"ndcg@{k}": 0.0 for k in ks_tuple},
        }

    # Per-position relevance flag: 1 if this retrieved chunk contains
    # at least one (still-uncovered) gold snippet. Covered set has
    # set-cover semantics for recall — each gold counts once total.
    relevance: list[int] = []
    covered: set[int] = set()
    for chunk in retrieved_lc:
        hit_this_position = False
        for gi, g in enumerate(gold_clean):
            if gi in covered:
                continue
            if g in chunk:
                covered.add(gi)
                hit_this_position = True
        relevance.append(1 if hit_this_position else 0)

    out: dict[str, float] = {}
    n_gold = len(gold_clean)
    for k in ks_tuple:
        prefix = relevance[:k]
        # Recall counts distinct gold covered within top-k (recompute
        # rather than reusing `covered` so cap-at-k semantics are
        # honored — see TestRecallAtK.test_recall_caps_at_k).
        recall_covered: set[int] = set()
        for chunk in retrieved_lc[:k]:
            for gi, g in enumerate(gold_clean):
                if gi in recall_covered:
                    continue
                if g in chunk:
                    recall_covered.add(gi)
        out[f"recall@{k}"] = len(recall_covered) / n_gold

        dcg = sum(
            rel / math.log2(pos + 2)  # pos is 0-indexed; rank = pos+1; log2(rank+1)
            for pos, rel in enumerate(prefix)
        )
        ideal_hits = min(n_gold, k)
        idcg = sum(1.0 / math.log2(pos + 2) for pos in range(ideal_hits))
        out[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0 else 0.0

    return out
