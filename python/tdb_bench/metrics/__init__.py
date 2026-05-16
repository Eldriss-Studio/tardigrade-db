"""Retrieval and answer metrics for the bench runner.

Currently exposes :func:`retrieval.compute_retrieval_metrics` for
audit-resistant retrieval-only headlines (Recall@k, NDCG@k) computed
in parallel with the LLM-Judge score.
"""

from tdb_bench.metrics.retrieval import compute_retrieval_metrics

__all__ = ["compute_retrieval_metrics"]
