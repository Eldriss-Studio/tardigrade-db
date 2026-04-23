#!/usr/bin/env python3
"""Traditional embedding RAG baseline for the 100-memory corpus.

Measurement-only experiment: this does not change TardigradeDB retrieval.
It compares text-memory + embedding retrieval against the same corpus used by
the Q*K 100-memory test.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(".").resolve() / "experiments"))

from corpus_100 import ALL_QUERIES, MEMORIES

MODEL_NAME = "intfloat/e5-small-v2"
DOMAINS = [
    "Work",
    "Parenting",
    "Cooking",
    "Health",
    "Legal",
    "Social",
    "Fitness",
    "Dreams",
    "Errands",
    "Media",
]


@dataclass(frozen=True)
class RankedResult:
    cell_id: int
    score: float


@dataclass(frozen=True)
class QueryCase:
    item_id: int
    text: str
    expected: tuple[int, ...]
    qtype: str


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at: dict[int, float]
    mrr: float
    unique_top1: int
    worst_top1_cell: int
    worst_top1_count: int
    negative_top_score_mean: float
    score_gap: float


class EmbeddingModelAdapter:
    """Adapter pattern: hides HuggingFace embedding-model mechanics."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self.model(**inputs)
            pooled = mean_pool(output.last_hidden_state, inputs["attention_mask"])
            embeddings.append(l2_normalize(pooled.detach().cpu().numpy().astype(np.float32)))
        return np.concatenate(embeddings, axis=0)


class CosineRankingStrategy:
    """Strategy pattern: ranks memories by cosine over normalized embeddings."""

    def __init__(self, memory_embeddings: np.ndarray):
        self.memory_embeddings = l2_normalize(memory_embeddings.astype(np.float32))

    def rank(self, query_embedding: np.ndarray) -> list[RankedResult]:
        query = l2_normalize(query_embedding.reshape(1, -1).astype(np.float32))[0]
        scores = self.memory_embeddings @ query
        order = np.argsort(-scores)
        return [RankedResult(int(idx), float(scores[idx])) for idx in order]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def l2_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, eps)


def load_query_cases() -> list[QueryCase]:
    return [
        QueryCase(idx, text, tuple(expected), qtype)
        for idx, (text, expected, qtype) in enumerate(ALL_QUERIES)
    ]


def reciprocal_rank(ranking: list[int], expected: tuple[int, ...]) -> float:
    expected_set = set(expected)
    for rank, cell_id in enumerate(ranking, start=1):
        if cell_id in expected_set:
            return 1.0 / rank
    return 0.0


def compute_metrics(
    rankings: dict[int, list[RankedResult]],
    queries: list[QueryCase],
    topks: tuple[int, ...] = (1, 3, 5, 10),
) -> RetrievalMetrics:
    positives = [q for q in queries if q.qtype != "negative"]
    negatives = [q for q in queries if q.qtype == "negative"]

    recall_at = {}
    for k in topks:
        hits = 0
        for query in positives:
            ranked_ids = [r.cell_id for r in rankings[query.item_id]]
            if set(ranked_ids[:k]).intersection(query.expected):
                hits += 1
        recall_at[k] = hits / max(len(positives), 1)

    mrr = float(np.mean([
        reciprocal_rank([r.cell_id for r in rankings[q.item_id]], q.expected)
        for q in positives
    ]))

    top1 = [rankings[q.item_id][0].cell_id for q in positives]
    counts = Counter(top1)
    worst_cell, worst_count = counts.most_common(1)[0] if counts else (-1, 0)

    expected_scores = []
    for query in positives:
        score_by_id = {r.cell_id: r.score for r in rankings[query.item_id]}
        expected_scores.append(max(score_by_id[e] for e in query.expected if e in score_by_id))

    negative_scores = [
        rankings[q.item_id][0].score
        for q in negatives
        if rankings[q.item_id]
    ]
    neg_mean = float(np.mean(negative_scores)) if negative_scores else 0.0
    score_gap = float(np.mean(expected_scores) - neg_mean) if expected_scores else 0.0

    return RetrievalMetrics(
        recall_at=recall_at,
        mrr=mrr,
        unique_top1=len(counts),
        worst_top1_cell=worst_cell,
        worst_top1_count=worst_count,
        negative_top_score_mean=neg_mean,
        score_gap=score_gap,
    )


def domain_for_expected(expected: tuple[int, ...]) -> int:
    return expected[0] // 10


def print_report(
    rankings: dict[int, list[RankedResult]],
    queries: list[QueryCase],
    metrics: RetrievalMetrics,
    encode_time: float,
    rank_latency_ms: list[float],
) -> None:
    positives = [q for q in queries if q.qtype != "negative"]
    cross = [q for q in positives if q.qtype == "cross"]
    within = [q for q in positives if q.qtype == "within"]

    def hits(items: list[QueryCase], k: int = 5) -> int:
        total = 0
        for query in items:
            ranked_ids = [r.cell_id for r in rankings[query.item_id]]
            total += bool(set(ranked_ids[:k]).intersection(query.expected))
        return total

    print(f"\n{'=' * 70}")
    print("  TRADITIONAL EMBEDDING RAG RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  Cross-domain recall@5:   {hits(cross)}/{len(cross)} ({100 * hits(cross) / len(cross):.1f}%)")
    print(f"  Within-domain recall@5:  {hits(within)}/{len(within)} ({100 * hits(within) / len(within):.1f}%)")
    for k, value in metrics.recall_at.items():
        print(f"  Overall recall@{k:<2}:      {value * 100:.1f}%")
    print(f"  MRR:                       {metrics.mrr:.3f}")
    print(f"\n  Unique top-1 memories:     {metrics.unique_top1}/{len(positives)}")
    print(f"  Worst gravity well:        mem {metrics.worst_top1_cell} ({metrics.worst_top1_count}x top-1)")
    print(f"  Negative top-score mean:   {metrics.negative_top_score_mean:.4f}")
    print(f"  Positive/negative gap:     {metrics.score_gap:+.4f}")
    print(f"\n  Avg ranking latency:       {np.mean(rank_latency_ms):.2f}ms")
    print(f"  P99 ranking latency:       {np.percentile(rank_latency_ms, 99):.2f}ms")
    print(f"  Encode time:               {encode_time:.1f}s")

    print(f"\n  -- Per-Domain Recall@5 --")
    print(f"  {'Domain':<12} {'Cross':>6} {'Within':>8}")
    print(f"  {'-' * 28}")
    for idx, domain in enumerate(DOMAINS):
        domain_cross = [q for q in cross if domain_for_expected(q.expected) == idx]
        domain_within = [q for q in within if domain_for_expected(q.expected) == idx]
        ch = hits(domain_cross) if domain_cross else 0
        wh = hits(domain_within) if domain_within else 0
        print(f"  {domain:<12} {ch}/{len(domain_cross) or 1:>3}  {wh}/{len(domain_within) or 1:>5}")

    misses = []
    for query in positives:
        ranked_ids = [r.cell_id for r in rankings[query.item_id]]
        if not set(ranked_ids[:5]).intersection(query.expected):
            misses.append((query, ranked_ids[0]))

    if misses:
        print(f"\n  Misses ({len(misses)}):")
        for query, top_id in misses[:20]:
            domain = DOMAINS[domain_for_expected(query.expected)]
            print(f"    X [{domain:>10}] \"{query.text[:48]}\" -> mem {top_id}")


def main() -> None:
    print("=" * 70)
    print("100-Memory Traditional RAG Baseline")
    print(f"Model: {MODEL_NAME} | Memories: {len(MEMORIES)} | Queries: {len(ALL_QUERIES)}")
    print("=" * 70)

    query_cases = load_query_cases()
    adapter = EmbeddingModelAdapter(MODEL_NAME)

    started = time.time()
    memory_texts = [f"passage: {memory}" for memory in MEMORIES]
    query_texts = [f"query: {query.text}" for query in query_cases]
    memory_embeddings = adapter.encode(memory_texts)
    query_embeddings = adapter.encode(query_texts)
    encode_time = time.time() - started

    ranker = CosineRankingStrategy(memory_embeddings)
    rankings = {}
    latencies = []
    for query, embedding in zip(query_cases, query_embeddings):
        start = time.perf_counter()
        rankings[query.item_id] = ranker.rank(embedding)
        latencies.append((time.perf_counter() - start) * 1000.0)

    metrics = compute_metrics(rankings, query_cases)
    print_report(rankings, query_cases, metrics, encode_time, latencies)


if __name__ == "__main__":
    main()
