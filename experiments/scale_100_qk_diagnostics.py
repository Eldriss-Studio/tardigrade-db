#!/usr/bin/env python3
"""100-memory Q*K diagnostic scorer lab.

This script is intentionally outside normal CI. It loads Qwen, computes the
query/key tensors once, and compares retrieval scorers without changing the
production engine.
"""

from __future__ import annotations

import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(".").resolve() / "experiments"))

from corpus_100 import ALL_QUERIES, MEMORIES

MODEL_NAME = "Qwen/Qwen3-0.6B"
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
GRAVITY_WATCHLIST = [87, 6, 62, 93]
TOKEN_RE = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "a",
    "about",
    "and",
    "at",
    "for",
    "from",
    "i",
    "in",
    "it",
    "me",
    "my",
    "of",
    "on",
    "the",
    "to",
    "was",
    "with",
}


@dataclass(frozen=True)
class TokenTrace:
    query_token: str
    memory_token: str
    score: float
    query_index: int
    memory_index: int
    head_index: int | None = None


@dataclass(frozen=True)
class RankingEntry:
    cell_id: int
    score: float
    trace: TokenTrace | None = None


@dataclass(frozen=True)
class TensorItem:
    item_id: int
    text: str
    tokens: list[str]
    vectors: np.ndarray
    expected: tuple[int, ...] = ()
    qtype: str = "memory"


@dataclass(frozen=True)
class Metrics:
    recall_at: dict[int, float]
    mrr: float
    unique_top1: int
    worst_top1_cell: int
    worst_top1_count: int
    negative_false_positive_rate: float
    score_gap: float
    shuffled_recall_at_5: float


def normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def pairwise_dot(query: np.ndarray, memory: np.ndarray, scale: bool = True) -> np.ndarray:
    scores = query @ memory.T
    if scale and query.shape[1] > 0:
        scores = scores / math.sqrt(query.shape[1])
    return scores


def best_pair_trace(
    scores: np.ndarray,
    query_tokens: list[str],
    memory_tokens: list[str],
    head_index: int | None = None,
) -> TokenTrace:
    qi, mi = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return TokenTrace(
        query_token=query_tokens[qi] if qi < len(query_tokens) else "?",
        memory_token=memory_tokens[mi] if mi < len(memory_tokens) else "?",
        score=float(scores[qi, mi]),
        query_index=int(qi),
        memory_index=int(mi),
        head_index=head_index,
    )


def max_sim_score(query: TensorItem, memory: TensorItem) -> RankingEntry:
    scores = pairwise_dot(query.vectors, memory.vectors)
    trace = best_pair_trace(scores, query.tokens, memory.tokens)
    return RankingEntry(memory.item_id, float(np.max(scores)), trace)


def colbert_sum_score(query: TensorItem, memory: TensorItem) -> RankingEntry:
    scores = pairwise_dot(query.vectors, memory.vectors)
    trace = best_pair_trace(scores, query.tokens, memory.tokens)
    return RankingEntry(memory.item_id, float(np.max(scores, axis=1).sum()), trace)


def topn_avg_score(query: TensorItem, memory: TensorItem, n: int = 5) -> RankingEntry:
    scores = pairwise_dot(query.vectors, memory.vectors)
    flat = np.sort(scores.ravel())
    take = min(n, len(flat))
    trace = best_pair_trace(scores, query.tokens, memory.tokens)
    return RankingEntry(memory.item_id, float(flat[-take:].mean()), trace)


def cosine_max_score(query: TensorItem, memory: TensorItem) -> RankingEntry:
    q = normalize_rows(query.vectors)
    k = normalize_rows(memory.vectors)
    scores = pairwise_dot(q, k, scale=False)
    trace = best_pair_trace(scores, query.tokens, memory.tokens)
    return RankingEntry(memory.item_id, float(np.max(scores)), trace)


def centered_max_score(query: TensorItem, memory: TensorItem, center: np.ndarray) -> RankingEntry:
    q = query.vectors - center
    k = memory.vectors - center
    scores = pairwise_dot(q, k)
    trace = best_pair_trace(scores, query.tokens, memory.tokens)
    return RankingEntry(memory.item_id, float(np.max(scores)), trace)


def per_head_max_score(
    query: TensorItem,
    memory: TensorItem,
    num_heads: int,
    head_dim: int,
) -> RankingEntry:
    q = query.vectors.reshape(query.vectors.shape[0], num_heads, head_dim)
    k = memory.vectors.reshape(memory.vectors.shape[0], num_heads, head_dim)

    head_scores = []
    best_trace = None
    best_score = -np.inf
    for head in range(num_heads):
        scores = (q[:, head, :] @ k[:, head, :].T) / math.sqrt(head_dim)
        score = float(np.max(scores))
        head_scores.append(score)
        if score > best_score:
            best_score = score
            best_trace = best_pair_trace(scores, query.tokens, memory.tokens, head)

    return RankingEntry(memory.item_id, float(np.mean(head_scores)), best_trace)


def lexical_tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def bm25_rank(query_text: str, memory_texts: list[str], k1: float = 1.5, b: float = 0.75) -> list[RankingEntry]:
    docs = [lexical_tokens(text) for text in memory_texts]
    query_terms = lexical_tokens(query_text)
    avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
    df = Counter(term for doc in docs for term in set(doc))
    doc_tfs = [Counter(doc) for doc in docs]
    n_docs = len(docs)

    rankings = []
    for doc_id, (doc, tf) in enumerate(zip(docs, doc_tfs)):
        score = 0.0
        best_term = ""
        best_term_score = -np.inf
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1.0 - b + b * len(doc) / max(avgdl, 1e-8))
            term_score = idf * (tf[term] * (k1 + 1.0)) / denom
            score += term_score
            if term_score > best_term_score:
                best_term = term
                best_term_score = term_score
        trace = TokenTrace(best_term, best_term, float(best_term_score), -1, -1) if best_term else None
        rankings.append(RankingEntry(doc_id, float(score), trace))

    return sorted(rankings, key=lambda r: r.score, reverse=True)


def rank_with_scorer(
    query: TensorItem,
    memories: list[TensorItem],
    scorer: Callable[[TensorItem, TensorItem], RankingEntry],
) -> list[RankingEntry]:
    return sorted((scorer(query, memory) for memory in memories), key=lambda r: r.score, reverse=True)


def reciprocal_rank(ranking: list[int], expected: Iterable[int]) -> float:
    expected_set = set(expected)
    for idx, cell_id in enumerate(ranking, start=1):
        if cell_id in expected_set:
            return 1.0 / idx
    return 0.0


def random_recall_at_k(num_memories: int, expected_count: int, k: int) -> float:
    if expected_count <= 0:
        return 0.0
    if k >= num_memories:
        return 1.0
    misses = math.comb(num_memories - expected_count, k) / math.comb(num_memories, k)
    return 1.0 - misses


def gravity_stats(top1_ids: list[int]) -> tuple[int, int, int]:
    if not top1_ids:
        return 0, -1, 0
    counts = Counter(top1_ids)
    worst_cell, worst_count = counts.most_common(1)[0]
    return len(counts), worst_cell, worst_count


def domain_confusion(rows: list[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((len(DOMAINS), len(DOMAINS)), dtype=np.int32)
    for expected_domain, predicted_domain in rows:
        matrix[expected_domain, predicted_domain] += 1
    return matrix


def compute_metrics(
    rankings_by_query: dict[int, list[RankingEntry]],
    queries: list[TensorItem],
    num_memories: int,
    topks: tuple[int, ...] = (1, 3, 5, 10),
    seed: int = 13,
) -> Metrics:
    positives = [q for q in queries if q.qtype != "negative"]
    negatives = [q for q in queries if q.qtype == "negative"]

    recall_at = {}
    for k in topks:
        hits = 0
        for query in positives:
            ranking = [r.cell_id for r in rankings_by_query[query.item_id]]
            if set(ranking[:k]).intersection(query.expected):
                hits += 1
        recall_at[k] = hits / max(len(positives), 1)

    mrr = float(
        np.mean([
            reciprocal_rank([r.cell_id for r in rankings_by_query[q.item_id]], q.expected)
            for q in positives
        ])
    )

    top1_ids = [rankings_by_query[q.item_id][0].cell_id for q in positives if rankings_by_query[q.item_id]]
    unique_top1, worst_cell, worst_count = gravity_stats(top1_ids)

    expected_scores = []
    for query in positives:
        scores = {r.cell_id: r.score for r in rankings_by_query[query.item_id]}
        expected_scores.append(max(scores[e] for e in query.expected if e in scores))

    negative_scores = [
        rankings_by_query[q.item_id][0].score
        for q in negatives
        if rankings_by_query[q.item_id]
    ]
    threshold = float(np.percentile(expected_scores, 10)) if expected_scores else float("inf")
    neg_fp = sum(score >= threshold for score in negative_scores) / max(len(negative_scores), 1)
    score_gap = float(np.mean(expected_scores) - np.mean(negative_scores)) if negative_scores else 0.0

    shuffled_expected = [q.expected for q in positives]
    rng = random.Random(seed)
    rng.shuffle(shuffled_expected)
    shuffled_hits = 0
    for query, expected in zip(positives, shuffled_expected):
        ranking = [r.cell_id for r in rankings_by_query[query.item_id]]
        if set(ranking[:5]).intersection(expected):
            shuffled_hits += 1

    return Metrics(
        recall_at=recall_at,
        mrr=mrr,
        unique_top1=unique_top1,
        worst_top1_cell=worst_cell,
        worst_top1_count=worst_count,
        negative_false_positive_rate=neg_fp,
        score_gap=score_gap,
        shuffled_recall_at_5=shuffled_hits / max(len(positives), 1),
    )


def print_metrics_table(results: dict[str, Metrics]) -> None:
    print("\n  -- Scorer Comparison --")
    print(
        f"  {'Scorer':<22} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} "
        f"{'MRR':>6} {'uniq':>6} {'worst':>8} {'negFP':>7} {'gap':>9} {'shuf@5':>8}"
    )
    print(f"  {'-' * 104}")
    for name, metrics in sorted(results.items(), key=lambda item: item[1].recall_at[5], reverse=True):
        print(
            f"  {name:<22} "
            f"{metrics.recall_at[1] * 100:5.1f}% "
            f"{metrics.recall_at[3] * 100:5.1f}% "
            f"{metrics.recall_at[5] * 100:5.1f}% "
            f"{metrics.recall_at[10] * 100:5.1f}% "
            f"{metrics.mrr:6.3f} "
            f"{metrics.unique_top1:6d} "
            f"{metrics.worst_top1_cell}:{metrics.worst_top1_count:<4d} "
            f"{metrics.negative_false_positive_rate * 100:6.1f}% "
            f"{metrics.score_gap:9.2f} "
            f"{metrics.shuffled_recall_at_5 * 100:7.1f}%"
        )


def print_domain_confusion(rankings: dict[int, list[RankingEntry]], queries: list[TensorItem]) -> None:
    rows = []
    for query in queries:
        if query.qtype == "negative" or not query.expected:
            continue
        expected_domain = query.expected[0] // 10
        predicted_domain = rankings[query.item_id][0].cell_id // 10
        rows.append((expected_domain, predicted_domain))

    matrix = domain_confusion(rows)
    print("\n  -- Domain Confusion (best scorer top-1) --")
    print("  rows=expected, cols=predicted")
    print(f"  {'':<11} " + " ".join(f"{d[:3]:>3}" for d in DOMAINS))
    for idx, row in enumerate(matrix):
        print(f"  {DOMAINS[idx]:<11} " + " ".join(f"{v:>3}" for v in row))


def print_gravity_token_pairs(
    scorer_name: str,
    rankings: dict[int, list[RankingEntry]],
    queries: list[TensorItem],
) -> None:
    print(f"\n  -- Gravity Token Pairs ({scorer_name}) --")
    query_by_id = {q.item_id: q for q in queries}
    for memory_id in GRAVITY_WATCHLIST:
        pairs = Counter()
        for query_id, ranking in rankings.items():
            query = query_by_id[query_id]
            if query.qtype == "negative" or not ranking or ranking[0].cell_id != memory_id:
                continue
            trace = ranking[0].trace
            if trace is not None:
                pairs[(trace.query_token, trace.memory_token)] += 1
        if pairs:
            rendered = ", ".join(f"{q}->{m} ({n})" for (q, m), n in pairs.most_common(5))
        else:
            rendered = "not top-1"
        print(f"  mem {memory_id:>2}: {rendered}")


def get_attn_layer(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx].self_attn
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx].attn
    raise AttributeError("Unsupported HuggingFace model layout")


def expand_k_for_gqa(k_tokens: np.ndarray, num_kv_heads: int, gqa_ratio: int, head_dim: int) -> np.ndarray:
    if gqa_ratio <= 1:
        return k_tokens
    reshaped = k_tokens.reshape(-1, num_kv_heads, head_dim)
    expanded = np.repeat(reshaped, gqa_ratio, axis=1)
    return expanded.reshape(-1, num_kv_heads * gqa_ratio * head_dim).astype(np.float32)


def project_item(
    item_id: int,
    text: str,
    model,
    tokenizer,
    layer_idx: int,
    projection: str,
    expected: tuple[int, ...] = (),
    qtype: str = "memory",
) -> TensorItem:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_hidden_states=True)

    hidden = out.hidden_states[layer_idx][0]
    attn = get_attn_layer(model, layer_idx)
    config = model.config
    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_q_heads)
    gqa_ratio = num_q_heads // num_kv_heads

    with torch.no_grad():
        if projection == "q":
            projected = attn.q_proj(hidden)
            if hasattr(attn, "q_norm"):
                projected = projected.view(-1, num_q_heads, head_dim)
                projected = attn.q_norm(projected)
                projected = projected.view(-1, num_q_heads * head_dim)
            vectors = projected[1:].detach().cpu().numpy().astype(np.float32)
        elif projection == "k":
            projected = attn.k_proj(hidden)
            if hasattr(attn, "k_norm"):
                projected = projected.view(-1, num_kv_heads, head_dim)
                projected = attn.k_norm(projected)
                projected = projected.view(-1, num_kv_heads * head_dim)
            vectors = projected[1:].detach().cpu().numpy().astype(np.float32)
            vectors = expand_k_for_gqa(vectors, num_kv_heads, gqa_ratio, head_dim)
        else:
            raise ValueError(f"unknown projection: {projection}")

    token_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)[1:]
    return TensorItem(item_id, text, tokens, vectors, expected=expected, qtype=qtype)


def build_rankings(
    memories: list[TensorItem],
    queries: list[TensorItem],
    num_heads: int,
    head_dim: int,
) -> dict[str, dict[int, list[RankingEntry]]]:
    center = np.concatenate([m.vectors for m in memories], axis=0).mean(axis=0, keepdims=True)
    memory_texts = [m.text for m in memories]
    scorers: dict[str, Callable[[TensorItem, TensorItem], RankingEntry]] = {
        "max_sim": max_sim_score,
        "colbert_sum": colbert_sum_score,
        "top5_pair_avg": lambda q, m: topn_avg_score(q, m, 5),
        "cosine_max": cosine_max_score,
        "mean_centered": lambda q, m: centered_max_score(q, m, center),
        "per_head_max": lambda q, m: per_head_max_score(q, m, num_heads, head_dim),
    }

    rankings = {
        name: {query.item_id: rank_with_scorer(query, memories, scorer) for query in queries}
        for name, scorer in scorers.items()
    }
    rankings["bm25"] = {
        query.item_id: bm25_rank(query.text, memory_texts)
        for query in queries
    }
    return rankings


def print_random_baseline(queries: list[TensorItem], num_memories: int) -> None:
    positives = [q for q in queries if q.qtype != "negative"]
    for k in (1, 3, 5, 10):
        expected = np.mean([random_recall_at_k(num_memories, len(q.expected), k) for q in positives])
        print(f"  Random expected recall@{k}: {expected * 100:.1f}%")


def main() -> None:
    print("=" * 78)
    print("100-Memory Q*K Diagnostic Scorer Lab")
    print(f"Model: {MODEL_NAME} | Memories: {len(MEMORIES)} | Queries: {len(ALL_QUERIES)}")
    print("=" * 78)

    print(f"\n  Loading {MODEL_NAME}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    layer_idx = int(model.config.num_hidden_layers * 0.67)
    num_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // num_heads)
    print(f"OK ({model.config.num_hidden_layers}L, layer={layer_idx}, heads={num_heads}, head_dim={head_dim})")

    started = time.time()
    print("\n--- PROJECTING MEMORIES ---")
    memories = []
    for idx, text in enumerate(MEMORIES):
        memories.append(project_item(idx, text, model, tokenizer, layer_idx, "k"))
        if (idx + 1) % 25 == 0:
            print(f"  {idx + 1}/100 projected")

    print("\n--- PROJECTING QUERIES ---")
    queries = [
        project_item(idx, text, model, tokenizer, layer_idx, "q", tuple(expected), qtype)
        for idx, (text, expected, qtype) in enumerate(ALL_QUERIES)
    ]
    print(f"  Projection time: {time.time() - started:.1f}s")

    print("\n--- SCORING ---")
    score_started = time.time()
    rankings = build_rankings(memories, queries, num_heads, head_dim)
    metrics = {name: compute_metrics(ranking, queries, len(memories)) for name, ranking in rankings.items()}
    print(f"  Scoring time: {time.time() - score_started:.1f}s")

    print("\n  -- Baselines --")
    print_random_baseline(queries, len(memories))
    print_metrics_table(metrics)

    best_name, best_metrics = max(metrics.items(), key=lambda item: item[1].recall_at[5])
    print_domain_confusion(rankings[best_name], queries)
    print_gravity_token_pairs(best_name, rankings[best_name], queries)
    if best_name != "max_sim":
        print_gravity_token_pairs("max_sim", rankings["max_sim"], queries)

    latent_names = [n for n in metrics if n != "bm25"]
    best_latent = max(latent_names, key=lambda name: metrics[name].recall_at[5])
    best_latent_metrics = metrics[best_latent]
    bm25_metrics = metrics["bm25"]

    print("\n--- DECISION GATE ---")
    print(f"  Best latent scorer: {best_latent} ({best_latent_metrics.recall_at[5] * 100:.1f}% R@5)")
    print(f"  BM25 scorer:        {bm25_metrics.recall_at[5] * 100:.1f}% R@5")

    clears_gate = (
        best_latent_metrics.recall_at[5] >= 0.70
        and best_latent_metrics.worst_top1_count <= 3
        and best_latent_metrics.negative_false_positive_rate <= 0.10
    )
    if clears_gate:
        verdict = "IMPLEMENT_LATENT_SCORER_IN_RUST"
    elif bm25_metrics.recall_at[5] > best_latent_metrics.recall_at[5]:
        verdict = "PLAN_HYBRID_CANDIDATE_RERANK"
    else:
        verdict = "REVISE_CORPUS_OR_REPRESENTATION"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
