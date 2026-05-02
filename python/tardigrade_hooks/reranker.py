"""Cross-encoder reranking — Stage-2 refinement when memo text exists.

The engine retrieves top-K candidates in pure latent space (per-token Top5Avg
+ optional ``RefinementMode::MeanCentered`` rescoring). When stored cells
have associated memo text, a cross-encoder model can re-score the top-K
using full query+document attention — strictly more expressive than the
bi-encoder dot product because it lets every query token attend to every
document token jointly.

This is the standard text-reranker pattern (Pinecone "Rerankers and
Two-Stage Retrieval", 2023) using small open-weight cross-encoders such
as ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (Wang et al., MiniLM,
arXiv:2002.10957) or BGE-Reranker-v2-m3 (BAAI, arXiv:2402.03216 — see
``docs/refs/external-references.md`` B1).

## Why Python-side, not Rust-side

The reranker is a transformer; the engine is a tensor store. Loading
PyTorch into Rust (via tch / candle) is plausible but adds a heavy
dependency for a refinement that's optional and text-dependent. The
existing engine `RefinementMode::{None, MeanCentered, LatentPrf}` covers
the no-text path. This Python module covers the text-available path.

## Usage

```python
from tardigrade_hooks.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker()

# `candidates` is whatever the engine returned. `get_text` maps each
# candidate to its memo text (or returns None if no text exists, in
# which case that candidate keeps its first-stage rank).
ordered = reranker.rerank(
    query_text="Where did Alice move?",
    candidates=engine.mem_read_tokens(query_tokens, k=10, owner=1),
    get_text=lambda c: lookup_text(c.cell_id),
)
```

Returns the candidates re-sorted by cross-encoder score, descending.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


class CrossEncoderReranker:
    """Late-interaction reranker for text-bearing first-stage candidates.

    Loads a small cross-encoder once (default ~22M params, runs on CPU
    or GPU). Operates as a Stage-2 pass over a list of candidates the
    engine returned. Candidates without text fall back to first-stage
    rank — they stay in place if everyone else gets reranked, or above
    everyone else if their original score was higher.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore

        self.model_name = model_name or self.DEFAULT_MODEL
        self._batch_size = batch_size
        self._model = CrossEncoder(self.model_name, device=device)

    def rerank(
        self,
        query_text: str,
        candidates: list[Any],
        get_text: Callable[[Any], str | None],
    ) -> list[Any]:
        """Re-sort ``candidates`` by cross-encoder relevance to ``query_text``.

        Args:
            query_text: Original query text (not the per-token tensor).
            candidates: First-stage results, in descending engine score.
            get_text: Callable mapping a candidate to its memo text, or
                ``None`` when no text is associated with that cell.

        Returns:
            Candidates re-sorted by cross-encoder score (descending). Any
            candidate whose ``get_text`` returns ``None`` keeps its
            first-stage relative position among other text-less items
            and lands after the text-scored items only if the
            cross-encoder ranks them all higher.
        """
        if not candidates:
            return []

        scored: list[tuple[float, int, Any]] = []
        unscored: list[tuple[int, Any]] = []
        pairs: list[tuple[str, str]] = []
        pair_idx: list[int] = []

        for original_rank, cand in enumerate(candidates):
            text = get_text(cand)
            if text is None or not text.strip():
                unscored.append((original_rank, cand))
                continue
            pair_idx.append(original_rank)
            pairs.append((query_text, text))

        if pairs:
            raw_scores = self._model.predict(pairs, batch_size=self._batch_size)
            # CrossEncoder returns numpy; cast to python floats for sort.
            for original_rank, score in zip(pair_idx, raw_scores):
                scored.append((float(score), original_rank, candidates[original_rank]))

        # Reranked items in descending score order.
        scored.sort(key=lambda t: (-t[0], t[1]))
        ordered: list[Any] = [c for _, _, c in scored]

        # Append text-less items in their original relative order so they
        # don't get silently dropped.
        unscored.sort(key=lambda t: t[0])
        ordered.extend(c for _, c in unscored)

        return ordered

    def metadata(self) -> dict[str, str]:
        return {"reranker": "cross_encoder", "model": self.model_name}


def rerank_pairs(
    query_text: str,
    pairs: Iterable[tuple[Any, str]],
    model_name: str | None = None,
    device: str | None = None,
) -> list[tuple[Any, float]]:
    """One-shot helper for ad-hoc reranking without instantiating a class.

    Returns ``[(item, score), ...]`` sorted by score descending. Use the
    class-based API for repeated calls so the model isn't reloaded.
    """
    items_and_texts = list(pairs)
    if not items_and_texts:
        return []
    reranker = CrossEncoderReranker(model_name=model_name, device=device)
    items = [item for item, _ in items_and_texts]
    text_lookup = {id(item): text for item, text in items_and_texts}
    ordered = reranker.rerank(
        query_text=query_text,
        candidates=items,
        get_text=lambda c: text_lookup[id(c)],
    )
    # Re-pair with the cross-encoder scores for return.
    score_map = {
        id(item): float(s)
        for item, s in zip(
            items,
            reranker._model.predict(  # noqa: SLF001 — internal helper, intentional
                [(query_text, text_lookup[id(it)]) for it in items],
                batch_size=32,
            ),
        )
    }
    return [(item, score_map[id(item)]) for item in ordered]
