"""Multi-layer query fusion — Composite pattern.

Runs retrieval at multiple transformer layers, fuses rankings via
Reciprocal Rank Fusion (RRF). Pure latent-space: no text, no external
model. Only the query is multi-layer; stored memories use single-layer keys.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .constants import DEFAULT_CAPTURE_LAYER_RATIO
from .encoding import encode_per_token

DEFAULT_LAYER_RATIOS = (0.50, DEFAULT_CAPTURE_LAYER_RATIO, 0.83)
DEFAULT_RRF_K = 60


def rrf_fuse(
    ranked_lists: list[list[dict]],
    k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    Each item must have a ``pack_id`` key. Returns a single list
    sorted by fused RRF score (descending).
    """
    if not ranked_lists:
        return []

    scores: dict[int, float] = defaultdict(float)
    pack_data: dict[int, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            pid = item["pack_id"]
            scores[pid] += 1.0 / (k + rank + 1)
            if pid not in pack_data:
                pack_data[pid] = item

    fused = []
    for pid, score in sorted(scores.items(), key=lambda x: -x[1]):
        entry = dict(pack_data[pid])
        entry["rrf_score"] = score
        fused.append(entry)

    return fused


class MultiLayerQuery:
    """Composite: queries the engine at multiple layers, fuses via RRF."""

    def __init__(
        self,
        engine,
        *,
        layer_ratios: tuple[float, ...] = DEFAULT_LAYER_RATIOS,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self._engine = engine
        self._layer_ratios = layer_ratios
        self._rrf_k = rrf_k

    def query(
        self,
        model,
        tokenizer,
        query_text: str,
        k: int = 5,
        owner: int | None = None,
    ) -> list[dict]:
        """Run retrieval at each layer, fuse via RRF."""
        import torch

        inputs = tokenizer(query_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size

        ranked_lists = []
        for ratio in self._layer_ratios:
            layer_idx = int(n_layers * ratio)
            hidden = out.hidden_states[layer_idx][0][1:]
            h_np = hidden.cpu().numpy().astype(np.float32)
            query_key = encode_per_token(h_np, hidden_size)
            results = self._engine.mem_read_pack(query_key, k * 2, owner)
            ranked_lists.append(results)

        fused = rrf_fuse(ranked_lists, k=self._rrf_k)
        return fused[:k]
