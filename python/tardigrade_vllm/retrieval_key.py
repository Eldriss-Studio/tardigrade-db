"""Retrieval key strategies for vLLM KV connector.

Strategy pattern: pluggable retrieval key computation.
The retrieval key determines which stored KV pack matches a new request.

Architecture constraint: the save side stores last-token K from the last
transformer layer. The load side uses the embedding table (no GPU forward
pass). These produce vectors in different spaces unless hidden_size == kv_dim.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger("tardigrade_vllm")


LAST_TOKEN_EMBEDDING = "last_token_embedding"
MEAN_POOL_EMBEDDING = "mean_pool_embedding"
PROJECTED_EMBEDDING = "projected_embedding"


class RetrievalKeyStrategy(ABC):
    """Strategy interface: compute a retrieval key from token IDs + embedding table."""

    @abstractmethod
    def compute(self, token_ids: list[int], embed_weights: np.ndarray) -> np.ndarray | None:
        """Return a retrieval key vector, or None if computation is not possible."""


class LastTokenEmbeddingStrategy(RetrievalKeyStrategy):
    """Last token's embedding vector as retrieval key.

    Matches save-side key (last-token K, last layer) ONLY when
    hidden_size == kv_dim. For models where these differ, retrieval
    recall degrades because load and save keys live in different
    vector spaces.

    Validated on: Qwen3-0.6B (hidden_size=1024, kv_dim=8*128=1024).
    """

    def compute(self, token_ids, embed_weights):
        if embed_weights.size == 0 or len(token_ids) == 0:
            return None
        valid_ids = [tid for tid in token_ids if 0 <= tid < embed_weights.shape[0]]
        if not valid_ids:
            return None
        return embed_weights[valid_ids[-1]].astype(np.float32)


class MeanPoolEmbeddingStrategy(RetrievalKeyStrategy):
    """Mean-pool all token embeddings as retrieval key.

    More robust than last-token for variable-length prompts: captures
    the overall semantic direction rather than depending on a single
    token. Still requires hidden_size == kv_dim for space alignment.
    """

    def compute(self, token_ids, embed_weights):
        if embed_weights.size == 0 or len(token_ids) == 0:
            return None
        valid_ids = [tid for tid in token_ids if 0 <= tid < embed_weights.shape[0]]
        if not valid_ids:
            return None
        embeddings = embed_weights[valid_ids]
        return embeddings.mean(axis=0).astype(np.float32)


class ProjectedEmbeddingStrategy(RetrievalKeyStrategy):
    """Project last-token embedding from hidden_size to kv_dim.

    Handles models where hidden_size != kv_dim (e.g., GQA models with
    fewer KV heads than query heads). Uses a learned or random projection
    matrix W in R^(kv_dim x hidden_size) to bridge the vector spaces.

    If no projection matrix is provided, initializes a random orthogonal
    projection (preserves distances approximately).
    """

    def __init__(self, kv_dim: int, hidden_size: int, projection: np.ndarray | None = None):
        self._kv_dim = kv_dim
        self._hidden_size = hidden_size
        if projection is not None:
            self._projection = projection.astype(np.float32)
        else:
            rng = np.random.default_rng(42)
            raw = rng.standard_normal((kv_dim, hidden_size)).astype(np.float32)
            u, _, vt = np.linalg.svd(raw, full_matrices=False)
            self._projection = u @ vt

    def compute(self, token_ids, embed_weights):
        if embed_weights.size == 0 or len(token_ids) == 0:
            return None
        valid_ids = [tid for tid in token_ids if 0 <= tid < embed_weights.shape[0]]
        if not valid_ids:
            return None
        embedding = embed_weights[valid_ids[-1]].astype(np.float32)
        return (self._projection @ embedding).astype(np.float32)


_STRATEGIES: dict[str, type[RetrievalKeyStrategy]] = {
    LAST_TOKEN_EMBEDDING: LastTokenEmbeddingStrategy,
    MEAN_POOL_EMBEDDING: MeanPoolEmbeddingStrategy,
}


def get_strategy(name: str) -> RetrievalKeyStrategy:
    """Factory Method: look up strategy by config name."""
    cls = _STRATEGIES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown retrieval key strategy: {name!r}. "
            f"Available: {list(_STRATEGIES)}"
        )
    return cls()


def check_key_alignment(hidden_size: int, kv_dim: int) -> bool:
    """Diagnostic: check if load-side and save-side key spaces align.

    Returns True if hidden_size == kv_dim (embedding lookup produces
    vectors in the same space as K projections). Logs a warning on
    mismatch.
    """
    aligned = hidden_size == kv_dim
    if not aligned:
        logger.warning(
            "Key space mismatch: hidden_size=%d != kv_dim=%d. "
            "Load-side retrieval keys (embedding table) live in a different "
            "vector space than save-side keys (K projections). Retrieval recall "
            "may degrade. Consider a projected embedding strategy.",
            hidden_size,
            kv_dim,
        )
    return aligned
