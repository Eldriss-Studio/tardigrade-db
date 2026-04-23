"""HuggingFaceHook — Reference implementation of TardigradeHook for HF transformers.

Adapter pattern: translates HuggingFace hidden_states tensor format
to TardigradeDB's flat f32 key/value API.
"""

import numpy as np

from .hook import MemoryCellHandle, TardigradeHook, WriteDecision


class HuggingFaceHook(TardigradeHook):
    """Hook for HuggingFace transformers models.

    Salience heuristic: L2 norm of hidden states — high norm indicates
    the layer is producing distinctive representations worth remembering.

    Prefill: queries the engine with the mean of query states across
    the sequence dimension, returning top-k memory cells.
    """

    def __init__(self, engine, owner: int = 1, k: int = 5, norm_threshold: float = 1.0):
        """Initialize the hook.

        Args:
            engine: A tardigrade_db.Engine instance.
            owner: Owner/agent ID for all writes.
            k: Number of cells to retrieve on prefill.
            norm_threshold: Minimum L2 norm of hidden states to trigger a write.
        """
        self.engine = engine
        self.owner = owner
        self.k = k
        self.norm_threshold = norm_threshold

    def on_generate(self, layer: int, **kwargs) -> WriteDecision:
        """Decide whether to persist based on hidden state norm.

        Higher norm means more salient, higher importance score.
        The key is the mean hidden state across the sequence dimension.
        """
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            return WriteDecision(should_write=False)

        # Handle both 2D (seq, hidden) and 3D (batch, seq, hidden) inputs.
        if hidden_states.ndim == 3:
            # Take first batch element.
            hidden_states = hidden_states[0]

        # Mean across sequence dimension → (hidden_dim,).
        mean_hidden = hidden_states.mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean_hidden))

        if norm < self.norm_threshold:
            return WriteDecision(should_write=False)

        # Map norm to salience in [0, 100]. Clamp at 100.
        salience = min(norm * 50.0, 100.0)

        return WriteDecision(
            should_write=True,
            salience=salience,
            key=mean_hidden,
            value=mean_hidden,  # In a real implementation, value would be the V projection.
        )

    def on_prefill(self, layer: int, **kwargs) -> list[MemoryCellHandle]:
        """Retrieve relevant KV from engine using mean query state."""
        query_states = kwargs.get("query_states")
        if query_states is None:
            return []

        if query_states.ndim == 3:
            query_states = query_states[0]

        mean_query = query_states.mean(axis=0).astype(np.float32)

        results = self.engine.mem_read(mean_query, self.k, self.owner)

        return [
            MemoryCellHandle(
                cell_id=r.cell_id,
                owner=r.owner,
                layer=r.layer,
                score=r.score,
                key=np.array(r.key(), dtype=np.float32),
                value=np.array(r.value(), dtype=np.float32),
            )
            for r in results
        ]
