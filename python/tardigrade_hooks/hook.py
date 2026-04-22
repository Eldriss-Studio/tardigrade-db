"""TardigradeHook — Abstract base class for LLM inference hooks.

Template Method pattern: defines the hook lifecycle that concrete
implementations (e.g., HuggingFaceHook) must fill in.

Strategy pattern: WriteDecision decouples write policy from the Engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class WriteDecision:
    """Decision returned by on_generate: should this KV pair be persisted?

    Strategy pattern: the hook controls *what* and *when* to write,
    without subclassing or modifying the Engine.
    """

    should_write: bool
    salience: float = 50.0
    parent_cell_id: Optional[int] = None
    key: Optional[np.ndarray] = field(default=None, repr=False)
    value: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class MemoryCellHandle:
    """A lightweight handle to a retrieved memory cell.

    Provides access to cell metadata and key/value vectors
    without exposing the full Rust MemoryCell internals.
    """

    cell_id: int
    owner: int
    layer: int
    score: float
    key: np.ndarray
    value: np.ndarray


class TardigradeHook(ABC):
    """Abstract base class for LLM inference hooks.

    Template Method pattern: subclasses implement model-specific logic
    for deciding what to write (on_generate) and what to inject (on_prefill).

    Usage::

        class MyHook(TardigradeHook):
            def on_generate(self, layer, hidden_states):
                # Decide whether to persist this layer's KV.
                ...
                return WriteDecision(should_write=True, salience=80.0)

            def on_prefill(self, layer, query_states):
                # Retrieve relevant KV from memory.
                ...
                return [MemoryCellHandle(...)]
    """

    @abstractmethod
    def on_generate(
        self, layer: int, hidden_states: np.ndarray
    ) -> WriteDecision:
        """Called during forward pass to decide whether to persist KV.

        Args:
            layer: Transformer layer index.
            hidden_states: Hidden states tensor (shape varies by model).

        Returns:
            WriteDecision indicating whether to write, with salience and optional parent.
        """

    @abstractmethod
    def on_prefill(
        self, layer: int, query_states: np.ndarray
    ) -> list[MemoryCellHandle]:
        """Called during prefill to inject retrieved KV into the attention cache.

        Args:
            layer: Transformer layer index.
            query_states: Query projection tensor (shape varies by model).

        Returns:
            List of MemoryCellHandle objects to inject as additional KV cache entries.
        """
