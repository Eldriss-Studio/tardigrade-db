"""Multi-view memory consolidation engine.

Command pattern: each ``consolidate(pack_id)`` call encapsulates one
consolidation operation.  Policy pattern: ``ConsolidationPolicy``
controls which packs are eligible and how many views each gets.

Views are stored as additional retrieval cells on the canonical pack via
``engine.add_view_keys()`` — the parent-document pattern: one canonical
pack, multiple retrieval surfaces, no separate view packs required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .constants import (
    CONSOLIDATION_MIN_TIER,
    DEFAULT_VIEW_FRAMINGS,
)

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class ConsolidationPolicy(ABC):
    """Controls which packs get consolidated and how many views each gets."""

    @abstractmethod
    def should_consolidate(self, *, tier: int, importance: float) -> bool: ...

    @abstractmethod
    def view_count(self, *, tier: int) -> int: ...


class DefaultConsolidationPolicy(ConsolidationPolicy):
    """Draft: skip.  Validated/Core: full default framing set."""

    def should_consolidate(self, *, tier: int, importance: float) -> bool:
        return tier >= CONSOLIDATION_MIN_TIER

    def view_count(self, *, tier: int) -> int:
        if tier < CONSOLIDATION_MIN_TIER:
            return 0
        return len(DEFAULT_VIEW_FRAMINGS)


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------


class MemoryConsolidator:
    """Produces multi-view retrieval surfaces for existing canonical memories.

    For each eligible pack, the consolidator:
    1. Reads the canonical text via ``engine.pack_text()``.
    2. Generates reframed views via ``ViewGenerator``.
    3. Creates a random retrieval key per view (v0 stub — real KV capture
       requires a model, deferred to v1).
    4. Attaches all view keys to the canonical pack via
       ``engine.add_view_keys()``.

    Views are **not** stored as separate packs; they are additional
    retrieval cells on the canonical pack (parent-document pattern).
    """

    def __init__(
        self,
        engine,
        *,
        owner: int = 1,
        view_generator=None,
        policy: ConsolidationPolicy | None = None,
        seed: int = 0,
    ):
        from .view_generator import ViewGenerator  # local import avoids circular

        self._engine = engine
        self._owner = owner
        self._view_gen = view_generator or ViewGenerator()
        self._policy = policy or DefaultConsolidationPolicy()
        self._rng = np.random.default_rng(seed)

    # -- Public API ----------------------------------------------------------

    def consolidate(self, pack_id: int) -> int:
        """Consolidate a single pack. Returns count of views attached.

        Returns 0 if the pack is ineligible (wrong tier) or already
        consolidated (idempotent).
        """
        pack_info = self._pack_info(pack_id)
        if pack_info is None:
            return 0

        tier = pack_info["tier"]
        importance = pack_info["importance"]

        if not self._policy.should_consolidate(tier=tier, importance=importance):
            return 0

        if self._already_consolidated(pack_id):
            return 0

        text = self._engine.pack_text(pack_id)
        if not text or not text.strip():
            return 0

        views = self._view_gen.generate(text)
        if not views:
            return 0

        view_keys = []
        for view_text in views:
            key = self._make_view_key(view_text)
            view_keys.append(key)

        return self._engine.add_view_keys(pack_id, view_keys)

    def consolidate_all(self, owner: int | None = None) -> dict[int, int]:
        """Consolidate all eligible packs. Returns {pack_id: views_attached}."""
        target_owner = owner if owner is not None else self._owner
        packs = self._engine.list_packs(target_owner)
        result: dict[int, int] = {}
        for p in packs:
            pid = p["pack_id"]
            count = self.consolidate(pid)
            if count > 0:
                result[pid] = count
        return result

    # -- Internals -----------------------------------------------------------

    def _pack_info(self, pack_id: int) -> dict | None:
        """Look up a pack's metadata from list_packs."""
        packs = self._engine.list_packs(self._owner)
        for p in packs:
            if p["pack_id"] == pack_id:
                return p
        return None

    def _already_consolidated(self, pack_id: int) -> bool:
        """Check if this pack already has view keys attached."""
        return self._engine.view_count(pack_id) > 0

    def _make_view_key(self, view_text: str) -> np.ndarray:
        """Generate a random retrieval key for a view (v0 stub)."""
        _ = view_text  # reserved for real model encoding in v1
        return self._rng.standard_normal(8).astype(np.float32)
