"""Multi-view memory consolidation engine.

Command pattern: each ``consolidate(pack_id)`` call encapsulates one
consolidation operation.  Policy pattern: ``ConsolidationPolicy``
controls which packs are eligible and how many views each gets.

Views are stored as separate packs linked to the canonical memory via
Supports edges — the engine's existing topology primitives, no new
Rust changes required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .constants import (
    CONSOLIDATION_MIN_TIER,
    DEFAULT_VIEW_FRAMINGS,
    EDGE_SUPPORTS,
    MAX_VIEWS_PER_MEMORY,
)
from .view_generator import ViewGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    """Produces multi-view packs for existing canonical memories.

    For each eligible pack, the consolidator:
    1. Reads the canonical text via ``engine.pack_text()``.
    2. Generates reframed views via ``ViewGenerator``.
    3. Stores each view as a new pack with a random retrieval key
       (v0 — real KV capture requires a model, deferred to v1).
    4. Links each view to the canonical pack via a Supports edge.
    """

    def __init__(
        self,
        engine,
        *,
        owner: int = 1,
        view_generator: ViewGenerator | None = None,
        policy: ConsolidationPolicy | None = None,
        seed: int = 0,
    ):
        self._engine = engine
        self._owner = owner
        self._view_gen = view_generator or ViewGenerator()
        self._policy = policy or DefaultConsolidationPolicy()
        self._rng = np.random.default_rng(seed)

    # -- Public API ----------------------------------------------------------

    def consolidate(self, pack_id: int) -> list[int]:
        """Consolidate a single pack.  Returns list of new view pack_ids.

        Returns empty list if the pack is ineligible (wrong tier) or
        already consolidated (idempotent).
        """
        pack_info = self._pack_info(pack_id)
        if pack_info is None:
            return []

        tier = pack_info["tier"]
        importance = pack_info["importance"]

        if not self._policy.should_consolidate(tier=tier, importance=importance):
            return []

        if self._already_consolidated(pack_id):
            return []

        text = self._engine.pack_text(pack_id)
        if not text or not text.strip():
            return []

        views = self._view_gen.generate(text)
        max_views = self._policy.view_count(tier=tier)
        views = views[:max_views]

        view_pack_ids: list[int] = []
        for view_text in views:
            vid = self._store_view_pack(view_text, pack_id)
            view_pack_ids.append(vid)

        return view_pack_ids

    def consolidate_all(self, owner: int | None = None) -> dict[int, list[int]]:
        """Consolidate all eligible packs for the given owner.

        Returns ``{canonical_pack_id: [view_pack_ids]}``.
        """
        target_owner = owner if owner is not None else self._owner
        packs = self._engine.list_packs(target_owner)
        result: dict[int, list[int]] = {}
        for p in packs:
            pid = p["pack_id"]
            views = self.consolidate(pid)
            if views:
                result[pid] = views
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
        """Check if any existing packs point to this one via Supports."""
        supporters = self._engine.pack_supports(pack_id)
        return len(supporters) > 0

    def _store_view_pack(self, view_text: str, canonical_pack_id: int) -> int:
        """Store a view as a new pack and link it to the canonical."""
        key = self._rng.standard_normal(8).astype(np.float32)
        value = self._rng.standard_normal(8).astype(np.float32)
        vid = self._engine.mem_write_pack(
            self._owner,
            key,
            [(0, value)],
            50.0,
            text=view_text,
        )
        self._engine.add_pack_edge(vid, canonical_pack_id, EDGE_SUPPORTS)
        return vid
