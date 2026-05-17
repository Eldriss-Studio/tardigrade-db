"""Filter + cap retrieved-evidence strings before they hit the prompt.

The Decorator adapter receives ``AdapterQueryResult.evidence``
from the inner adapter as a ``list[str]`` already (the inner adapter
does the cell-handle → text mapping). The formatter's job is the
slice between that raw list and the prompt:

* drop empty / whitespace / ``None`` entries (upstream adapters return
  some of these);
* drop *adjacent* duplicates (same chunk surfaced twice in top-k
  wastes prompt budget);
* cap to ``top_k`` so the prompt stays within the LLM's context budget
  and ``LLM_GATE_EVIDENCE_TOP_K`` controls the trade-off in one place.

Non-adjacent duplicates are *kept* — they encode rank-position
information the model can use ("this chunk appeared as #3 and #5").
"""

from __future__ import annotations

from .constants import LLM_GATE_EVIDENCE_TOP_K


class EvidenceFormatter:
    """Filter + cap retrieved evidence in rank order."""

    def format(
        self,
        evidence: list[str | None],
        top_k: int = LLM_GATE_EVIDENCE_TOP_K,
    ) -> list[str]:
        """Return up to ``top_k`` non-empty chunks, rank-ordered."""
        if top_k <= 0:
            return []

        result: list[str] = []
        for chunk in evidence:
            if chunk is None:
                continue
            if not chunk.strip():
                continue
            if result and result[-1] == chunk:
                continue
            result.append(chunk)
            if len(result) >= top_k:
                break
        return result
