"""ATDD: EvidenceFormatter (Slice L2).

Sits between the inner adapter's ``AdapterQueryResult.evidence``
(``list[str]``) and the prompt. Filter, cap, preserve rank order.
"""

from __future__ import annotations

from tdb_bench.answerers import EvidenceFormatter
from tdb_bench.answerers.constants import LLM_GATE_EVIDENCE_TOP_K


class TestEvidenceFormatter:
    def test_caps_to_top_k(self):
        evidence = [f"chunk-{i}" for i in range(10)]
        result = EvidenceFormatter().format(evidence, top_k=5)
        assert len(result) == 5

    def test_preserves_rank_order(self):
        evidence = ["first", "second", "third", "fourth"]
        result = EvidenceFormatter().format(evidence, top_k=3)
        assert result == ["first", "second", "third"]

    def test_skips_empty_strings(self):
        evidence = ["a", "", "b", "", "c"]
        result = EvidenceFormatter().format(evidence, top_k=5)
        assert result == ["a", "b", "c"]

    def test_skips_whitespace_only(self):
        evidence = ["a", "   ", "\n\t", "b"]
        result = EvidenceFormatter().format(evidence, top_k=5)
        assert result == ["a", "b"]

    def test_skips_none(self):
        # Mixed list with None entries — defensive against upstream
        # adapters that return Optional[str] in evidence (some do).
        evidence = ["a", None, "b", None]  # type: ignore[list-item]
        result = EvidenceFormatter().format(evidence, top_k=5)
        assert result == ["a", "b"]

    def test_deduplicates_consecutive_repeats(self):
        # Same chunk appearing twice in top-k wastes prompt budget.
        # Drop adjacent duplicates; preserve order otherwise.
        evidence = ["a", "a", "b", "a", "c"]
        result = EvidenceFormatter().format(evidence, top_k=5)
        # Adjacent "a"/"a" collapses; later non-adjacent "a" is kept.
        assert result == ["a", "b", "a", "c"]

    def test_top_k_larger_than_evidence_returns_all_valid(self):
        evidence = ["a", "b"]
        result = EvidenceFormatter().format(evidence, top_k=10)
        assert result == ["a", "b"]

    def test_top_k_zero_returns_empty(self):
        evidence = ["a", "b"]
        result = EvidenceFormatter().format(evidence, top_k=0)
        assert result == []

    def test_empty_input_returns_empty(self):
        assert EvidenceFormatter().format([], top_k=5) == []

    def test_default_top_k_from_constants(self):
        # No top_k arg → uses the module constant. Avoids magic-number
        # drift between adapter wiring and formatter behavior.
        evidence = [f"chunk-{i}" for i in range(LLM_GATE_EVIDENCE_TOP_K + 5)]
        result = EvidenceFormatter().format(evidence)
        assert len(result) == LLM_GATE_EVIDENCE_TOP_K
