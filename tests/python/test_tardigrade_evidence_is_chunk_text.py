"""ATDD: TardigradeAdapter returns *chunk* text as evidence, not the
parent item's full context (Phase 1B audit 2026-05-16 #97).

The reranker already correctly uses ``_cell_to_chunk_text`` to score
candidates. But the evidence path that builds the result handed to
the LLM still used ``mapped.context`` — the parent BenchmarkItem's
full conversation (~62K chars on full-conv LoCoMo). With top_k=20
chunks from the same item that meant ~1.25M chars in the prompt,
exceeding DeepSeek's 64K context and hiding the answer-bearing text.

Pin the contract on the small extracted helper so the test is
hermetic — no model, tokenizer, or engine needed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tdb_bench.adapters.tardigrade import TardigradeAdapter  # noqa: E402
from tdb_bench.models import BenchmarkItem  # noqa: E402


@dataclass
class _Handle:
    cell_id: int


def _make_adapter() -> TardigradeAdapter:
    """Build an adapter shell without invoking model loading.

    The helper under test only touches ``_cell_to_item`` and
    ``_cell_to_chunk_text`` — both populated post-construction —
    so bypassing ``__init__`` keeps the test isolated.
    """
    adapter = TardigradeAdapter.__new__(TardigradeAdapter)
    adapter._cell_to_item = {}
    adapter._cell_to_chunk_text = {}
    return adapter


def _item(text: str = "FULL_CONVERSATION_CONTEXT") -> BenchmarkItem:
    return BenchmarkItem(
        item_id="i-1",
        dataset="locomo",
        context=text,
        question="?",
        ground_truth="answer-1",
    )


class TestEvidenceIsChunkText:
    def test_returns_chunk_text_when_tracked(self):
        adapter = _make_adapter()
        item = _item("PARENT_CONTEXT_DO_NOT_USE")
        adapter._cell_to_item[10] = item
        adapter._cell_to_item[11] = item
        adapter._cell_to_chunk_text[10] = "chunk-A specific"
        adapter._cell_to_chunk_text[11] = "chunk-B specific"

        evidence, answer = adapter._build_evidence_and_answer(
            [_Handle(10), _Handle(11)], top_k=5,
        )
        assert evidence == ["chunk-A specific", "chunk-B specific"]
        # Parent context must NOT leak through.
        for e in evidence:
            assert "PARENT_CONTEXT_DO_NOT_USE" not in e
        assert answer == "answer-1"

    def test_falls_back_to_parent_context_when_chunk_text_missing(self):
        # Lexical-RLS fallback path never ingests through the chunker
        # and therefore never populates _cell_to_chunk_text. The
        # helper must still return *something* useful in that case
        # (parent context) — substring matchers downstream can still
        # find answers there.
        adapter = _make_adapter()
        item = _item("FALLBACK_PARENT")
        adapter._cell_to_item[20] = item
        # No chunk text tracked.

        evidence, answer = adapter._build_evidence_and_answer(
            [_Handle(20)], top_k=5,
        )
        assert evidence == ["FALLBACK_PARENT"]
        assert answer == "answer-1"

    def test_blank_chunk_text_falls_back_to_parent_context(self):
        # Empty string in chunk_text dict should fall back too —
        # otherwise the LLM gets blank evidence and refuses.
        adapter = _make_adapter()
        item = _item("FALLBACK_PARENT")
        adapter._cell_to_item[30] = item
        adapter._cell_to_chunk_text[30] = ""

        evidence, _ = adapter._build_evidence_and_answer(
            [_Handle(30)], top_k=5,
        )
        assert evidence == ["FALLBACK_PARENT"]

    def test_top_k_cap_applied(self):
        adapter = _make_adapter()
        item = _item()
        for i in range(10):
            adapter._cell_to_item[i] = item
            adapter._cell_to_chunk_text[i] = f"chunk-{i}"

        evidence, _ = adapter._build_evidence_and_answer(
            [_Handle(i) for i in range(10)], top_k=3,
        )
        assert evidence == ["chunk-0", "chunk-1", "chunk-2"]

    def test_unmapped_cell_id_is_skipped(self):
        adapter = _make_adapter()
        item = _item()
        adapter._cell_to_item[1] = item
        adapter._cell_to_chunk_text[1] = "kept"
        # cell_id 999 isn't in _cell_to_item — skip it.
        evidence, answer = adapter._build_evidence_and_answer(
            [_Handle(999), _Handle(1)], top_k=5,
        )
        assert evidence == ["kept"]
        assert answer == "answer-1"

    def test_empty_handles_returns_empty(self):
        adapter = _make_adapter()
        evidence, answer = adapter._build_evidence_and_answer([], top_k=5)
        assert evidence == []
        assert answer == ""

    def test_duplicate_chunk_text_emitted_only_once(self):
        # Smoke #5 (Phase 1B audit 2026-05-16 #98) found that LoCoMo's
        # prep emits one row per QA, so a conversation with N QAs gets
        # ingested N times → identical chunk text under N distinct
        # cell IDs. Top-5 was returning 5 copies of the same chunk.
        adapter = _make_adapter()
        item = _item()
        # Five distinct cells, all pointing to the same chunk text.
        for i in range(1, 6):
            adapter._cell_to_item[i] = item
            adapter._cell_to_chunk_text[i] = "duplicated chunk text"
        # And a unique sixth.
        adapter._cell_to_item[6] = item
        adapter._cell_to_chunk_text[6] = "unique chunk text"

        evidence, _ = adapter._build_evidence_and_answer(
            [_Handle(i) for i in range(1, 7)], top_k=5,
        )
        # Dedup keeps each distinct text once, then yields the unique
        # sixth in the same top-5 budget. No 5x repetition.
        assert evidence == ["duplicated chunk text", "unique chunk text"]

    def test_dedup_processes_more_handles_until_top_k_distinct(self):
        # Dedup must look past duplicates to find top_k distinct items
        # rather than returning fewer than top_k.
        adapter = _make_adapter()
        item = _item()
        adapter._cell_to_item[1] = item
        adapter._cell_to_chunk_text[1] = "alpha"
        adapter._cell_to_item[2] = item
        adapter._cell_to_chunk_text[2] = "alpha"  # dup
        adapter._cell_to_item[3] = item
        adapter._cell_to_chunk_text[3] = "alpha"  # dup
        adapter._cell_to_item[4] = item
        adapter._cell_to_chunk_text[4] = "beta"
        adapter._cell_to_item[5] = item
        adapter._cell_to_chunk_text[5] = "gamma"

        evidence, _ = adapter._build_evidence_and_answer(
            [_Handle(i) for i in range(1, 6)], top_k=3,
        )
        assert evidence == ["alpha", "beta", "gamma"]

    def test_answer_taken_from_first_mapped_item(self):
        # When multiple chunks come from different items, the answer
        # comes from the first one — preserves prior behaviour.
        adapter = _make_adapter()
        item_a = BenchmarkItem(
            item_id="a", dataset="locomo", context="ctx-A",
            question="?", ground_truth="ANSWER-A",
        )
        item_b = BenchmarkItem(
            item_id="b", dataset="locomo", context="ctx-B",
            question="?", ground_truth="ANSWER-B",
        )
        adapter._cell_to_item[1] = item_a
        adapter._cell_to_item[2] = item_b
        adapter._cell_to_chunk_text[1] = "chunk-a"
        adapter._cell_to_chunk_text[2] = "chunk-b"

        _, answer = adapter._build_evidence_and_answer(
            [_Handle(1), _Handle(2)], top_k=5,
        )
        assert answer == "ANSWER-A"
