"""ATDD: RetrieveThenReadAdapter (Slice L3).

Decorator wraps any ``BenchmarkAdapter`` and replaces the inner
adapter's ``answer`` with an LLM-generated string from
``(question, retrieved evidence)``. Everything else
(``ingest``, ``reset``, ``metadata``, evidence list, retrieval latency)
passes through.
"""

from __future__ import annotations

import time

import pytest

from tdb_bench.adapters.retrieve_then_read import RetrieveThenReadAdapter
from tdb_bench.answerers import (
    AnswerGenerator,
    MockAnswerGenerator,
    NoOpAnswerGenerator,
    PromptBuilder,
)
from tdb_bench.contracts import BenchmarkAdapter
from tdb_bench.models import AdapterQueryResult, BenchmarkItem


def _item(qid: str = "q1", question: str = "Who married Sonia?") -> BenchmarkItem:
    return BenchmarkItem(
        item_id=qid,
        dataset="locomo",
        context="ctx",
        question=question,
        ground_truth="Aaron",
    )


class _StubAdapter(BenchmarkAdapter):
    """Records calls; returns a configured result."""

    name = "stub"

    def __init__(
        self,
        answer: str = "STUB_ANSWER",
        evidence: list[str] | None = None,
        status: str = "ok",
        error: str | None = None,
        latency_ms: float = 7.0,
    ) -> None:
        self._answer = answer
        self._evidence = evidence if evidence is not None else ["e1", "e2", "e3"]
        self._status = status
        self._error = error
        self._latency_ms = latency_ms
        self.ingest_calls: list[list[BenchmarkItem]] = []
        self.query_calls: list[tuple[BenchmarkItem, int]] = []
        self.reset_count = 0

    def ingest(self, items: list[BenchmarkItem]) -> None:
        self.ingest_calls.append(items)

    def query(self, item: BenchmarkItem, top_k: int) -> AdapterQueryResult:
        self.query_calls.append((item, top_k))
        return AdapterQueryResult(
            answer=self._answer,
            evidence=list(self._evidence),
            latency_ms=self._latency_ms,
            status=self._status,
            error=self._error,
        )

    def reset(self) -> None:
        self.reset_count += 1

    def metadata(self) -> dict[str, str]:
        return {"system": "stub", "version": "0"}


class TestRetrieveThenReadDecoratorReplacesAnswer:
    def test_answer_is_replaced_by_generator_output(self):
        inner = _StubAdapter(answer="WRONG_ANSWER")
        gen = MockAnswerGenerator(canned="GENERATED")
        outer = RetrieveThenReadAdapter(inner=inner, generator=gen)

        result = outer.query(_item(), top_k=5)

        assert result.answer == "GENERATED"

    def test_evidence_passes_through_when_under_cap(self):
        # 3 chunks < outer top_k=5 → cap is a no-op; this test still
        # proves "Decorator does not mangle or reorder inner evidence."
        inner = _StubAdapter(evidence=["alpha", "beta", "gamma"])
        outer = RetrieveThenReadAdapter(inner=inner, generator=MockAnswerGenerator("x"))

        result = outer.query(_item(), top_k=5)

        assert result.evidence == ["alpha", "beta", "gamma"]

    def test_inner_query_called_once_with_same_item(self):
        # The outer top_k is intentionally widened to LLM_GATE_INNER_TOP_K
        # inside the Decorator; this test now pins the item-identity
        # passthrough (the widening invariant lives in
        # TestRetrieveThenReadDecoratorInnerBudget).
        inner = _StubAdapter()
        outer = RetrieveThenReadAdapter(inner=inner, generator=MockAnswerGenerator("x"))
        item = _item()

        outer.query(item, top_k=7)

        assert len(inner.query_calls) == 1
        called_item, _ = inner.query_calls[0]
        assert called_item == item

    def test_generator_receives_prompt_with_question_and_evidence(self):
        inner = _StubAdapter(evidence=["chunk-A", "chunk-B"])
        gen = MockAnswerGenerator(canned="x")
        outer = RetrieveThenReadAdapter(inner=inner, generator=gen)
        item = _item(question="What is X?")

        outer.query(item, top_k=5)

        assert gen.last_prompt is not None
        assert "What is X?" in gen.last_prompt
        assert "chunk-A" in gen.last_prompt
        assert "chunk-B" in gen.last_prompt


class TestRetrieveThenReadDecoratorLatency:
    def test_latency_includes_both_retrieval_and_generation(self):
        inner = _StubAdapter(latency_ms=50.0)

        class _SlowGen(AnswerGenerator):
            def generate(self, prompt: str) -> str:
                time.sleep(0.02)  # ~20ms
                return "x"

        outer = RetrieveThenReadAdapter(inner=inner, generator=_SlowGen())

        result = outer.query(_item(), top_k=5)

        # latency = retrieval (50) + generation (≥ ~20) — strict lower bound
        assert result.latency_ms >= 50.0 + 15.0  # 15 slack for sleep jitter


class TestRetrieveThenReadDecoratorFailurePropagation:
    def test_failed_inner_skips_generation(self):
        inner = _StubAdapter(status="failed", error="no_retrieval_match", answer="")
        gen = MockAnswerGenerator(canned="SHOULD_NOT_BE_CALLED")
        outer = RetrieveThenReadAdapter(inner=inner, generator=gen)

        result = outer.query(_item(), top_k=5)

        # No point asking the LLM to answer with no evidence — propagate
        # the failure so it shows up correctly in run aggregates.
        assert result.status == "failed"
        assert result.error == "no_retrieval_match"
        assert gen.call_count == 0

    def test_empty_evidence_skips_generation(self):
        inner = _StubAdapter(status="ok", evidence=[], answer="anything")
        gen = MockAnswerGenerator(canned="SHOULD_NOT_BE_CALLED")
        outer = RetrieveThenReadAdapter(inner=inner, generator=gen)

        result = outer.query(_item(), top_k=5)

        assert gen.call_count == 0
        assert result.answer == ""
        assert result.status == "failed"
        assert result.error == "no_evidence_for_generation"

    def test_generator_exception_becomes_failed_status(self):
        class _BrokenGen(AnswerGenerator):
            def generate(self, prompt: str) -> str:
                raise RuntimeError("API down")

        inner = _StubAdapter()
        outer = RetrieveThenReadAdapter(inner=inner, generator=_BrokenGen())

        result = outer.query(_item(), top_k=5)

        assert result.status == "failed"
        assert "generator_failed" in (result.error or "")
        # Evidence still surfaces — useful for retrieval-only metrics.
        assert len(result.evidence) > 0


class TestRetrieveThenReadDecoratorPassThrough:
    def test_ingest_passes_through(self):
        inner = _StubAdapter()
        outer = RetrieveThenReadAdapter(inner=inner, generator=NoOpAnswerGenerator())
        items = [_item("q1"), _item("q2")]

        outer.ingest(items)

        assert inner.ingest_calls == [items]

    def test_reset_passes_through(self):
        inner = _StubAdapter()
        outer = RetrieveThenReadAdapter(inner=inner, generator=NoOpAnswerGenerator())

        outer.reset()

        assert inner.reset_count == 1

    def test_metadata_merges_inner_with_decorator_metadata(self):
        inner = _StubAdapter()
        outer = RetrieveThenReadAdapter(inner=inner, generator=MockAnswerGenerator("x"))

        meta = outer.metadata()

        # Inner identity preserved
        assert meta["system"] == "stub"
        # Decorator metadata added so run records the pipeline
        assert "answerer_model" in meta
        assert "prompt_template_version" in meta


class TestRetrieveThenReadDecoratorEvidenceCapBothSides:
    """Slice E2 + E3 combined: evidence is capped on BOTH sides of the
    Decorator — the prompt sees up to ``LLM_GATE_PROMPT_TOP_K`` and the
    serialized ``result.evidence`` is capped to the runner's ``top_k`` for
    fair JSON reporting alongside non-gated systems.
    """

    def test_serialized_evidence_capped_to_outer_top_k(self):
        evidence = [f"chunk-{i}" for i in range(25)]
        inner = _StubAdapter(evidence=evidence)
        outer = RetrieveThenReadAdapter(inner=inner, generator=MockAnswerGenerator("x"))

        result = outer.query(_item(), top_k=5)

        assert len(result.evidence) == 5
        assert result.evidence == [f"chunk-{i}" for i in range(5)]

    def test_prompt_evidence_capped_to_prompt_top_k(self):
        # LLM_GATE_PROMPT_TOP_K = 10 (default).
        evidence = [f"chunk-{i}" for i in range(25)]
        inner = _StubAdapter(evidence=evidence)
        gen = MockAnswerGenerator(canned="x")
        outer = RetrieveThenReadAdapter(inner=inner, generator=gen)

        outer.query(_item(), top_k=5)

        for i in range(10):
            assert f"chunk-{i}" in gen.last_prompt
        for i in range(10, 25):
            assert f"chunk-{i}" not in gen.last_prompt


class TestRetrieveThenReadDecoratorInnerBudget:
    """Slice E1: the Decorator owns its retrieval budget and asks the
    inner adapter for ``LLM_GATE_INNER_TOP_K`` chunks regardless of the
    outer ``top_k`` the runner passes in.
    """

    def test_inner_query_widened_to_inner_top_k(self):
        from tdb_bench.answerers.constants import LLM_GATE_INNER_TOP_K

        inner = _StubAdapter()
        outer = RetrieveThenReadAdapter(inner=inner, generator=MockAnswerGenerator("x"))

        outer.query(_item(), top_k=5)

        assert len(inner.query_calls) == 1
        _, called_top_k = inner.query_calls[0]
        assert called_top_k == LLM_GATE_INNER_TOP_K


class TestRetrieveThenReadDecoratorEnvOverrides:
    """Slice E4: per-run knobs without touching the bench profile."""

    def test_inner_top_k_env_override(self, monkeypatch):
        monkeypatch.setenv("TDB_LLM_GATE_INNER_TOP_K", "12")
        inner = _StubAdapter()
        outer = RetrieveThenReadAdapter(inner=inner, generator=MockAnswerGenerator("x"))

        outer.query(_item(), top_k=5)

        _, called_top_k = inner.query_calls[0]
        assert called_top_k == 12

    def test_prompt_top_k_env_override(self, monkeypatch):
        monkeypatch.setenv("TDB_LLM_GATE_PROMPT_TOP_K", "4")
        evidence = [f"chunk-{i}" for i in range(20)]
        inner = _StubAdapter(evidence=evidence)
        gen = MockAnswerGenerator(canned="x")
        outer = RetrieveThenReadAdapter(inner=inner, generator=gen)

        outer.query(_item(), top_k=5)

        for i in range(4):
            assert f"chunk-{i}" in gen.last_prompt
        for i in range(4, 20):
            assert f"chunk-{i}" not in gen.last_prompt


class TestRetrieveThenReadDecoratorSubstitutability:
    def test_is_a_benchmark_adapter(self):
        outer = RetrieveThenReadAdapter(
            inner=_StubAdapter(), generator=NoOpAnswerGenerator()
        )
        assert isinstance(outer, BenchmarkAdapter)
