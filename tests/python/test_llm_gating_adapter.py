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

    def test_evidence_passes_through_unchanged(self):
        inner = _StubAdapter(evidence=["alpha", "beta", "gamma"])
        outer = RetrieveThenReadAdapter(inner=inner, generator=MockAnswerGenerator("x"))

        result = outer.query(_item(), top_k=5)

        assert result.evidence == ["alpha", "beta", "gamma"]

    def test_inner_query_called_once_with_same_args(self):
        inner = _StubAdapter()
        outer = RetrieveThenReadAdapter(inner=inner, generator=MockAnswerGenerator("x"))
        item = _item()

        outer.query(item, top_k=7)

        assert len(inner.query_calls) == 1
        called_item, called_top_k = inner.query_calls[0]
        assert called_item == item
        assert called_top_k == 7

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


class TestRetrieveThenReadDecoratorEvidenceCapping:
    def test_evidence_capped_in_prompt_but_not_in_result(self):
        # 12 evidence items; LLM_GATE_EVIDENCE_TOP_K = 10.
        # Prompt should contain 10; AdapterQueryResult.evidence should
        # carry all the inner adapter returned so downstream evaluator
        # has full retrieval context for scoring.
        evidence = [f"chunk-{i}" for i in range(12)]
        inner = _StubAdapter(evidence=evidence)
        gen = MockAnswerGenerator(canned="x")
        outer = RetrieveThenReadAdapter(inner=inner, generator=gen)

        result = outer.query(_item(), top_k=5)

        assert len(result.evidence) == 12  # unchanged passthrough
        assert "chunk-0" in gen.last_prompt
        assert "chunk-9" in gen.last_prompt
        # chunk-10 and chunk-11 should NOT appear (capped at 10).
        assert "chunk-10" not in gen.last_prompt
        assert "chunk-11" not in gen.last_prompt


class TestRetrieveThenReadDecoratorSubstitutability:
    def test_is_a_benchmark_adapter(self):
        outer = RetrieveThenReadAdapter(
            inner=_StubAdapter(), generator=NoOpAnswerGenerator()
        )
        assert isinstance(outer, BenchmarkAdapter)
