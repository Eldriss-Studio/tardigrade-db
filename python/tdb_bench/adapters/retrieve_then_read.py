"""``RetrieveThenReadAdapter`` — Decorator over any ``BenchmarkAdapter``.

Replaces the inner adapter's ``answer = mapped.ground_truth``
shortcut with a real LLM-generated answer from
``(question, retrieved evidence)`` — the canonical retrieve-then-read
protocol used by every memory-system leaderboard (Mem0, Memobase,
ByteRover, MemMachine).

The inner adapter still drives retrieval. The decorator owns:

1. Pulling ``AdapterQueryResult.evidence`` from the inner result.
2. Capping that evidence to ``LLM_GATE_EVIDENCE_TOP_K`` for the
   prompt (the full inner ``evidence`` list is *passed through* in
   the returned result so the evaluator scores against the full
   retrieval).
3. Assembling the prompt via :class:`PromptBuilder`.
4. Asking the configured :class:`AnswerGenerator` (Strategy).
5. Sum-recording retrieval + generation latency.
6. Failure isolation — a generator exception is recorded as
   ``status="failed"``/``error="generator_failed:<...>"`` per item
   rather than aborting the run.

Pattern stack:
    Decorator (this class) → Strategy (AnswerGenerator) →
    Template Method (PromptBuilder) → Repository (EvidenceFormatter)
"""

from __future__ import annotations

import time

from tdb_bench.answerers import (
    AnswerGenerator,
    EvidenceFormatter,
    PromptBuilder,
)
from tdb_bench.contracts import BenchmarkAdapter
from tdb_bench.models import AdapterQueryResult, BenchmarkItem


_NO_EVIDENCE_ERROR = "no_evidence_for_generation"
_GENERATOR_FAILED_PREFIX = "generator_failed"


class RetrieveThenReadAdapter(BenchmarkAdapter):
    """Decorator: turns retrieval evidence into an LLM-generated answer."""

    name = "tardigrade-llm-gated"

    def __init__(
        self,
        *,
        inner: BenchmarkAdapter,
        generator: AnswerGenerator,
        prompt_builder: PromptBuilder | None = None,
        evidence_formatter: EvidenceFormatter | None = None,
        answerer_model: str = "",
    ) -> None:
        self._inner = inner
        self._generator = generator
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._evidence_formatter = evidence_formatter or EvidenceFormatter()
        self._answerer_model = answerer_model

    # --- pass-through ---

    def ingest(self, items: list[BenchmarkItem]) -> None:
        self._inner.ingest(items)

    def reset(self) -> None:
        self._inner.reset()

    def metadata(self) -> dict[str, str]:
        merged = dict(self._inner.metadata())
        merged["answerer_model"] = self._answerer_model
        merged["prompt_template_version"] = self._prompt_builder.template_version()
        return merged

    # --- decorated query ---

    def query(self, item: BenchmarkItem, top_k: int) -> AdapterQueryResult:
        inner = self._inner.query(item, top_k)
        if inner.status != "ok":
            return inner

        prompt_evidence = self._evidence_formatter.format(inner.evidence)
        if not prompt_evidence:
            return AdapterQueryResult(
                answer="",
                evidence=inner.evidence,
                latency_ms=inner.latency_ms,
                status="failed",
                error=_NO_EVIDENCE_ERROR,
            )

        prompt = self._prompt_builder.build(
            question=item.question, evidence=prompt_evidence
        )
        gen_start = time.perf_counter()
        try:
            answer = self._generator.generate(prompt)
        except Exception as exc:  # noqa: BLE001 — generator failures
            gen_ms = (time.perf_counter() - gen_start) * 1000.0
            return AdapterQueryResult(
                answer="",
                evidence=inner.evidence,
                latency_ms=inner.latency_ms + gen_ms,
                status="failed",
                error=f"{_GENERATOR_FAILED_PREFIX}:{type(exc).__name__}",
            )
        gen_ms = (time.perf_counter() - gen_start) * 1000.0

        return AdapterQueryResult(
            answer=answer,
            evidence=inner.evidence,
            latency_ms=inner.latency_ms + gen_ms,
            status="ok",
            error=None,
        )
