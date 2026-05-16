"""``RetrieveThenReadAdapter`` — Decorator over any ``BenchmarkAdapter``.

Replaces the inner adapter's ``answer = mapped.ground_truth``
shortcut with a real LLM-generated answer from
``(question, retrieved evidence)`` — the canonical retrieve-then-read
protocol used by every memory-system leaderboard (Mem0, Memobase,
ByteRover, MemMachine).

## Decorator-owned retrieval budget

The Decorator widens its inner retrieval call to
``LLM_GATE_INNER_TOP_K`` (env-overridable, default 25) so the LLM has
enough material to answer. Measurement: item-level R@25 ≈ 84% on
LoCoMo vs R@5 ≈ 30% (Phase 1B.2 audit) — feeding only top-5 caused
~70% of generated answers to be "I don't know."

The widening is **private to the Decorator**. The bench runner still
calls ``query(item, top_k=N)`` with the profile-level ``top_k``, and
fairness validation (``python/tdb_bench/fairness.py``) sees the same
``top_k`` for every system. What the Decorator does internally with
the inner adapter is below that abstraction line.

The returned ``AdapterQueryResult.evidence`` is **capped to the
runner's ``top_k``** so the run JSON stays symmetric with non-gated
systems — anyone reading the JSON sees the same number of evidence
chunks per row across all systems.

The prompt sees up to ``LLM_GATE_PROMPT_TOP_K`` chunks (default 10).
That value can be smaller than ``LLM_GATE_INNER_TOP_K`` (limit
prompt size) or equal (use everything retrieved); both are valid.

## Responsibilities

1. Pull ``LLM_GATE_INNER_TOP_K`` candidates from the inner adapter.
2. Cap the prompt's evidence to ``LLM_GATE_PROMPT_TOP_K`` via
   :class:`EvidenceFormatter`.
3. Assemble the prompt via :class:`PromptBuilder`.
4. Ask the configured :class:`AnswerGenerator` (Strategy).
5. Cap the serialized ``result.evidence`` to the runner's ``top_k``.
6. Sum-record retrieval + generation latency.
7. Failure isolation — a generator exception is recorded as
   ``status="failed"``/``error="generator_failed:<...>"`` per item
   rather than aborting the run.

## Pattern stack

    Decorator (this class) → Strategy (AnswerGenerator) →
    Template Method (PromptBuilder) → Repository (EvidenceFormatter)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from tdb_bench.answerers import (
    AnswerGenerator,
    EvidenceFormatter,
    PromptBuilder,
)
from tdb_bench.answerers.constants import (
    LLM_GATE_INNER_TOP_K,
    LLM_GATE_PROMPT_TOP_K,
)
from tdb_bench.contracts import BenchmarkAdapter
from tdb_bench.models import AdapterQueryResult, BenchmarkItem


@dataclass(frozen=True)
class RetrievalResult:
    """Output of the *retrieve* half of the two-phase pipeline.

    The runner's parallel codepath calls :meth:`RetrieveThenReadAdapter.retrieve`
    sequentially (GPU-bound) and then dispatches
    :meth:`RetrieveThenReadAdapter.generate_answer` calls to a thread pool
    (LLM-bound). This dataclass carries the state between the two phases.
    """

    inner_evidence: list[str]
    inner_status: str
    inner_error: str | None
    retrieval_latency_ms: float


_NO_EVIDENCE_ERROR = "no_evidence_for_generation"
_GENERATOR_FAILED_PREFIX = "generator_failed"

_INNER_TOP_K_ENV = "TDB_LLM_GATE_INNER_TOP_K"
_PROMPT_TOP_K_ENV = "TDB_LLM_GATE_PROMPT_TOP_K"


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
        inner_top_k: int | None = None,
        prompt_top_k: int | None = None,
    ) -> None:
        self._inner = inner
        self._generator = generator
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._evidence_formatter = evidence_formatter or EvidenceFormatter()
        self._answerer_model = answerer_model
        self._inner_top_k = _resolve_int_env(
            inner_top_k, _INNER_TOP_K_ENV, LLM_GATE_INNER_TOP_K
        )
        self._prompt_top_k = _resolve_int_env(
            prompt_top_k, _PROMPT_TOP_K_ENV, LLM_GATE_PROMPT_TOP_K
        )

    # --- pass-through ---

    def ingest(self, items: list[BenchmarkItem]) -> None:
        self._inner.ingest(items)

    def reset(self) -> None:
        self._inner.reset()

    def metadata(self) -> dict[str, str]:
        merged = dict(self._inner.metadata())
        merged["answerer_model"] = self._answerer_model
        merged["prompt_template_version"] = self._prompt_builder.template_version()
        merged["inner_top_k"] = str(self._inner_top_k)
        merged["prompt_top_k"] = str(self._prompt_top_k)
        return merged

    # --- decorated query (two-phase, composable) ---

    def retrieve(self, item: BenchmarkItem, top_k: int) -> RetrievalResult:  # noqa: ARG002
        """Phase 1: GPU-bound retrieval. Safe to call serially with a GPU lock."""
        inner = self._inner.query(item, self._inner_top_k)
        return RetrievalResult(
            inner_evidence=inner.evidence,
            inner_status=inner.status,
            inner_error=inner.error,
            retrieval_latency_ms=inner.latency_ms,
        )

    def generate_answer(
        self,
        *,
        item: BenchmarkItem,
        inner_evidence: list[str],
        inner_status: str,
        inner_error: str | None,
        retrieval_latency_ms: float,
        outer_top_k: int,
    ) -> AdapterQueryResult:
        """Phase 2: LLM-bound answer generation. Safe to call concurrently."""
        if inner_status != "ok":
            return AdapterQueryResult(
                answer="",
                evidence=_cap_evidence(inner_evidence, outer_top_k),
                latency_ms=retrieval_latency_ms,
                status=inner_status,
                error=inner_error,
            )

        prompt_evidence = self._evidence_formatter.format(
            inner_evidence, top_k=self._prompt_top_k
        )
        if not prompt_evidence:
            return AdapterQueryResult(
                answer="",
                evidence=_cap_evidence(inner_evidence, outer_top_k),
                latency_ms=retrieval_latency_ms,
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
                evidence=_cap_evidence(inner_evidence, outer_top_k),
                latency_ms=retrieval_latency_ms + gen_ms,
                status="failed",
                error=f"{_GENERATOR_FAILED_PREFIX}:{type(exc).__name__}",
            )
        gen_ms = (time.perf_counter() - gen_start) * 1000.0

        return AdapterQueryResult(
            answer=answer,
            evidence=_cap_evidence(inner_evidence, outer_top_k),
            latency_ms=retrieval_latency_ms + gen_ms,
            status="ok",
            error=None,
        )

    def query(self, item: BenchmarkItem, top_k: int) -> AdapterQueryResult:
        """Convenience composition of :meth:`retrieve` and :meth:`generate_answer`.

        Preserves the original single-call contract for the
        non-parallel runner codepath and the existing tests.
        """
        ret = self.retrieve(item, top_k)
        return self.generate_answer(
            item=item,
            inner_evidence=ret.inner_evidence,
            inner_status=ret.inner_status,
            inner_error=ret.inner_error,
            retrieval_latency_ms=ret.retrieval_latency_ms,
            outer_top_k=top_k,
        )


def _resolve_int_env(explicit: int | None, env_var: str, default: int) -> int:
    """Pick explicit > env > default and validate positive."""
    if explicit is not None:
        value = explicit
    else:
        raw = os.getenv(env_var, "").strip()
        value = int(raw) if raw else default
    if value < 1:
        raise ValueError(f"{env_var or 'budget'} must be >= 1; got {value}")
    return value


def _cap_evidence(evidence: list[str], outer_top_k: int) -> list[str]:
    """Cap serialized evidence at the runner's ``top_k`` for JSON symmetry."""
    if outer_top_k <= 0:
        return []
    return evidence[:outer_top_k]
