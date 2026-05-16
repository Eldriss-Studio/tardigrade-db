"""``JustifyThenJudgeEvaluator`` — retrieve → answer → justify → judge.

The canonical leaderboard pipeline used by Mem0, Memobase, and
ByteRover 2.0 for LoCoMo evaluation. Adds a *justify* stage between
the adapter's generated answer and the final LLM judgment: the
justifier produces a reasoning trace explaining why the answer
follows (or doesn't follow) from the evidence, which is then attached
to the judge prompt. Empirically this lets the judge separate
"semantically equivalent paraphrase" from "topical but wrong" — both
of which a plain ``(question, ground_truth, answer)`` prompt confuses.

## Pattern stack

* **Strategy** — this evaluator plugs into the same slot as
  :class:`tdb_bench.evaluators.llm.LLMGatedEvaluator`.
* **Template Method** — :class:`JustifyPromptBuilder` and
  :class:`JudgeWithJustificationPromptBuilder` own their prompt
  templates with versioned constants for cache-key correctness.
* **Chain of Responsibility** — both the justify and judge stages
  iterate over a list of :class:`JudgeProvider` instances; the first
  available + non-erroring provider wins per stage.
* **Graceful degradation** — failures in the justify stage fall back
  to plain judgment (no justification in the prompt); failures in
  *both* stages fall back to deterministic word-overlap scoring.

## SOLID

* SRP — prompt builders, the two-stage flow, and the LLM transport
  (``JudgeProvider``) are separate concerns.
* OCP — new providers plug in as new ``JudgeProvider`` impls. New
  prompt templates increment the ``*_TEMPLATE_VERSION`` constants
  rather than editing in place.
* LSP — returns the same :class:`ScoreResult` shape any
  :class:`Evaluator` returns; the runner is agnostic.
"""

from __future__ import annotations

import re

from tdb_bench.contracts import Evaluator
from tdb_bench.models import BenchmarkItem, ScoreResult

from .deterministic import DeterministicEvaluator
from .providers import JudgeProvider


# ─── Constants (no magic values) ──────────────────────────────────────────

# Reasoning traces can be longer than the answer — the model needs
# room to step through the evidence. 512 picked because LoCoMo's
# multi-hop questions are 2-4 evidence chunks; ~500 tokens covers
# a clean rationale with bookkeeping for citations.
JUSTIFY_MAX_TOKENS = 512

# Judge stage stays at the cheap 60-token budget — it returns
# ``{"score": <number>}`` JSON only.
JUDGE_MAX_TOKENS = 60

# Template versions feed any future cache-key for prompt outputs.
# Bump when the template body changes so prior cached responses
# invalidate naturally. Format `v{N}-{ISO-date}`.
JUSTIFY_TEMPLATE_VERSION = "v1-2026-05-16"
JUDGE_WITH_JUSTIFICATION_TEMPLATE_VERSION = "v1-2026-05-16"

# ScoreResult.score >= this → judgment="llm_pass". Mirrors the
# threshold used by LLMGatedEvaluator so the two pipelines produce
# comparable judgment strings.
_PASS_THRESHOLD = 0.8

# Mode labels surfaced in ScoreResult.evaluator_mode. Reading order:
#   llm_justify_<provider>             — both stages succeeded
#   llm_justify_<provider>_no_justification — justify failed, judge ran
#   deterministic_fallback             — both stages failed
_MODE_JUSTIFY_PROVIDER_PREFIX = "llm_justify"
_MODE_NO_JUSTIFICATION_SUFFIX = "no_justification"
_MODE_DETERMINISTIC_FALLBACK = "deterministic_fallback"


# ─── Prompt builders (Template Method) ────────────────────────────────────


_JUSTIFY_TEMPLATE = (
    "You are a careful reasoning aid. Given a question, retrieved evidence, "
    "and a candidate answer, explain whether and how the evidence supports "
    "the answer. Walk through the evidence step by step. If the evidence "
    "lets you derive the answer (including via straightforward inference "
    "like resolving 'two days ago' against a session timestamp), justify "
    "it. If the evidence does not support the answer, explain why.\n\n"
    "Evidence:\n{evidence_block}\n\n"
    "Question: {question}\n"
    "Candidate answer: {answer}\n"
    "Reasoning trace:"
)


class JustifyPromptBuilder:
    """Builds the prompt for the justify stage."""

    def build(self, *, question: str, evidence: list[str], answer: str) -> str:
        if evidence:
            evidence_block = "\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(evidence))
        else:
            evidence_block = "(no evidence retrieved)"
        return _JUSTIFY_TEMPLATE.format(
            evidence_block=evidence_block, question=question, answer=answer
        )

    @staticmethod
    def template_version() -> str:
        return JUSTIFY_TEMPLATE_VERSION


_JUDGE_WITH_JUSTIFICATION_TEMPLATE = (
    'Score answer correctness from 0.0 to 1.0 as JSON {{"score": number}}.\n'
    "Use the reasoning trace as an aid; the score is on the answer vs the "
    "ground truth, not the trace.\n\n"
    "Question: {question}\n"
    "Ground truth: {ground_truth}\n"
    "Answer: {answer}\n"
    "Reasoning trace: {justification}"
)


class JudgeWithJustificationPromptBuilder:
    """Builds the prompt for the judge stage when a justification is available."""

    def build(self, *, question: str, ground_truth: str, answer: str, justification: str) -> str:
        return _JUDGE_WITH_JUSTIFICATION_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            answer=answer,
            justification=justification,
        )

    @staticmethod
    def template_version() -> str:
        return JUDGE_WITH_JUSTIFICATION_TEMPLATE_VERSION


# ─── Two-stage evaluator ──────────────────────────────────────────────────


_PLAIN_JUDGE_TEMPLATE = (
    'Score answer correctness from 0.0 to 1.0 as JSON {{"score": number}}.\n'
    "Question: {question}\n"
    "Ground truth: {ground_truth}\n"
    "Answer: {answer}"
)


class JustifyThenJudgeEvaluator(Evaluator):
    """Strategy: ``retrieve → answer → justify → judge`` evaluator."""

    def __init__(
        self,
        *,
        justify_providers: list[JudgeProvider],
        judge_providers: list[JudgeProvider],
        fallback: Evaluator | None = None,
        justify_prompt_builder: JustifyPromptBuilder | None = None,
        judge_prompt_builder: JudgeWithJustificationPromptBuilder | None = None,
    ) -> None:
        self._justify_providers = justify_providers
        self._judge_providers = judge_providers
        self._fallback = fallback or DeterministicEvaluator()
        self._justify_prompt_builder = justify_prompt_builder or JustifyPromptBuilder()
        self._judge_prompt_builder = (
            judge_prompt_builder or JudgeWithJustificationPromptBuilder()
        )

    def score(
        self, item: BenchmarkItem, answer: str, evidence: list[str]
    ) -> ScoreResult:
        justify_text, justify_provider = self._run_stage(
            providers=self._justify_providers,
            prompt=self._justify_prompt_builder.build(
                question=item.question, evidence=evidence, answer=answer
            ),
            max_tokens=JUSTIFY_MAX_TOKENS,
        )

        if justify_text is not None:
            judge_prompt = self._judge_prompt_builder.build(
                question=item.question,
                ground_truth=item.ground_truth,
                answer=answer,
                justification=justify_text,
            )
            mode_base = f"{_MODE_JUSTIFY_PROVIDER_PREFIX}_{justify_provider}"
        else:
            judge_prompt = _PLAIN_JUDGE_TEMPLATE.format(
                question=item.question,
                ground_truth=item.ground_truth,
                answer=answer,
            )
            mode_base = f"{_MODE_JUSTIFY_PROVIDER_PREFIX}_{_MODE_NO_JUSTIFICATION_SUFFIX}"

        judge_text, judge_provider = self._run_stage(
            providers=self._judge_providers,
            prompt=judge_prompt,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        if judge_text is None:
            fb = self._fallback.score(item, answer, evidence)
            return ScoreResult(
                score=fb.score,
                judgment=f"{fb.judgment}_fallback",
                evaluator_mode=_MODE_DETERMINISTIC_FALLBACK,
            )

        score = _parse_score(judge_text)
        verdict = "pass" if score >= _PASS_THRESHOLD else "fail"
        return ScoreResult(
            score=score,
            judgment=f"llm_{verdict}",
            evaluator_mode=f"{mode_base}_judge_{judge_provider}"
            if justify_text is not None
            else mode_base,
        )

    @staticmethod
    def _run_stage(
        *, providers: list[JudgeProvider], prompt: str, max_tokens: int
    ) -> tuple[str | None, str]:
        """Iterate providers; return ``(text, provider_name)`` or ``(None, "")``."""
        for provider in providers:
            if not provider.is_available():
                continue
            try:
                text = provider.judge(prompt, max_tokens=max_tokens)
                return text, provider.name()
            except Exception:
                continue
        return None, ""


def _parse_score(text: str) -> float:
    """Extract numeric score from judge response; 0.0 on parse failure.

    Matches the same JSON shape as :mod:`tdb_bench.evaluators.llm`
    so downstream tooling can compare modes side-by-side.
    """
    if not text:
        return 0.0
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text)
    match = re.search(r'\{\s*"score"\s*:\s*([0-9]*\.?[0-9]+)\s*\}', cleaned)
    if not match:
        return 0.0
    score = float(match.group(1))
    return max(0.0, min(1.0, round(score, 6)))
