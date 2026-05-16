"""ATDD: PromptBuilder + MockAnswerGenerator (Slice L1).

The prompt is part of the experiment — it gets versioned via
``PROMPT_TEMPLATE_VERSION`` so cache keys and reproducibility tags pin
to a specific template. Tests below pin:

* the question and evidence both appear verbatim;
* the prompt is deterministic (same inputs → same SHA-256);
* changing inputs changes the hash;
* the answer instruction is present so the model knows the output
  contract;
* ``MockAnswerGenerator`` returns the configured deterministic answer
  regardless of prompt content, and records calls for assertion.
"""

from __future__ import annotations

import hashlib

import pytest

from tdb_bench.answerers import (
    AnswerGenerator,
    MockAnswerGenerator,
    NoOpAnswerGenerator,
    PromptBuilder,
)
from tdb_bench.answerers.constants import PROMPT_TEMPLATE_VERSION


_Q = "Who is Sonia's husband?"
_E = [
    "Sonia married Aaron in 2019 after a long engagement.",
    "Aaron and Sonia have a dog named Bowie.",
    "Sonia's mother lives in Buenos Aires.",
]


class TestPromptBuilder:
    def test_contains_question_verbatim(self):
        prompt = PromptBuilder().build(question=_Q, evidence=_E)
        assert _Q in prompt

    def test_contains_each_evidence_chunk_verbatim(self):
        prompt = PromptBuilder().build(question=_Q, evidence=_E)
        for chunk in _E:
            assert chunk in prompt

    def test_contains_answer_format_instruction(self):
        prompt = PromptBuilder().build(question=_Q, evidence=_E)
        # The model needs to know it must answer (not chat / not refuse).
        # The exact phrasing is internal but some directive must be present.
        lower = prompt.lower()
        assert any(token in lower for token in ("answer", "respond"))

    def test_is_deterministic_for_identical_inputs(self):
        a = PromptBuilder().build(question=_Q, evidence=_E)
        b = PromptBuilder().build(question=_Q, evidence=_E)
        assert a == b
        assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()

    def test_different_question_yields_different_prompt(self):
        a = PromptBuilder().build(question=_Q, evidence=_E)
        b = PromptBuilder().build(question="Who owns the dog?", evidence=_E)
        assert a != b

    def test_different_evidence_yields_different_prompt(self):
        a = PromptBuilder().build(question=_Q, evidence=_E)
        b = PromptBuilder().build(question=_Q, evidence=_E + ["Extra fact."])
        assert a != b

    def test_empty_evidence_still_builds(self):
        # Edge: retrieval returned zero hits. Prompt must still be valid
        # so the model can refuse rather than the harness crashing.
        prompt = PromptBuilder().build(question=_Q, evidence=[])
        assert _Q in prompt

    def test_exposes_template_version(self):
        # Cache keys depend on this — must be readable from the builder.
        assert PromptBuilder().template_version() == PROMPT_TEMPLATE_VERSION

    def test_permits_inference_from_evidence(self):
        # Phase 1B audit 2026-05-16 #94 — the v1 prompt's strict
        # "only the evidence below" wording caused IDK refusals on
        # items needing trivial inference (paraphrase, multi-line
        # combination, relative-date resolution). The v2 wording
        # permits cautious inference. The strict "only the evidence"
        # phrase must be gone, and an inference-permitting cue must
        # be present.
        prompt = PromptBuilder().build(question=_Q, evidence=_E).lower()
        assert "only the evidence below" not in prompt
        # Any of these signals the relaxation is in place.
        assert any(
            cue in prompt
            for cue in ("combine", "paraphrase", "resolve", "may")
        )

    def test_still_permits_refusal_when_evidence_lacks_answer(self):
        # The relaxation must NOT remove the IDK escape hatch — without
        # it the model hallucinates on out-of-evidence questions.
        prompt = PromptBuilder().build(question=_Q, evidence=_E).lower()
        assert "i don't know" in prompt

    def test_mentions_session_date_resolution(self):
        # Pairs with the full-conv-mode date-header injection
        # (commit a9ecf20). The prompt must teach the model how to
        # use those headers, otherwise the dates are dead bytes.
        prompt = PromptBuilder().build(question=_Q, evidence=_E).lower()
        assert "session" in prompt and "date" in prompt


class TestMockAnswerGenerator:
    def test_returns_configured_answer(self):
        gen = MockAnswerGenerator(canned="Aaron")
        assert gen.generate("any prompt") == "Aaron"

    def test_returns_same_canned_for_different_prompts(self):
        gen = MockAnswerGenerator(canned="ZZZ")
        assert gen.generate("prompt A") == "ZZZ"
        assert gen.generate("prompt B") == "ZZZ"

    def test_records_call_count(self):
        gen = MockAnswerGenerator(canned="x")
        gen.generate("a")
        gen.generate("b")
        gen.generate("c")
        assert gen.call_count == 3

    def test_records_last_prompt(self):
        gen = MockAnswerGenerator(canned="x")
        gen.generate("the last prompt")
        assert gen.last_prompt == "the last prompt"

    def test_is_substitutable_for_protocol(self):
        # Behavioral substitutability — caller depends only on .generate(str) -> str.
        gen: AnswerGenerator = MockAnswerGenerator(canned="ok")
        assert isinstance(gen.generate("x"), str)


class TestNoOpAnswerGenerator:
    def test_returns_empty_string(self):
        gen = NoOpAnswerGenerator()
        assert gen.generate("any prompt") == ""

    def test_is_substitutable(self):
        gen: AnswerGenerator = NoOpAnswerGenerator()
        assert isinstance(gen.generate("x"), str)


class TestAnswerGeneratorProtocol:
    def test_cannot_instantiate_protocol_directly(self):
        with pytest.raises(TypeError):
            AnswerGenerator()  # type: ignore[abstract]
