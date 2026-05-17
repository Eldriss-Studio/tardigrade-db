"""LLM answer generation for retrieve-then-read evaluation.

This module: protocol + mock + prompt builder.

The retrieve-then-read pipeline is the canonical evaluation protocol
for memory-augmented QA systems (LoCoMo, LongMemEval, BEIR-derived
benchmarks). Every leaderboard system uses it:

    retrieve evidence → LLM generates answer from (Q, evidence) → LLM-as-Judge

This package supplies the *answer generation* step. The judge step
already lives in ``tdb_bench.evaluators``.

Module layout:

* :class:`AnswerGenerator` — Strategy protocol (``generate(prompt) -> str``).
* :class:`MockAnswerGenerator` / :class:`NoOpAnswerGenerator` — test doubles.
* :class:`PromptBuilder` — Template Method assembling ``(question, evidence)``
  into a single prompt; pins :data:`PROMPT_TEMPLATE_VERSION` for cache keys.
"""

from __future__ import annotations

from .base import AnswerGenerator, GeneratorExhausted
from .cache import CachedAnswerGenerator
from .deepseek import DeepSeekAnswerer
from .evidence_formatter import EvidenceFormatter
from .factory import build_answerer_from_env
from .mock import MockAnswerGenerator, NoOpAnswerGenerator
from .openai import OpenAIAnswerer
from .openai_compatible import OpenAICompatibleAnswerer
from .prompt_builder import PromptBuilder
from .retry import RetryingGenerator

__all__ = [
    "AnswerGenerator",
    "CachedAnswerGenerator",
    "DeepSeekAnswerer",
    "EvidenceFormatter",
    "GeneratorExhausted",
    "MockAnswerGenerator",
    "NoOpAnswerGenerator",
    "OpenAIAnswerer",
    "OpenAICompatibleAnswerer",
    "PromptBuilder",
    "RetryingGenerator",
    "build_answerer_from_env",
]
