"""Multi-view text framing for memory consolidation.

Strategy pattern: each ``FramingStrategy`` rewrites a memory's source
text into an alternative surface form.  The engine stores each view as
a separate pack linked to the canonical memory via a Supports edge, so
queries phrased in *any* framing can discover the same underlying fact.

Rule-based v0 — no LLM dependency.  Individual strategies can be
upgraded to model-powered later without changing the generator interface.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .constants import (
    DEFAULT_VIEW_FRAMINGS,
    LLM_VIEW_PROMPT_TEMPLATE,
    MAX_VIEW_CANDIDATES,
    MAX_VIEWS_KEPT,
    VIEW_DIVERSITY_THRESHOLD,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class FramingStrategy(ABC):
    """Produces an alternative text surface for a memory fact."""

    @abstractmethod
    def reframe(self, text: str) -> str: ...


# ---------------------------------------------------------------------------
# Concrete strategies (rule-based v0)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the is was were are been be am being "
    "in on at to for of by from with and or but "
    "that this it its he she they them his her "
    "do did does has had have will would shall "
    "can could may might not no so if as".split()
)


class SummaryFraming(FramingStrategy):
    """Extractive summary: keeps the first clause up to the first
    comma, semicolon, or period-after-minimum-length."""

    def reframe(self, text: str) -> str:
        stripped = text.strip()
        for sep in (",", ";", " — ", " - "):
            idx = stripped.find(sep)
            if idx > 10:
                return stripped[:idx].strip()
        if len(stripped) > 60:
            boundary = stripped.rfind(" ", 0, 60)
            if boundary > 10:
                return stripped[:boundary].strip()
        return stripped[:60].strip() if len(stripped) > 60 else stripped


class QuestionFraming(FramingStrategy):
    """Generates a question the memory could answer.

    Extracts the subject via simple heuristics and wraps it in a
    WH-question template.
    """

    _TEMPLATES = (
        "What did {subject} do?",
        "What is known about {subject}?",
        "What happened involving {subject}?",
    )

    def reframe(self, text: str) -> str:
        subject = self._extract_subject(text)
        template = self._select_template(text)
        return template.format(subject=subject)

    def _extract_subject(self, text: str) -> str:
        words = text.strip().split()
        subject_words: list[str] = []
        for w in words:
            clean = re.sub(r"[,;:.!?]$", "", w)
            if clean.lower() in _STOP_WORDS and subject_words:
                break
            subject_words.append(clean)
            if len(subject_words) >= 4:
                break
        return " ".join(subject_words) if subject_words else words[0]

    def _select_template(self, text: str) -> str:
        lower = text.lower()
        if any(verb in lower for verb in ("translated", "moved", "built", "created", "wrote")):
            return self._TEMPLATES[0]
        return self._TEMPLATES[1]


class ParaphraseFraming(FramingStrategy):
    """Produces a surface-reworded version by reordering clauses and
    replacing content words with synonyms from a small built-in map."""

    _SYNONYMS: dict[str, str] = {
        "translated": "converted",
        "pharmaceutical": "medical",
        "patent": "document",
        "startup": "company",
        "built": "constructed",
        "moved": "relocated",
        "created": "developed",
        "large": "big",
        "small": "little",
        "important": "significant",
        "research": "study",
        "wrote": "authored",
        "discovered": "found",
        "designed": "engineered",
        "published": "released",
    }

    def reframe(self, text: str) -> str:
        clauses = re.split(r"[,;]\s*", text.strip())
        if len(clauses) > 1:
            clauses = list(reversed(clauses))
        reworded = ", ".join(clauses)
        for original, replacement in self._SYNONYMS.items():
            reworded = re.sub(
                rf"\b{re.escape(original)}\b",
                replacement,
                reworded,
                flags=re.IGNORECASE,
            )
        return reworded


class LLMQuestionFraming(FramingStrategy):
    """LLM-powered question generation (HyPE pattern)."""

    def reframe(self, text: str, model=None, tokenizer=None) -> str:
        if model is None or tokenizer is None:
            raise ValueError("LLMQuestionFraming requires model and tokenizer")
        import torch
        prompt = LLM_VIEW_PROMPT_TEMPLATE.format(fact_text=text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=60,
                do_sample=True, temperature=0.7, top_p=0.9,
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        question = generated.strip().split("\n")[0].strip()
        if not question.endswith("?"):
            question += "?"
        return question


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FRAMINGS: dict[str, type[FramingStrategy]] = {
    "summary": SummaryFraming,
    "question": QuestionFraming,
    "paraphrase": ParaphraseFraming,
    "llm_question": LLMQuestionFraming,
}


# ---------------------------------------------------------------------------
# Diversity filter
# ---------------------------------------------------------------------------


def filter_diverse(
    candidates: list,
    threshold: float = VIEW_DIVERSITY_THRESHOLD,
    max_kept: int = MAX_VIEWS_KEPT,
) -> list:
    """Keep only candidates whose mean vectors are pairwise below threshold."""
    if not candidates:
        return []
    import numpy as np

    def _mean(v):
        return v.mean(axis=0) if v.ndim > 1 else v

    def _cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    kept, kept_means = [], []
    for c in candidates:
        c_mean = _mean(c)
        if all(_cosine(c_mean, km) < threshold for km in kept_means):
            kept.append(c)
            kept_means.append(c_mean)
        if len(kept) >= max_kept:
            break
    return kept


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class ViewGenerator:
    """Produces multiple text views of a single memory fact.

    Each view is generated by a registered ``FramingStrategy``.  Operates
    in two modes:

    - ``"rule"`` (default): rule-based framings, no model required.
    - ``"llm"``: LLM-powered question generation (HyPE pattern); requires
      ``model`` and ``tokenizer`` to be provided.
    """

    def __init__(
        self,
        *,
        model=None,
        tokenizer=None,
        framings: Sequence[str] = DEFAULT_VIEW_FRAMINGS,
        mode: str = "rule",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mode = mode
        self._framings: dict[str, FramingStrategy] = {}
        if mode == "llm":
            self._framings["llm_question"] = LLMQuestionFraming()
        else:
            for name in framings:
                cls = _FRAMINGS.get(name)
                if cls is None:
                    raise ValueError(
                        f"Unknown framing '{name}'. "
                        f"Available: {', '.join(sorted(_FRAMINGS))}"
                    )
                self._framings[name] = cls()

    def generate(self, text: str | None) -> list[str]:
        """Return one reframed view per configured framing strategy.

        Returns an empty list for empty / None / whitespace-only input.
        """
        if not text or not text.strip():
            return []
        if self.mode == "llm":
            return self._generate_llm(text)
        views: list[str] = []
        for strategy in self._framings.values():
            views.append(strategy.reframe(text))
        return views

    def _generate_llm(self, text: str) -> list[str]:
        """Generate up to ``MAX_VIEWS_KEPT`` diverse questions via LLM."""
        strategy = self._framings["llm_question"]
        candidates = []
        for _ in range(MAX_VIEW_CANDIDATES):
            q = strategy.reframe(text, model=self.model, tokenizer=self.tokenizer)
            candidates.append(q)
        return candidates[:MAX_VIEWS_KEPT]
