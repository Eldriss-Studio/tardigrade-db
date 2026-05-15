"""Template Method for assembling ``(question, evidence) → prompt``.

The prompt is part of the experiment. ``PROMPT_TEMPLATE_VERSION`` pins
the template so:

* response cache keys can detect prompt-template changes
  (Slice L7);
* run metadata records which template produced which results
  (reproducibility);
* prompt experiments are version-tracked rather than mutated in place.

The default template instructs the model to answer concisely from the
given evidence. Refinements happen in new ``PromptBuilder`` subclasses
with bumped ``PROMPT_TEMPLATE_VERSION`` — not by editing in place.
"""

from __future__ import annotations

from .constants import PROMPT_TEMPLATE_VERSION


# Format strings kept at module scope so the layout is one read away.
_ANSWER_INSTRUCTION = (
    "Answer the question using only the evidence below. "
    "Be concise — one short phrase or sentence. "
    "If the evidence does not contain the answer, respond with: I don't know."
)

_EVIDENCE_HEADER = "Evidence:"
_QUESTION_HEADER = "Question:"
_NO_EVIDENCE_MARKER = "(no evidence retrieved)"


class PromptBuilder:
    """Assembles the prompt for the answerer LLM.

    Implements Template Method: :meth:`build` is the fixed skeleton
    (instruction → evidence block → question → answer cue);
    subclasses can override individual section formatters to vary the
    surface form without changing the skeleton. Bump
    :data:`PROMPT_TEMPLATE_VERSION` in :mod:`.constants` whenever any
    visible byte of the output changes.
    """

    def build(self, *, question: str, evidence: list[str]) -> str:
        """Return the full prompt string for ``question`` + ``evidence``."""
        return "\n".join(
            [
                _ANSWER_INSTRUCTION,
                "",
                _EVIDENCE_HEADER,
                self._format_evidence(evidence),
                "",
                f"{_QUESTION_HEADER} {question}",
                "Answer:",
            ]
        )

    @staticmethod
    def template_version() -> str:
        """Pinned template version — feeds cache keys and run metadata."""
        return PROMPT_TEMPLATE_VERSION

    @staticmethod
    def _format_evidence(evidence: list[str]) -> str:
        if not evidence:
            return _NO_EVIDENCE_MARKER
        return "\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(evidence))
