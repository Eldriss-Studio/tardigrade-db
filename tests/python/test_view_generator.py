"""Acceptance tests for ViewGenerator — multi-view text framing.

Each framing strategy produces a reworded version of a memory's text,
creating an alternative retrieval surface for the same underlying fact.
"""

import pytest

from tardigrade_hooks.constants import (
    DEFAULT_VIEW_FRAMINGS,
    VIEW_FRAMING_PARAPHRASE,
    VIEW_FRAMING_QUESTION,
    VIEW_FRAMING_SUMMARY,
)
from tardigrade_hooks.view_generator import (
    FramingStrategy,
    ParaphraseFraming,
    QuestionFraming,
    SummaryFraming,
    ViewGenerator,
)

SAMPLE_TEXT = "Sonia translated a pharmaceutical patent from German to English for a Berlin-based biotech startup in March 2024."


class TestViewGeneratorContract:
    """Top-level generator produces the right number and type of views."""

    def test_generate_returns_one_view_per_framing(self):
        gen = ViewGenerator()
        views = gen.generate(SAMPLE_TEXT)
        assert len(views) == len(DEFAULT_VIEW_FRAMINGS)

    def test_all_views_are_nonempty_strings(self):
        gen = ViewGenerator()
        views = gen.generate(SAMPLE_TEXT)
        for v in views:
            assert isinstance(v, str)
            assert len(v.strip()) > 0

    def test_empty_input_returns_empty_list(self):
        gen = ViewGenerator()
        assert gen.generate("") == []

    def test_none_input_returns_empty_list(self):
        gen = ViewGenerator()
        assert gen.generate(None) == []

    def test_whitespace_only_returns_empty_list(self):
        gen = ViewGenerator()
        assert gen.generate("   \n\t  ") == []

    def test_custom_framings_subset(self):
        gen = ViewGenerator(framings=(VIEW_FRAMING_SUMMARY,))
        views = gen.generate(SAMPLE_TEXT)
        assert len(views) == 1

    def test_views_differ_from_each_other(self):
        gen = ViewGenerator()
        views = gen.generate(SAMPLE_TEXT)
        assert len(set(views)) == len(views), "Each view should be distinct"


class TestSummaryFraming:
    """Summary framing should condense the input."""

    def test_output_shorter_than_input(self):
        framing = SummaryFraming()
        result = framing.reframe(SAMPLE_TEXT)
        assert len(result) < len(SAMPLE_TEXT)

    def test_output_is_nonempty(self):
        framing = SummaryFraming()
        result = framing.reframe(SAMPLE_TEXT)
        assert len(result.strip()) > 0

    def test_preserves_key_entity(self):
        framing = SummaryFraming()
        result = framing.reframe(SAMPLE_TEXT)
        assert "sonia" in result.lower() or "translat" in result.lower()


class TestQuestionFraming:
    """Question framing should produce a question the memory could answer."""

    def test_output_contains_question_mark(self):
        framing = QuestionFraming()
        result = framing.reframe(SAMPLE_TEXT)
        assert "?" in result

    def test_output_is_nonempty(self):
        framing = QuestionFraming()
        result = framing.reframe(SAMPLE_TEXT)
        assert len(result.strip()) > 0


class TestParaphraseFraming:
    """Paraphrase framing should reword with limited word overlap."""

    def test_output_differs_from_input(self):
        framing = ParaphraseFraming()
        result = framing.reframe(SAMPLE_TEXT)
        assert result.strip() != SAMPLE_TEXT.strip()

    def test_content_word_overlap_below_threshold(self):
        """Overlap measured on content words only (excluding stop words,
        proper nouns, and numbers) since those carry retrieval signal."""
        framing = ParaphraseFraming()
        result = framing.reframe(SAMPLE_TEXT)
        stop = {"a", "an", "the", "is", "was", "in", "on", "to", "for",
                "of", "by", "from", "with", "and", "or", "but"}

        def content_words(text):
            return {w for w in text.lower().split()
                    if w not in stop and not w[0].isdigit() and len(w) > 2}

        original = content_words(SAMPLE_TEXT)
        reworded = content_words(result)
        if not original:
            return
        overlap = len(original & reworded) / len(original)
        assert overlap <= 0.60, f"Content-word overlap {overlap:.0%} exceeds 60%"

    def test_output_is_nonempty(self):
        framing = ParaphraseFraming()
        result = framing.reframe(SAMPLE_TEXT)
        assert len(result.strip()) > 0


class TestFramingStrategyABC:
    """FramingStrategy is a proper ABC."""

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            FramingStrategy()

    def test_concrete_implementations_are_substitutable(self):
        strategies = [SummaryFraming(), QuestionFraming(), ParaphraseFraming()]
        for s in strategies:
            assert isinstance(s, FramingStrategy)
            result = s.reframe(SAMPLE_TEXT)
            assert isinstance(result, str)
            assert len(result) > 0


class TestFramingRegistry:
    """Framings are discoverable by name."""

    def test_default_framings_all_resolve(self):
        gen = ViewGenerator()
        for name in DEFAULT_VIEW_FRAMINGS:
            assert name in gen._framings, f"Framing '{name}' not registered"
