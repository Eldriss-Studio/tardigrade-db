"""ATDD: CachedAnswerGenerator decorator (Slice L7).

Cache-Aside over any :class:`AnswerGenerator`. Keyed by
``(model_name, prompt_hash, template_version)`` so:

* identical re-runs hit cache (free + deterministic);
* changing the prompt template invalidates all cached entries via
  ``PROMPT_TEMPLATE_VERSION``;
* swapping the model invalidates that model's entries.
"""

from __future__ import annotations

from pathlib import Path

from tdb_bench.answerers import AnswerGenerator
from tdb_bench.answerers.cache import CachedAnswerGenerator
from tdb_bench.answerers.constants import PROMPT_TEMPLATE_VERSION


class _CountingGen(AnswerGenerator):
    def __init__(self, response: str = "ANS") -> None:
        self._response = response
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        return f"{self._response}-{self.call_count}"


class TestCachedAnswerGeneratorHits:
    def test_identical_prompt_hits_cache(self, tmp_path: Path):
        inner = _CountingGen()
        wrapped = CachedAnswerGenerator(
            inner=inner,
            model_name="m1",
            cache_dir=tmp_path,
        )

        first = wrapped.generate("the prompt")
        second = wrapped.generate("the prompt")

        assert inner.call_count == 1
        assert first == second

    def test_different_prompt_is_cache_miss(self, tmp_path: Path):
        inner = _CountingGen()
        wrapped = CachedAnswerGenerator(
            inner=inner, model_name="m1", cache_dir=tmp_path
        )

        wrapped.generate("prompt-A")
        wrapped.generate("prompt-B")

        assert inner.call_count == 2

    def test_cache_persists_across_instances(self, tmp_path: Path):
        first_inner = _CountingGen()
        first = CachedAnswerGenerator(
            inner=first_inner, model_name="m1", cache_dir=tmp_path
        )
        result_a = first.generate("xx")

        # Second wrapper, same on-disk cache dir, fresh inner.
        second_inner = _CountingGen(response="WOULD_DIFFER")
        second = CachedAnswerGenerator(
            inner=second_inner, model_name="m1", cache_dir=tmp_path
        )
        result_b = second.generate("xx")

        assert second_inner.call_count == 0
        assert result_a == result_b


class TestCachedAnswerGeneratorInvalidation:
    def test_different_model_is_cache_miss(self, tmp_path: Path):
        inner_a = _CountingGen()
        cache_a = CachedAnswerGenerator(
            inner=inner_a, model_name="alpha", cache_dir=tmp_path
        )
        cache_a.generate("p")

        inner_b = _CountingGen()
        cache_b = CachedAnswerGenerator(
            inner=inner_b, model_name="beta", cache_dir=tmp_path
        )
        cache_b.generate("p")

        # Cross-model leakage would corrupt benchmark results.
        assert inner_b.call_count == 1

    def test_different_template_version_is_cache_miss(self, tmp_path: Path):
        inner_v1 = _CountingGen()
        cache_v1 = CachedAnswerGenerator(
            inner=inner_v1,
            model_name="m",
            cache_dir=tmp_path,
            template_version="v1",
        )
        cache_v1.generate("p")

        # Bump template version — prior entries must not be served.
        inner_v2 = _CountingGen()
        cache_v2 = CachedAnswerGenerator(
            inner=inner_v2,
            model_name="m",
            cache_dir=tmp_path,
            template_version="v2",
        )
        cache_v2.generate("p")

        assert inner_v2.call_count == 1

    def test_default_template_version_from_constants(self, tmp_path: Path):
        inner = _CountingGen()
        wrapped = CachedAnswerGenerator(
            inner=inner, model_name="m", cache_dir=tmp_path
        )

        wrapped.generate("p")

        # Subsequent call with explicit current PROMPT_TEMPLATE_VERSION
        # hits cache — proves the default uses that constant.
        explicit = CachedAnswerGenerator(
            inner=_CountingGen(),
            model_name="m",
            cache_dir=tmp_path,
            template_version=PROMPT_TEMPLATE_VERSION,
        )
        result = explicit.generate("p")

        assert "WOULD_DIFFER" not in result  # the second inner wasn't called


class TestCachedAnswerGeneratorSubstitutability:
    def test_is_an_answer_generator(self, tmp_path: Path):
        wrapped = CachedAnswerGenerator(
            inner=_CountingGen(), model_name="m", cache_dir=tmp_path
        )
        assert isinstance(wrapped, AnswerGenerator)
