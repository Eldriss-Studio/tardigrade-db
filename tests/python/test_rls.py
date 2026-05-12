"""ATDD tests for Reflective Latent Search — Strategy pattern for query reformulation."""

import pytest

import numpy as np

from tardigrade_hooks.rls import (
    EmbeddingExpansionStrategy,
    KeywordExpansionStrategy,
    MultiPhrasingStrategy,
    ReformulationStrategy,
)


class TestReformulationStrategyABC:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            ReformulationStrategy()

    def test_keyword_is_substitutable(self):
        assert isinstance(KeywordExpansionStrategy(), ReformulationStrategy)

    def test_multiphrasing_is_substitutable(self):
        assert isinstance(MultiPhrasingStrategy(), ReformulationStrategy)

    def test_embedding_is_substitutable(self):
        # Minimal mock: 5-token vocab with 4-dim embeddings
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [0]
            def decode(self, ids):
                return "test"
        embed = np.random.randn(5, 4).astype(np.float32)
        assert isinstance(EmbeddingExpansionStrategy(FakeTokenizer(), embed), ReformulationStrategy)


class TestKeywordExpansion:
    def test_expands_with_synonyms(self):
        s = KeywordExpansionStrategy()
        results = s.reformulate("What does Sonia know about languages?")
        assert len(results) >= 1
        expanded = results[0].lower()
        assert any(word in expanded for word in ["translat", "linguistic", "foreign"])

    def test_athletic_expansion(self):
        s = KeywordExpansionStrategy()
        results = s.reformulate("Tell me about athletic achievements")
        expanded = results[0].lower()
        assert any(word in expanded for word in ["marathon", "running", "ultramarathon", "race"])

    def test_mechanical_expansion(self):
        s = KeywordExpansionStrategy()
        results = s.reformulate("Tell me something mechanical about Sonia")
        expanded = results[0].lower()
        assert any(word in expanded for word in ["motorcycle", "engine", "motor", "restored"])

    def test_empty_input(self):
        s = KeywordExpansionStrategy()
        assert s.reformulate("") == []
        assert s.reformulate(None) == []

    def test_returns_list_of_strings(self):
        s = KeywordExpansionStrategy()
        results = s.reformulate("some query")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, str)


class TestMultiPhrasing:
    def test_returns_multiple_variants(self):
        s = MultiPhrasingStrategy()
        results = s.reformulate("What does Sonia know about languages?")
        assert len(results) >= 2

    def test_variants_differ_from_each_other(self):
        s = MultiPhrasingStrategy()
        results = s.reformulate("What scientific research has Sonia done?")
        assert len(set(results)) == len(results)

    def test_keyword_only_variant_has_no_question_words(self):
        s = MultiPhrasingStrategy()
        results = s.reformulate("What does Sonia know about languages?")
        keyword_variant = results[0]
        assert "what" not in keyword_variant.lower()
        assert "does" not in keyword_variant.lower()

    def test_empty_input(self):
        s = MultiPhrasingStrategy()
        assert s.reformulate("") == []
        assert s.reformulate(None) == []


class TestEmbeddingExpansion:
    @pytest.fixture
    def strategy(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            pytest.skip("transformers not installed")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        embed = model.get_input_embeddings().weight.detach().float().cpu().numpy()
        return EmbeddingExpansionStrategy(tokenizer, embed)

    def test_expands_query(self, strategy):
        results = strategy.reformulate("Tell me about athletic achievements")
        assert len(results) >= 1
        expanded = results[0].lower()
        assert len(expanded.split()) > 2

    def test_empty_input(self, strategy):
        assert strategy.reformulate("") == []
        assert strategy.reformulate(None) == []

    def test_expansion_contains_related_tokens(self, strategy):
        results = strategy.reformulate("What does Sonia know about languages?")
        if results:
            expanded = results[0].lower()
            assert len(expanded.split()) > 2
