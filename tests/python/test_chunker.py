"""Acceptance tests for TextChunker — document chunking for file ingestion."""

import pytest

from tardigrade_hooks.constants import (
    CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
)
from tardigrade_hooks.chunker import Chunk, TextChunker


# -- Stub tokenizer (word-level, 1 word ≈ 1 token) --------------------------

class _WordTokenizer:
    """Minimal tokenizer that treats whitespace-separated words as tokens."""

    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.fixture
def tokenizer():
    return _WordTokenizer()


# -- Basic contract ----------------------------------------------------------

class TestChunkerContract:
    def test_single_short_text_returns_one_chunk(self, tokenizer):
        chunker = TextChunker(tokenizer, max_tokens=100)
        chunks = chunker.chunk("Hello world this is a test.")
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_empty_input_returns_empty(self, tokenizer):
        chunker = TextChunker(tokenizer, max_tokens=100)
        assert chunker.chunk("") == []

    def test_none_input_returns_empty(self, tokenizer):
        chunker = TextChunker(tokenizer, max_tokens=100)
        assert chunker.chunk(None) == []

    def test_whitespace_only_returns_empty(self, tokenizer):
        chunker = TextChunker(tokenizer, max_tokens=100)
        assert chunker.chunk("   \n\t  ") == []

    def test_chunk_dataclass_fields(self, tokenizer):
        chunker = TextChunker(tokenizer, max_tokens=100)
        chunks = chunker.chunk("Hello world.")
        c = chunks[0]
        assert isinstance(c, Chunk)
        assert isinstance(c.text, str)
        assert isinstance(c.start_char, int)
        assert isinstance(c.end_char, int)
        assert isinstance(c.chunk_index, int)
        assert isinstance(c.token_count, int)


# -- Chunking behavior -------------------------------------------------------

class TestChunkSizing:
    def test_no_chunk_exceeds_max_tokens(self, tokenizer):
        text = " ".join(f"word{i}" for i in range(200))
        chunker = TextChunker(tokenizer, max_tokens=50, overlap_tokens=0)
        chunks = chunker.chunk(text)
        for c in chunks:
            assert c.token_count <= 50, (
                f"Chunk {c.chunk_index} has {c.token_count} tokens > 50"
            )

    def test_produces_expected_chunk_count(self, tokenizer):
        text = " ".join(f"word{i}" for i in range(100))
        chunker = TextChunker(tokenizer, max_tokens=25, overlap_tokens=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 4  # 100 / 25 = 4

    def test_overlap_creates_shared_tokens(self, tokenizer):
        text = " ".join(f"word{i}" for i in range(100))
        chunker = TextChunker(tokenizer, max_tokens=30, overlap_tokens=5)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        # Check first two chunks share some text
        words_0 = set(chunks[0].text.split())
        words_1 = set(chunks[1].text.split())
        shared = words_0 & words_1
        assert len(shared) >= 3, "Overlapping chunks should share words"

    def test_zero_overlap_non_overlapping(self, tokenizer):
        text = " ".join(f"word{i}" for i in range(100))
        chunker = TextChunker(tokenizer, max_tokens=50, overlap_tokens=0)
        chunks = chunker.chunk(text)
        all_words = []
        for c in chunks:
            all_words.extend(c.text.split())
        assert len(all_words) == len(set(all_words)), "No overlap expected"


# -- Character offsets -------------------------------------------------------

class TestCharOffsets:
    def test_offsets_cover_full_text(self, tokenizer):
        text = "Alpha bravo charlie delta echo foxtrot golf hotel."
        chunker = TextChunker(tokenizer, max_tokens=4, overlap_tokens=0)
        chunks = chunker.chunk(text)
        reconstructed = ""
        for c in chunks:
            reconstructed += text[c.start_char:c.end_char]
        # All original words should appear in reconstruction
        for word in text.split():
            assert word in reconstructed or word.strip(".") in reconstructed

    def test_chunk_indices_sequential(self, tokenizer):
        text = " ".join(f"word{i}" for i in range(100))
        chunker = TextChunker(tokenizer, max_tokens=20, overlap_tokens=0)
        chunks = chunker.chunk(text)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


# -- Unicode -----------------------------------------------------------------

class TestUnicode:
    def test_cjk_text_handled(self, tokenizer):
        text = "你好 世界 这是 一个 测试 句子 用于 验证"
        chunker = TextChunker(tokenizer, max_tokens=3, overlap_tokens=0)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.token_count <= 3

    def test_emoji_text_handled(self, tokenizer):
        text = "hello 🌍 world 🎉 test 🚀 data 📊 end"
        chunker = TextChunker(tokenizer, max_tokens=3, overlap_tokens=0)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
