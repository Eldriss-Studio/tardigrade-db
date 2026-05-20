"""Acceptance tests for `tardigrade_db.find_chunk_boundary`.

The Python `BoundaryStrategy` ABC in `tardigrade_hooks.chunker` now
delegates to this Rust implementation. ATs pin the public boundary
contract — whitespace returns the index of the last space, sentence
returns the index *after* the terminal punctuation (left chunk keeps
the punctuation), paragraph prefers `\\n\\n` then sentence then
whitespace fallback.
"""

from __future__ import annotations

import pytest

from tardigrade_db import find_chunk_boundary


def test_empty_text_returns_zero():
    assert find_chunk_boundary("", 100, "whitespace") == 0


def test_max_pos_beyond_text_length_clamps():
    text = "abc"
    assert find_chunk_boundary(text, 100, "whitespace") == len(text)


def test_whitespace_returns_last_space():
    text = "alpha beta gamma"
    expected = text.rfind(" ")
    assert find_chunk_boundary(text, len(text), "whitespace") == expected


def test_sentence_returns_position_after_terminal_punctuation():
    text = "first sentence. second"
    idx = find_chunk_boundary(text, len(text), "sentence")
    assert idx == text.rfind(".") + 1
    assert text[:idx] == "first sentence."


def test_sentence_handles_cjk_terminator():
    text = "こんにちは。次の文"
    idx = find_chunk_boundary(text, len(text.encode("utf-8")), "sentence")
    # 5 hiragana × 3 bytes = 15, plus 3 bytes for '。' = 18.
    assert idx == 18
    assert text.encode("utf-8")[:idx].decode("utf-8") == "こんにちは。"


def test_paragraph_prefers_double_newline():
    text = "first turn\n\nsecond turn"
    idx = find_chunk_boundary(text, len(text), "paragraph")
    assert idx == text.find("\n\n")


def test_paragraph_falls_back_through_sentence_to_whitespace():
    text = "no paragraph here just a sentence. tail"
    idx = find_chunk_boundary(text, len(text), "paragraph")
    assert idx == text.rfind(".") + 1


def test_unknown_strategy_raises_value_error():
    with pytest.raises((RuntimeError, ValueError)):
        find_chunk_boundary("text", 4, "banana")


def test_no_boundary_within_range_returns_max_pos():
    # All-letters string, no spaces or punctuation.
    text = "abcdefghij"
    assert find_chunk_boundary(text, 5, "whitespace") == 5
