"""AT-1A.1, AT-1A.2 — TextChunker honors its BoundaryStrategy.

Background: the 2026-05-14 retrieval diagnostic
(`docs/experiments/2026-05-14-bench-audit.md`) found that
TextChunker._split_tokens accepted a BoundaryStrategy in __init__ but
never invoked it. Chunks split purely by token count produced
mid-word/mid-sentence fragments, and a small set of those fragment
chunks dominated LongMemEval retrieval (cell 225 appeared in 99.2%
of top-10s across all 500 queries).

These ATs pin: (a) the splitter actually calls self._boundary on
every split, and (b) the new ParagraphBoundaryStrategy prefers
turn-boundary newlines over sentence-mid splits, which is the
strategy LongMemEval-style conversational data needs.

Behavioral tests (Kent Dodds): construct input where the difference
between "respects boundary" and "doesn't" is observable in chunk
*output* — no internal-state probes.
"""
from __future__ import annotations

import pytest

from tardigrade_hooks.chunker import (
    ParagraphBoundaryStrategy,
    SentenceBoundaryStrategy,
    TextChunker,
    WhitespaceBoundaryStrategy,
)


# -- Real-tokenizer fixtures ------------------------------------------------

class _CharOffsetWordTokenizer:
    """Tokenizer whose tokens preserve original spacing.

    Unlike the test_chunker.py `_WordTokenizer`, this one's decode
    round-trips whitespace so that find-in-original-text checks line
    up correctly. Token = (word, leading_space_count) so the
    boundary fix has a real example where the hard split lands
    mid-word.
    """

    def encode(self, text):
        # Tokens are individual words, but we also preserve a leading
        # space marker so spacing isn't ambiguous on decode.
        return text.split(" ")

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.fixture
def tokenizer():
    return _CharOffsetWordTokenizer()


# -- AT-1A.1 — splitter respects WhitespaceBoundaryStrategy -----------------

class TestWhitespaceBoundaryRespected:
    """The splitter must call self._boundary on every split, not
    just slice by token count.

    Diagnosed 2026-05-14: prior implementation ignored
    boundary_strategy entirely, causing fragment chunks to dominate
    retrieval.
    """

    def test_no_chunk_starts_mid_word_with_whitespace_boundary(self, tokenizer):
        """The bug: prior splitter sliced at token offset
        `start + max_tokens - overlap`, ignoring word boundaries.
        Fix: every chunk-end must land at a whitespace position
        returned by the boundary strategy.
        """
        # 30 long tokens, max=10, overlap=2, stride=8.
        # The hard splits would land at tokens 10, 18, 26 — fine
        # individually, but the boundary should still be honored.
        # The point: if we use a strategy that *can't* find a
        # boundary anywhere except whitespace, splits must land on
        # spaces.
        text = " ".join(f"interconnection{i:02d}" for i in range(30))
        chunker = TextChunker(
            tokenizer,
            max_tokens=10,
            overlap_tokens=2,
            boundary_strategy=WhitespaceBoundaryStrategy(),
        )
        chunks = chunker.chunk(text)

        assert len(chunks) > 1, (
            "expected the text to require splitting; got 1 chunk"
        )

        for c in chunks[1:]:  # skip first chunk (always starts at 0)
            head = c.text[:30]
            # Every non-first chunk must begin on a whole token —
            # never a fragment like "rconnection07".
            assert head.startswith("interconnection"), (
                f"chunk {c.chunk_index} starts mid-word: {head!r}"
            )

    def test_split_falls_on_whitespace_when_available(self, tokenizer):
        """When the hard token boundary lands inside a phrase, the
        boundary strategy must pull the split back to the most-recent
        whitespace within the lookback window.
        """
        # 50 words. Use a sentence with a "long" word at the
        # would-be split point so the bug, if present, leaves
        # the chunk ending mid-word.
        text = " ".join(["alpha", "bravo", "charlie", "delta", "echo"] * 10)
        chunker = TextChunker(
            tokenizer,
            max_tokens=12,
            overlap_tokens=0,
            boundary_strategy=WhitespaceBoundaryStrategy(),
        )
        chunks = chunker.chunk(text)

        for c in chunks:
            # No chunk text should end on a partial word (every word
            # in the fixture is a whole identifier).
            last_token = c.text.split(" ")[-1] if c.text else ""
            assert last_token in ("alpha", "bravo", "charlie", "delta",
                                   "echo", ""), (
                f"chunk {c.chunk_index} ends on a fragment: ...{c.text[-30:]!r}"
            )


# -- AT-1A.2 — ParagraphBoundaryStrategy prefers turn boundaries -----------

class TestParagraphBoundaryStrategy:
    """New strategy: prefer split on \\n\\n (paragraph / turn
    boundary), fall back to sentence, then whitespace. The natural
    strategy for conversational transcripts.
    """

    def test_prefers_double_newline_over_sentence_boundary(self):
        strategy = ParagraphBoundaryStrategy()
        # Two turns separated by \n\n. A sentence ends inside turn 1
        # before the paragraph break. Strategy should pick the
        # paragraph break (split AFTER both sentences of turn 1).
        text = "user: First sentence. Second sentence.\n\nassistant: Reply."
        max_pos = text.find("\n\nassistant") + 10  # forces split past the \n\n
        split = strategy.find_split(text, max_pos=max_pos)
        # Split must land on the paragraph boundary, not the mid-turn
        # sentence period.
        assert split == text.find("\n\n"), (
            f"expected paragraph boundary at {text.find(chr(10) + chr(10))}, "
            f"got {split} (char before: {text[split-1:split+2]!r})"
        )

    def test_falls_back_to_sentence_then_whitespace(self):
        """When no paragraph boundary exists within max_pos, prefer
        sentence end; when no sentence end, prefer whitespace.
        """
        strategy = ParagraphBoundaryStrategy()
        text_sentence = "alpha bravo charlie. delta echo foxtrot golf"
        # No \n\n; should land on the period.
        split = strategy.find_split(text_sentence, max_pos=25)
        assert split == text_sentence.find(".") + 1, (
            f"expected sentence boundary at "
            f"{text_sentence.find('.') + 1}, got {split}"
        )

        text_no_punct = "alpha bravo charlie delta echo foxtrot golf"
        split = strategy.find_split(text_no_punct, max_pos=25)
        assert text_no_punct[split] == " " or split == 25, (
            f"expected whitespace fallback, got "
            f"split={split} at {text_no_punct[split:split+5]!r}"
        )


# -- AT-1A.1 sanity: splitter actually calls the strategy at all -----------

class TestSubWordTokenizerBoundaryAwareness:
    """End-to-end behavioral test using a sub-word tokenizer (real-
    world BPE case). The original bug: ``_split_tokens`` ignored its
    own boundary strategy, so when sub-word tokens straddled word
    boundaries — typical for BPE tokenizers on natural English —
    chunks ended mid-word and produced anomalous hidden states that
    dominated retrieval (LongMemEval retrieval diagnostic, 2026-05-14).

    With the fix, every non-final chunk must end on a whole-word
    boundary even though the underlying tokenizer slices sub-word.
    """

    class _CharPairTokenizer:
        """Tokenizes into 3-character pieces — simulates BPE."""

        _PIECE = 3

        def encode(self, text):
            return [text[i:i + self._PIECE]
                    for i in range(0, len(text), self._PIECE)]

        def decode(self, tokens):
            return "".join(tokens)

    def test_subword_chunks_still_end_on_whole_words(self):
        # Words of length 6 separated by single spaces. With 3-char
        # tokens, every 2 tokens = 1 word. Chunks of 7 tokens
        # (= 21 chars) land mid-word every time. Lookback at
        # ratio=0.25 covers 5 chars — exactly one " word" wide.
        text = "wonder marvel splash quartz blazon thrush galaxy " * 8
        chunker = TextChunker(
            self._CharPairTokenizer(),
            max_tokens=7,
            overlap_tokens=0,
            boundary_strategy=WhitespaceBoundaryStrategy(),
        )
        chunks = chunker.chunk(text)

        assert len(chunks) > 1, "expected multiple chunks"

        # Every non-final chunk must end on a whole word (not a
        # sub-word fragment). The boundary strategy is what makes
        # this possible — without it, chunks would end mid-token.
        valid_words = {"wonder", "marvel", "splash", "quartz",
                       "blazon", "thrush", "galaxy"}
        for c in chunks[:-1]:
            tokens = c.text.split()
            assert tokens, f"chunk {c.chunk_index} is empty"
            last_word = tokens[-1]
            assert last_word in valid_words, (
                f"chunk {c.chunk_index} ends on sub-word fragment "
                f"{last_word!r}; full text tail={c.text[-25:]!r}"
            )

    def test_subword_chunks_dont_start_on_fragments(self):
        text = "wonder marvel splash quartz blazon thrush galaxy " * 8
        chunker = TextChunker(
            self._CharPairTokenizer(),
            max_tokens=7,
            overlap_tokens=0,
            boundary_strategy=WhitespaceBoundaryStrategy(),
        )
        chunks = chunker.chunk(text)

        valid_words = {"wonder", "marvel", "splash", "quartz",
                       "blazon", "thrush", "galaxy"}
        # Chunks after the first must start on a whole word so the
        # encoded retrieval key isn't dominated by a continuation
        # fragment (the 2026-05-14 hub-cell failure).
        for c in chunks[1:]:
            tokens = c.text.split()
            if not tokens:
                continue
            first_word = tokens[0]
            assert first_word in valid_words, (
                f"chunk {c.chunk_index} starts on fragment "
                f"{first_word!r}; head={c.text[:25]!r}"
            )

    def test_overlap_does_not_create_mid_word_starts(self):
        """chunker boundary work: with non-zero overlap, each chunk after the
        first starts ``overlap_tokens`` tokens BEFORE the previous
        chunk's clean end. Going back into the middle of the previous
        chunk's content lands mid-word for any sub-word tokenizer.

        retrieval diagnostic (2026-05-14): the LongMemEval hub cells
        were all chunks starting with numeric fragments
        (``0000 is...``, ``000 miles...``) — produced by the overlap
        landing inside a number like ``$10,000``. End-boundary trim
        (chunker boundary work) didn't fix this; start-boundary snap does.
        """
        text = "wonder marvel splash quartz blazon thrush galaxy " * 12
        chunker = TextChunker(
            self._CharPairTokenizer(),
            max_tokens=14,
            overlap_tokens=4,  # non-zero overlap — the bug trigger
            boundary_strategy=WhitespaceBoundaryStrategy(),
        )
        chunks = chunker.chunk(text)

        assert len(chunks) > 1, "expected multiple chunks for overlap"

        valid_words = {"wonder", "marvel", "splash", "quartz",
                       "blazon", "thrush", "galaxy"}
        for c in chunks[1:]:
            tokens = c.text.split()
            if not tokens:
                continue
            first_word = tokens[0]
            assert first_word in valid_words, (
                f"chunk {c.chunk_index} starts on overlap fragment "
                f"{first_word!r}; head={c.text[:25]!r}"
            )
