"""Document chunker for file ingestion into TardigradeDB.

Builder pattern for configuration, Strategy pattern for boundary detection.
The chunker splits text into token-bounded chunks suitable for per-chunk
KV capture via forward pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constants import CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_TOKENS, MIN_CHUNK_TOKENS

if TYPE_CHECKING:
    pass


@dataclass
class Chunk:
    """One chunk of a document, ready for KV capture."""

    text: str
    start_char: int
    end_char: int
    chunk_index: int
    token_count: int


# ---------------------------------------------------------------------------
# Boundary strategies
# ---------------------------------------------------------------------------


class BoundaryStrategy(ABC):
    """Finds a split point that respects natural text boundaries."""

    @abstractmethod
    def find_split(self, text: str, max_pos: int) -> int:
        """Return a character position ≤ max_pos to split at.

        Should prefer sentence or whitespace boundaries.  If no good
        boundary exists, return max_pos (hard break).
        """
        ...


class WhitespaceBoundaryStrategy(BoundaryStrategy):
    """Split at the last whitespace before max_pos."""

    def find_split(self, text: str, max_pos: int) -> int:
        idx = text.rfind(" ", 0, max_pos)
        if idx > 0:
            return idx
        return max_pos


class SentenceBoundaryStrategy(BoundaryStrategy):
    """Split at the last sentence-ending punctuation before max_pos,
    falling back to whitespace."""

    _ENDINGS = (".", "!", "?", "。", "！", "？")

    def find_split(self, text: str, max_pos: int) -> int:
        best = -1
        for ch in self._ENDINGS:
            idx = text.rfind(ch, 0, max_pos)
            if idx > best:
                best = idx
        if best > 0:
            return best + 1  # include the punctuation
        return WhitespaceBoundaryStrategy().find_split(text, max_pos)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class TextChunker:
    """Splits text into token-bounded chunks with optional overlap.

    Args:
        tokenizer: Any object with ``.encode(text) -> list`` and
            ``.decode(tokens) -> str`` methods.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of trailing tokens from previous chunk
            to prepend to the next chunk.
        boundary_strategy: How to find split points.  Defaults to
            ``WhitespaceBoundaryStrategy``.
    """

    def __init__(
        self,
        tokenizer,
        max_tokens: int = DEFAULT_CHUNK_TOKENS,
        overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
        boundary_strategy: BoundaryStrategy | None = None,
    ):
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._boundary = boundary_strategy or WhitespaceBoundaryStrategy()

    def chunk(self, text: str | None) -> list[Chunk]:
        """Split *text* into chunks respecting token limits."""
        if not text or not text.strip():
            return []

        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self._max_tokens:
            return [Chunk(
                text=text.strip(),
                start_char=0,
                end_char=len(text),
                chunk_index=0,
                token_count=len(tokens),
            )]

        return self._split_tokens(text, tokens)

    def _split_tokens(self, text: str, all_tokens: list) -> list[Chunk]:
        chunks: list[Chunk] = []
        stride = max(1, self._max_tokens - self._overlap_tokens)
        start = 0
        idx = 0
        search_from = 0

        while start < len(all_tokens):
            end = min(start + self._max_tokens, len(all_tokens))
            chunk_tokens = all_tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens)

            # Find chunk text in original — handles both str and int tokens
            pos = text.find(chunk_text.strip(), search_from)
            if pos < 0:
                # Decoded text may have extra whitespace; fuzzy match
                first_word = chunk_text.strip().split()[0] if chunk_text.strip() else ""
                pos = text.find(first_word, search_from) if first_word else search_from
                if pos < 0:
                    pos = search_from
            start_char = pos
            end_char = min(start_char + len(chunk_text.strip()), len(text))
            search_from = start_char + 1

            chunks.append(Chunk(
                text=chunk_text.strip(),
                start_char=start_char,
                end_char=end_char,
                chunk_index=idx,
                token_count=len(chunk_tokens),
            ))

            if end >= len(all_tokens):
                break
            start += stride
            idx += 1

        return chunks
