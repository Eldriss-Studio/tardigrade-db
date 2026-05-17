"""Document chunker for file ingestion into TardigradeDB.

Builder pattern for configuration, Strategy pattern for boundary detection.
The chunker splits text into token-bounded chunks suitable for per-chunk
KV capture via forward pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constants import (
    BOUNDARY_LOOKBACK_RATIO,
    CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
)

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


class ParagraphBoundaryStrategy(BoundaryStrategy):
    """Split preferring paragraph / turn boundaries (``\\n\\n``), then
    sentence end, then whitespace.

    Designed for conversational transcripts where each turn is
    delimited by a blank line — e.g. LongMemEval haystack sessions.
    Splitting at turn boundaries preserves speaker-message integrity
    and prevents the mid-message fragment chunks diagnosed in the
    2026-05-14 audit (`docs/experiments/2026-05-14-bench-audit.md`).

    Decorator-style fallback: paragraph → sentence → whitespace. Each
    fallback only runs if the preceding strategy finds no usable
    boundary in the lookback window.
    """

    def find_split(self, text: str, max_pos: int) -> int:
        # Prefer paragraph boundary (\n\n or \r\n\r\n).
        idx = text.rfind("\n\n", 0, max_pos)
        if idx > 0:
            return idx
        idx = text.rfind("\r\n\r\n", 0, max_pos)
        if idx > 0:
            return idx
        # Fall back to sentence boundary.
        sentence = SentenceBoundaryStrategy().find_split(text, max_pos)
        if sentence > 0 and sentence < max_pos:
            return sentence
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
        """Slice tokens with boundary-aware ends.

        Before 2026-05-14 this method advanced by a fixed token
        stride and ignored ``self._boundary`` entirely, producing
        mid-word/mid-sentence fragment chunks. Those fragments
        carried degenerate hidden states that dominated retrieval
        (LongMemEval diagnostic: cell 225 appeared in 99.2% of top-10s
        across all 500 queries). Now: every non-final chunk's
        char-space end is pulled back to the most-recent boundary
        within a ``BOUNDARY_LOOKBACK_RATIO``-sized window; only
        runaway content with no usable boundary in that window
        receives a hard cut.

        Tokens consumed per chunk are derived from the trimmed text
        (re-encoded) so the stride reflects actual chunk size rather
        than the hard ``max_tokens`` count.
        """
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        search_from = 0

        while start < len(all_tokens):
            hard_end = min(start + self._max_tokens, len(all_tokens))
            chunk_tokens = all_tokens[start:hard_end]
            decoded = self._tokenizer.decode(chunk_tokens).strip()
            if not decoded:
                # Defensive: empty decode (rare with proper tokenizers)
                break

            # Locate in original text — handles both string and int tokens
            pos = text.find(decoded, search_from)
            if pos < 0:
                first_word = decoded.split()[0] if decoded else ""
                pos = text.find(first_word, search_from) if first_word else search_from
                if pos < 0:
                    pos = search_from
            start_char = pos
            end_char = min(start_char + len(decoded), len(text))

            # chunker boundary work — boundary-aware START trim for non-first
            # chunks. The overlap stride lands chunk N+1's start
            # ``overlap_tokens`` tokens back into chunk N, which for
            # any sub-word tokenizer almost always falls mid-word.
            # retrieval diagnostic 2026-05-14 showed those mid-word
            # starts (especially mid-number, e.g. "0000" inside
            # "$10,000") produce anomalous hidden states that
            # dominate retrieval as hub cells. Snap start_char
            # forward to the next whitespace within a lookahead
            # window so chunks begin on whole words.
            is_first = start == 0
            starts_cleanly = (
                start_char == 0
                or text[start_char - 1].isspace()
            )
            if not is_first and not starts_cleanly:
                lookahead_chars = max(
                    1,
                    int((end_char - start_char) * BOUNDARY_LOOKBACK_RATIO),
                )
                search_limit = min(
                    start_char + lookahead_chars,
                    end_char,
                )
                for probe in range(start_char, search_limit):
                    if text[probe].isspace():
                        start_char = probe + 1
                        break

            # Boundary-aware trim — non-final chunks only, and only
            # when the chunk would otherwise end mid-word. If the next
            # original character is already whitespace (or we're at
            # end-of-text), the decoded chunk lands on a natural word
            # boundary; trimming would just discard a usable word.
            is_last = hard_end >= len(all_tokens)
            ends_cleanly = (
                end_char >= len(text)
                or text[end_char].isspace()
            )
            if not is_last and not ends_cleanly:
                window_chars = max(
                    1,
                    int((end_char - start_char) * BOUNDARY_LOOKBACK_RATIO),
                )
                min_end_char = max(start_char + 1, end_char - window_chars)
                boundary_pos = self._boundary.find_split(text, max_pos=end_char)
                if min_end_char <= boundary_pos < end_char:
                    end_char = boundary_pos

            chunk_text = text[start_char:end_char].strip()
            if not chunk_text:
                # Trimmed to nothing — shouldn't happen given min_end_char
                # guard, but stop defensively.
                break

            # Recover the token count actually consumed by the trim.
            consumed_tokens = self._tokenizer.encode(chunk_text)
            consumed_count = max(1, len(consumed_tokens))

            chunks.append(Chunk(
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                chunk_index=idx,
                token_count=consumed_count,
            ))

            if is_last:
                break

            # Advance by tokens actually consumed minus the overlap.
            stride = max(1, consumed_count - self._overlap_tokens)
            start += stride
            search_from = start_char + 1
            idx += 1

        return chunks
