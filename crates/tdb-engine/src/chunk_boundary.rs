//! Text-chunk boundary search.
//!
//! `TextChunker` (Python side) tokenises a document, picks a byte
//! position to split at, then walks backward from that position to a
//! natural boundary — a whitespace, sentence end, or paragraph break.
//! That backward walk is pure byte scanning; the tokenizer's not
//! involved. Doing it in Rust keeps `str.rfind` loops off the hot
//! ingest path and gives the same semantics regardless of which
//! consumer is driving.

/// Sentence-terminating punctuation. Includes the four CJK fullwidth
/// variants so the same boundary search works on Japanese, Chinese, and
/// Korean text without a separate code path.
const SENTENCE_ENDINGS: &[&str] = &[".", "!", "?", "。", "！", "？"];

/// Strict paragraph delimiter (LF + LF).
const PARAGRAPH_LF: &str = "\n\n";

/// Strict paragraph delimiter when newlines are CRLF.
const PARAGRAPH_CRLF: &str = "\r\n\r\n";

/// Which natural boundary to prefer when splitting a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryStrategy {
    /// Last whitespace before `max_pos`.
    Whitespace,
    /// Last sentence terminator; falls back to whitespace if none found.
    Sentence,
    /// Last paragraph break (`\n\n` or `\r\n\r\n`); falls back to
    /// sentence, then whitespace.
    Paragraph,
}

impl BoundaryStrategy {
    /// Parse the string form callers pass through the `PyO3` layer.
    ///
    /// # Errors
    /// Returns [`UnknownBoundaryStrategy`] if `name` is not one of
    /// `"whitespace"`, `"sentence"`, or `"paragraph"`.
    pub fn parse(name: &str) -> Result<Self, UnknownBoundaryStrategy> {
        match name {
            "whitespace" => Ok(Self::Whitespace),
            "sentence" => Ok(Self::Sentence),
            "paragraph" => Ok(Self::Paragraph),
            other => Err(UnknownBoundaryStrategy { name: other.to_owned() }),
        }
    }
}

/// Returned when [`BoundaryStrategy::parse`] receives an unknown name.
#[derive(Debug, Clone)]
pub struct UnknownBoundaryStrategy {
    pub name: String,
}

impl std::fmt::Display for UnknownBoundaryStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown boundary strategy {:?}; expected one of \"whitespace\", \"sentence\", \"paragraph\"",
            self.name
        )
    }
}

impl std::error::Error for UnknownBoundaryStrategy {}

/// Find a byte position `<= max_pos` to split `text` at, preferring the
/// natural boundary indicated by `strategy`.
///
/// Semantics match `tardigrade_hooks/chunker.py::BoundaryStrategy.find_split`
/// — whitespace returns the index of the last space, sentence returns
/// the index *after* the punctuation (so the punctuation stays with the
/// left chunk), paragraph returns the index of the delimiter (so the
/// blank line goes to the right chunk).
///
/// `max_pos` is clamped to `text.len()`. An empty `text` always returns
/// `0`. When no boundary is found, the function returns `max_pos`
/// (clamped) so the caller can still make forward progress.
#[must_use]
pub fn find_chunk_boundary(text: &str, max_pos: usize, strategy: BoundaryStrategy) -> usize {
    let bound = max_pos.min(text.len());
    if bound == 0 {
        return 0;
    }
    match strategy {
        BoundaryStrategy::Whitespace => whitespace_split(text, bound),
        BoundaryStrategy::Sentence => sentence_split(text, bound),
        BoundaryStrategy::Paragraph => paragraph_split(text, bound),
    }
}

fn whitespace_split(text: &str, bound: usize) -> usize {
    match text[..bound].rfind(' ') {
        Some(idx) if idx > 0 => idx,
        _ => bound,
    }
}

fn sentence_split(text: &str, bound: usize) -> usize {
    let mut best: Option<usize> = None;
    let head = &text[..bound];
    for &delim in SENTENCE_ENDINGS {
        if let Some(idx) = head.rfind(delim) {
            let end = idx + delim.len();
            if end > best.unwrap_or(0) {
                best = Some(end);
            }
        }
    }
    match best {
        Some(end) if end > 0 => end,
        _ => whitespace_split(text, bound),
    }
}

fn paragraph_split(text: &str, bound: usize) -> usize {
    let head = &text[..bound];
    if let Some(idx) = head.rfind(PARAGRAPH_LF)
        && idx > 0
    {
        return idx;
    }
    if let Some(idx) = head.rfind(PARAGRAPH_CRLF)
        && idx > 0
    {
        return idx;
    }
    let sentence = sentence_split(text, bound);
    if sentence > 0 && sentence < bound {
        return sentence;
    }
    whitespace_split(text, bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_returns_zero() {
        assert_eq!(find_chunk_boundary("", 10, BoundaryStrategy::Whitespace), 0);
    }

    #[test]
    fn max_pos_beyond_text_length_clamps() {
        let text = "abc";
        assert_eq!(find_chunk_boundary(text, 100, BoundaryStrategy::Whitespace), text.len());
    }

    #[test]
    fn whitespace_returns_last_space_before_bound() {
        let text = "alpha beta gamma";
        // last ' ' in text[..16] is at index 10
        assert_eq!(find_chunk_boundary(text, 16, BoundaryStrategy::Whitespace), 10);
    }

    #[test]
    fn whitespace_falls_back_to_max_pos_when_no_space() {
        let text = "abcdefghij";
        assert_eq!(find_chunk_boundary(text, 5, BoundaryStrategy::Whitespace), 5);
    }

    #[test]
    fn sentence_returns_position_after_terminal_punctuation() {
        let text = "first sentence. second";
        // '.' at index 14, returned position is 15
        let idx = find_chunk_boundary(text, text.len(), BoundaryStrategy::Sentence);
        assert_eq!(idx, 15);
        assert_eq!(&text[..idx], "first sentence.");
    }

    #[test]
    fn sentence_falls_back_to_whitespace_with_no_terminator() {
        let text = "no terminators here";
        let idx = find_chunk_boundary(text, text.len(), BoundaryStrategy::Sentence);
        // last space sits between "terminators" and "here"
        assert_eq!(idx, text.rfind(' ').unwrap());
    }

    #[test]
    fn sentence_recognises_cjk_terminator() {
        let text = "こんにちは。次の文";
        let idx = find_chunk_boundary(text, text.len(), BoundaryStrategy::Sentence);
        // '。' starts at byte 15 (5 hiragana × 3 bytes) and is 3 bytes wide,
        // so the returned position is 18.
        assert_eq!(idx, 18);
        assert_eq!(&text[..idx], "こんにちは。");
    }

    #[test]
    fn paragraph_prefers_double_newline() {
        let text = "first turn\n\nsecond turn";
        let idx = find_chunk_boundary(text, text.len(), BoundaryStrategy::Paragraph);
        assert_eq!(idx, 10);
    }

    #[test]
    fn paragraph_falls_back_through_sentence_to_whitespace() {
        let text = "no paragraph here just a sentence. tail";
        let idx = find_chunk_boundary(text, text.len(), BoundaryStrategy::Paragraph);
        // Paragraph fallback lands on the sentence boundary one byte after '.'
        let expected = text.rfind('.').unwrap() + 1;
        assert_eq!(idx, expected);
    }

    #[test]
    fn paragraph_handles_crlf_delimiter() {
        let text = "first turn\r\n\r\nsecond";
        let idx = find_chunk_boundary(text, text.len(), BoundaryStrategy::Paragraph);
        assert_eq!(idx, 10);
    }

    #[test]
    fn parse_accepts_known_strategies() {
        assert_eq!(BoundaryStrategy::parse("whitespace").unwrap(), BoundaryStrategy::Whitespace);
        assert_eq!(BoundaryStrategy::parse("sentence").unwrap(), BoundaryStrategy::Sentence);
        assert_eq!(BoundaryStrategy::parse("paragraph").unwrap(), BoundaryStrategy::Paragraph);
    }

    #[test]
    fn parse_rejects_unknown_strategy() {
        assert!(BoundaryStrategy::parse("banana").is_err());
    }
}
