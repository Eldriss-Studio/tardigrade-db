//! Pluggable compression codecs for already-quantized cell bytes.
//!
//! Each `MemoryCell` is first Q4-quantized (lossy, ~4× shrink over raw f32) and
//! then optionally passed through a [`CompressionCodec`] before being appended to
//! the segment record. The codec is selected per-cell by tier so that the
//! cold/warm distinction can be exploited without changing the retrieval path:
//!
//! - Draft cells use [`CompressionCodec::UniformQ4`] (the raw Q4 byte stream).
//!   Writes turn over fast and pay no codec cost.
//! - Validated and Core cells use [`CompressionCodec::ZstdQ4`], which wraps the
//!   Q4 bytes in zstd level 3 for an additional ~3–4× footprint reduction.
//!
//! The codec is identified per-record by a single `codec_id` byte stored in the
//! segment record header (segment file format `v2`). Old (`v1`) segments contain
//! no `codec_id` byte and are read as [`CompressionCodec::UniformQ4`] unconditionally
//! for backward compatibility.
//!
//! # Why this lives next to [`quantization`]
//!
//! Quantization decides the *value space*: how a tensor element becomes bits.
//! Compression decides the *byte layout*: how those bits are laid out on disk.
//! They are independent concerns and pluggable independently — a future
//! variable-bitrate quantizer would still be free to compose with the zstd codec.
//!
//! [`quantization`]: crate::quantization

use std::io;

/// Identifier for the compression codec applied to a cell's quantized bytes.
///
/// Stored as a single byte in the segment record (segment format v2). Numeric
/// values are part of the on-disk format and must not change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CompressionCodec {
    /// No additional compression — the Q4 byte stream is written verbatim.
    /// Used for Draft-tier writes and for all reads of segment format v1.
    UniformQ4 = 0,
    /// zstd level 3 over the Q4 byte stream. Used for Validated/Core writes.
    ZstdQ4 = 1,
}

impl CompressionCodec {
    /// Parse a codec id byte read from a segment record.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidData`] if the byte does not correspond to
    /// any known codec — guards against forward-incompatible segment formats.
    pub fn from_u8(byte: u8) -> io::Result<Self> {
        match byte {
            0 => Ok(Self::UniformQ4),
            1 => Ok(Self::ZstdQ4),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown compression codec id {other}"),
            )),
        }
    }

    /// On-disk byte representation.
    #[must_use]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Encode raw Q4 bytes for on-disk storage.
    ///
    /// `UniformQ4` is a no-op; `ZstdQ4` wraps the input in a zstd level 3 frame.
    ///
    /// # Errors
    /// Returns [`io::Error`] if the underlying zstd encoder fails. The `UniformQ4`
    /// variant cannot fail.
    pub fn encode(self, q4_bytes: &[u8]) -> io::Result<Vec<u8>> {
        match self {
            Self::UniformQ4 => Ok(q4_bytes.to_vec()),
            Self::ZstdQ4 => zstd::encode_all(q4_bytes, ZSTD_LEVEL),
        }
    }

    /// Decode bytes read off disk back into the raw Q4 byte stream.
    ///
    /// # Errors
    /// Returns [`io::Error`] if a zstd frame is malformed (corruption or
    /// codec-id/byte-payload mismatch).
    pub fn decode(self, on_disk: &[u8]) -> io::Result<Vec<u8>> {
        match self {
            Self::UniformQ4 => Ok(on_disk.to_vec()),
            Self::ZstdQ4 => zstd::decode_all(on_disk),
        }
    }
}

/// zstd compression level used by [`CompressionCodec::ZstdQ4`].
///
/// Level 3 is the encode/ratio sweet spot for byte-oriented payloads — level 19
/// buys ~5% more compression for 10× the encode time, which is not worth the
/// write-path latency on a memory engine. Documented in `docs/refs/`.
const ZSTD_LEVEL: i32 = 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_q4_is_byte_identity() {
        let bytes: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let encoded = CompressionCodec::UniformQ4.encode(&bytes).unwrap();
        let decoded = CompressionCodec::UniformQ4.decode(&encoded).unwrap();
        assert_eq!(encoded, bytes);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn zstd_q4_round_trip_preserves_bytes() {
        let bytes: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let encoded = CompressionCodec::ZstdQ4.encode(&bytes).unwrap();
        let decoded = CompressionCodec::ZstdQ4.decode(&encoded).unwrap();
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn zstd_q4_handles_empty_payload() {
        let bytes: Vec<u8> = Vec::new();
        let encoded = CompressionCodec::ZstdQ4.encode(&bytes).unwrap();
        let decoded = CompressionCodec::ZstdQ4.decode(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn from_u8_round_trips_known_codecs() {
        for codec in [CompressionCodec::UniformQ4, CompressionCodec::ZstdQ4] {
            assert_eq!(CompressionCodec::from_u8(codec.as_u8()).unwrap(), codec);
        }
    }

    #[test]
    fn from_u8_rejects_unknown_codec() {
        let err = CompressionCodec::from_u8(7).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn zstd_q4_decoding_garbage_fails() {
        // A buffer that isn't a valid zstd frame must surface as InvalidData
        // rather than corrupting the read path silently.
        let garbage = vec![0xffu8; 32];
        let err = CompressionCodec::ZstdQ4.decode(&garbage).unwrap_err();
        assert!(matches!(err.kind(), io::ErrorKind::InvalidData | io::ErrorKind::Other));
    }
}
