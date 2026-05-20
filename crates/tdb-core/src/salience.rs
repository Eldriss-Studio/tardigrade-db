//! Salience derivation modes for pack writes.
//!
//! A pack's *salience* is a 0..100 score that seeds its initial position
//! on the importance ladder. Two write contracts coexist:
//!
//! * The caller may pass an explicit salience the engine stores verbatim
//!   (the default for legacy `mem_write_pack` callers).
//! * The caller may pass a [`SalienceMode`] and let the engine derive
//!   salience from the retrieval key's magnitude. This keeps the formula
//!   in Rust, out of every consumer's Python layer, and consistent across
//!   producers (`HuggingFace` hook, vLLM connector, MCP server).
//!
//! All derived modes apply [`SALIENCE_SCALE`] and clamp at [`SALIENCE_CAP`]
//! so callers cannot accidentally over-rate a pack on a single write.

/// Multiplier mapping a per-element retrieval-key magnitude into the
/// 0..100 salience range. Chosen so that typical N(0, 1) hidden-state
/// data sits in the 5..40 band — well clear of the cap.
pub const SALIENCE_SCALE: f32 = 50.0;

/// Maximum salience an automated derivation may assign. Caller-supplied
/// salience is **not** clamped — explicit always wins.
pub const SALIENCE_CAP: f32 = 100.0;

/// How the engine should derive salience from the encoded retrieval key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SalienceMode {
    /// Preserve the caller's explicit salience. The default — and the
    /// behaviour when no `salience_mode` argument is passed.
    Explicit,
    /// `min(SALIENCE_SCALE * (||key_data|| / len(key_data)), SALIENCE_CAP)`.
    /// L2 norm of the *non-header* portion of the encoded key, divided
    /// by the number of elements to get a per-element magnitude.
    L2,
    /// `min(SALIENCE_SCALE * max(|key_data|), SALIENCE_CAP)`. Peak
    /// magnitude rather than average — surfaces packs with at least one
    /// strong activation even if most tokens are flat.
    Max,
}

impl SalienceMode {
    /// Parse the string form callers pass through the `PyO3` layer.
    pub fn parse(name: &str) -> Result<Self, UnknownSalienceMode> {
        match name {
            "none" | "explicit" => Ok(Self::Explicit),
            "l2" => Ok(Self::L2),
            "max" => Ok(Self::Max),
            other => Err(UnknownSalienceMode { name: other.to_owned() }),
        }
    }

    /// Resolve the effective salience given the caller's explicit value
    /// and the encoded retrieval key. The `header_size` argument is how
    /// many leading f32s of `encoded_key` to skip before measuring —
    /// callers pass `tdb_retrieval::per_token::HEADER_SIZE` for the
    /// per-token encoded form.
    pub fn resolve(self, explicit: f32, encoded_key: &[f32], header_size: usize) -> f32 {
        if self == Self::Explicit || encoded_key.len() <= header_size {
            return explicit;
        }
        let data = &encoded_key[header_size..];
        let raw = match self {
            Self::L2 => {
                let n = data.len().max(1) as f32;
                let sum_sq: f32 = data.iter().map(|v| v * v).sum();
                (sum_sq.sqrt() / n) * SALIENCE_SCALE
            }
            Self::Max => {
                let abs_max = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
                abs_max * SALIENCE_SCALE
            }
            Self::Explicit => return explicit,
        };
        raw.min(SALIENCE_CAP)
    }
}

/// Returned when [`SalienceMode::parse`] receives an unknown name.
#[derive(Debug, Clone)]
pub struct UnknownSalienceMode {
    pub name: String,
}

impl std::fmt::Display for UnknownSalienceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown salience_mode {:?}; expected one of \"l2\", \"max\", \"none\"",
            self.name
        )
    }
}

impl std::error::Error for UnknownSalienceMode {}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: usize = 4;

    #[test]
    fn explicit_returns_caller_value() {
        assert!((SalienceMode::Explicit.resolve(42.0, &[0.0; 32], HEADER) - 42.0).abs() < 1e-6);
    }

    #[test]
    fn l2_uses_non_header_portion() {
        // Header is all zeros; data is [1.0, 1.0, 1.0, 1.0].
        let mut key = vec![0.0_f32; HEADER];
        key.extend([1.0; 4]);
        let s = SalienceMode::L2.resolve(0.0, &key, HEADER);
        // L2 = sqrt(4) = 2.0; per-element = 0.5; scaled = 25.0.
        assert!((s - 25.0).abs() < 1e-4, "l2 salience {s} != 25.0");
    }

    #[test]
    fn max_uses_inf_norm_of_non_header_portion() {
        let mut key = vec![0.0_f32; HEADER];
        key.extend([0.1, 0.5, -0.7, 0.3]);
        let s = SalienceMode::Max.resolve(0.0, &key, HEADER);
        // max(|x|) = 0.7; scaled = 35.0.
        assert!((s - 35.0).abs() < 1e-4, "max salience {s} != 35.0");
    }

    #[test]
    fn clamps_at_cap() {
        let mut key = vec![0.0_f32; HEADER];
        key.extend([100.0_f32; 4]);
        assert!((SalienceMode::Max.resolve(0.0, &key, HEADER) - SALIENCE_CAP).abs() < 1e-6);
    }

    #[test]
    fn short_key_falls_back_to_explicit() {
        let key = vec![0.0_f32; HEADER];
        assert!((SalienceMode::L2.resolve(7.5, &key, HEADER) - 7.5).abs() < 1e-6);
    }

    #[test]
    fn parse_accepts_known_modes() {
        assert_eq!(SalienceMode::parse("l2").unwrap(), SalienceMode::L2);
        assert_eq!(SalienceMode::parse("max").unwrap(), SalienceMode::Max);
        assert_eq!(SalienceMode::parse("none").unwrap(), SalienceMode::Explicit);
        assert_eq!(SalienceMode::parse("explicit").unwrap(), SalienceMode::Explicit);
    }

    #[test]
    fn parse_rejects_unknown_mode() {
        assert!(SalienceMode::parse("banana").is_err());
    }
}
