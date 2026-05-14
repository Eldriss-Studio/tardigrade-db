//! Retrieval-side quantization — distinct from `tdb-storage`'s
//! archival quantization.
//!
//! `TardigradeDB` stores tokens in two places:
//!
//! - **Archival** (`tdb-storage/src/quantization.rs::Q4`): on-disk
//!   representation. Aggressive 4-bit group quantization for space.
//!   *Not* used for retrieval scoring — `Q4_0` collapses non-outlier
//!   values to zero when activation outliers dominate a group
//!   (Dettmers 2022, "`LLM.int8()`").
//!
//! - **Retrieval** (this module): the precision tier used at scoring
//!   time. `INT8` group quantization paired with a per-channel
//!   pre-scaling sidecar (`SmoothQuant`-for-storage pattern). Keeps
//!   outlier crush out of the scoring path.
//!
//! See `docs/refs/external-references.md` §A3f for the literature
//! body that gates this design (`LLM.int8`, `SmoothQuant`, `AWQ`,
//! `ColBERTv2`, `Faiss` SQ8 / `ScaNN` / `Qdrant` production
//! references).
//!
//! # Module layout
//!
//! - [`calibrator`] — [`PerChannelScaleCalibrator`]: accumulates the
//!   per-dimension scale vector σ from a calibration sample.
//! - [`int8_group`] — [`Int8Group32`] plain INT8 group quantizer.
//! - [`per_channel_scaled`] — [`PerChannelScaled`] decorator that
//!   wraps an inner strategy with calibrated per-channel σ.

pub mod calibrator;
pub mod int8_group;
pub mod per_channel_scaled;
pub mod strategy;

pub use calibrator::{
    DEFAULT_CALIBRATION_WINDOW_TOKENS, PER_CHANNEL_SCALE_FLOOR, PerChannelScaleCalibrator,
};
pub use int8_group::{INT8_GROUP_SIZE, Int8Group32};
pub use per_channel_scaled::PerChannelScaled;
pub use strategy::{QuantizedToken, RetrievalQuantStrategy};
