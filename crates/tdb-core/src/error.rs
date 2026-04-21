use thiserror::Error;

use crate::types::CellId;

#[derive(Debug, Error)]
pub enum TardigradeError {
    #[error("cell not found: {0}")]
    CellNotFound(CellId),

    #[error("segment full: cannot append to segment at {path}")]
    SegmentFull { path: String },

    #[error("storage I/O error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    #[error("quantization error: {0}")]
    Quantization(String),

    #[error("dimension mismatch: expected {expected}, got {actual} for {context}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
        context: String,
    },

    #[error("WAL recovery failed: {0}")]
    WalRecovery(String),

    #[error("index error: {0}")]
    Index(String),

    #[error("governance error: {0}")]
    Governance(String),

    #[error("capacity exceeded: {0}")]
    CapacityExceeded(String),
}

pub type Result<T> = std::result::Result<T, TardigradeError>;
