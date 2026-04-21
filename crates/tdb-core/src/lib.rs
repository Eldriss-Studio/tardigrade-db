pub mod error;
pub mod memory_cell;
pub mod synaptic_bank;
pub mod types;

pub use error::TardigradeError;
pub use memory_cell::MemoryCell;
pub use synaptic_bank::SynapticBankEntry;
pub use types::{CellId, LayerId, OwnerId, SynapticId, TagBits, Tier};
