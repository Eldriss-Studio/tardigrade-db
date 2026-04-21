/// Unique identifier for a memory cell within the storage engine.
pub type CellId = u64;

/// Identifier for an agent or user that owns memory cells.
pub type OwnerId = u64;

/// Transformer layer index from which KV tensors were captured.
pub type LayerId = u16;

/// Bitfield for tagging memory cells with categorical metadata.
pub type TagBits = u32;

/// Unique identifier for a synaptic bank entry (`LoRA` adapter pack).
pub type SynapticId = u64;

/// Maturity tier in the Adaptive Knowledge Lifecycle.
///
/// Transitions use hysteresis to prevent oscillation:
/// - Draft → Validated at ι ≥ 65, demotes back at ι < 35
/// - Validated → Core at ι ≥ 85, demotes back at ι < 60
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Tier {
    #[default]
    Draft = 0,
    Validated = 1,
    Core = 2,
}
