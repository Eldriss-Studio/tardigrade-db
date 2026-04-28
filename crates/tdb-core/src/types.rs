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

/// Retrieval boost for Draft-tier memories (no advantage).
const DRAFT_RETRIEVAL_BOOST: f32 = 1.0;
/// Retrieval boost for Validated-tier memories (accessed enough to cross ι≥65).
const VALIDATED_RETRIEVAL_BOOST: f32 = 1.1;
/// Retrieval boost for Core-tier memories (stable, repeatedly accessed, ι≥85).
const CORE_RETRIEVAL_BOOST: f32 = 1.25;

impl Tier {
    /// Score multiplier applied during retrieval based on maturity.
    ///
    /// Core memories have proven their value through repeated access;
    /// they rank higher than untested Draft memories.
    pub fn retrieval_boost(self) -> f32 {
        match self {
            Self::Draft => DRAFT_RETRIEVAL_BOOST,
            Self::Validated => VALIDATED_RETRIEVAL_BOOST,
            Self::Core => CORE_RETRIEVAL_BOOST,
        }
    }
}
