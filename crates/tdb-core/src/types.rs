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
    #[must_use]
    pub fn retrieval_boost(self) -> f32 {
        match self {
            Self::Draft => DRAFT_RETRIEVAL_BOOST,
            Self::Validated => VALIDATED_RETRIEVAL_BOOST,
            Self::Core => CORE_RETRIEVAL_BOOST,
        }
    }
}

/// Read-visibility mode for retrieval APIs that participate in the
/// confirmed-vs-unconfirmed durability contract.
///
/// [`ReadVisibility::Unconfirmed`] is the default and existing
/// behaviour: the read returns immediately with whatever the
/// retrieval pipeline currently has. [`ReadVisibility::Confirmed`]
/// blocks until every write issued before the read became durable.
///
/// See CLAUDE.md's "Reliability & Consistency Rules" section for the
/// project-canonical contract this enum surfaces, and
/// `~/.claude/plans/spacetimedb-confirmed-reads-contract.md` for the
/// implementation plan.
///
/// # Why an explicit timeout is required on `Confirmed`
///
/// A confirmed read with no deadline can block forever — if the
/// underlying write fails fsync without surfacing an error, the
/// reader's wait will never complete. Requiring an explicit deadline
/// makes that failure path bounded: the reader sees a `ReadTimeout`
/// instead of hanging, and the consumer's higher-level error
/// handling can decide what to do.
#[derive(Debug, Clone, Copy, Default)]
pub enum ReadVisibility {
    /// Default mode — return immediately, no durability wait.
    #[default]
    Unconfirmed,
    /// Block until the snapshot offset captured at request entry is
    /// durable, or until `timeout` elapses (whichever comes first).
    Confirmed {
        /// Maximum wall-clock wait before returning `ReadTimeout`.
        timeout: std::time::Duration,
    },
}
