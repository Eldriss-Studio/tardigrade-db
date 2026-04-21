use half::f16;

use crate::types::{OwnerId, SynapticId};

/// A per-agent/user `LoRA` adapter pack for weight-like memory.
///
/// Unlike episodic KV memory (`MemoryCell`), synaptic entries encode
/// stable preferences and patterns as low-rank weight deltas
/// applied to the base model at runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct SynapticBankEntry {
    /// Unique identifier for this adapter set.
    pub id: SynapticId,
    /// The agent/user these adapters belong to.
    pub owner: OwnerId,
    /// `LoRA` matrix A (rank × `d_model`), stored in FP16.
    pub lora_a: Vec<f16>,
    /// `LoRA` matrix B (`d_model` × rank), stored in FP16.
    pub lora_b: Vec<f16>,
    /// Scaling factor applied to the low-rank delta.
    pub scale: f16,
    /// Rank of the `LoRA` decomposition.
    pub rank: u32,
    /// Model dimension (`d_model`).
    pub d_model: u32,
    /// Unix timestamp of last use.
    pub last_used: u64,
    /// Quality score from evaluation feedback.
    pub quality: f32,
}

impl SynapticBankEntry {
    /// Creates a new entry, validating that matrix dimensions match `rank` and `d_model`.
    ///
    /// # Panics
    /// Panics if `lora_a.len() != rank * d_model` or `lora_b.len() != d_model * rank`.
    pub fn new(
        id: SynapticId,
        owner: OwnerId,
        lora_a: Vec<f16>,
        lora_b: Vec<f16>,
        scale: f16,
        rank: u32,
        d_model: u32,
    ) -> Self {
        let expected = (rank as usize) * (d_model as usize);
        assert_eq!(
            lora_a.len(),
            expected,
            "lora_a length {} does not match rank({}) × d_model({})",
            lora_a.len(),
            rank,
            d_model
        );
        assert_eq!(
            lora_b.len(),
            expected,
            "lora_b length {} does not match d_model({}) × rank({})",
            lora_b.len(),
            d_model,
            rank
        );
        Self { id, owner, lora_a, lora_b, scale, rank, d_model, last_used: 0, quality: 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_validates_dimensions() {
        let entry = SynapticBankEntry::new(
            1,
            42,
            vec![f16::from_f32(1.0); 8], // rank=2, d_model=4 → 8 elements
            vec![f16::from_f32(2.0); 8],
            f16::from_f32(0.5),
            2,
            4,
        );
        assert_eq!(entry.rank, 2);
        assert_eq!(entry.d_model, 4);
        assert_eq!(entry.lora_a.len(), 8);
    }

    #[test]
    #[should_panic(expected = "lora_a length")]
    fn test_new_rejects_wrong_dimensions() {
        SynapticBankEntry::new(
            1,
            42,
            vec![f16::from_f32(1.0); 5], // wrong size
            vec![f16::from_f32(2.0); 8],
            f16::from_f32(0.5),
            2,
            4,
        );
    }
}
