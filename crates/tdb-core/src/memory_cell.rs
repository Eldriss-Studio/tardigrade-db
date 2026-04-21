use crate::types::{CellId, LayerId, OwnerId, TagBits, Tier};

/// Metadata governing a cell's lifecycle within the Adaptive Knowledge Lifecycle.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CellMeta {
    /// Unix timestamp (nanoseconds) when this cell was created.
    pub created_at: u64,
    /// Unix timestamp (nanoseconds) of the last access or update.
    pub updated_at: u64,
    /// Importance score ι ∈ [0.0, 100.0]. Boosted on access (+3) and update (+5),
    /// decayed daily by factor 0.995.
    pub importance: f32,
    /// Categorical tags as a bitfield.
    pub tags: TagBits,
    /// Current maturity tier in the AKL.
    pub tier: Tier,
}

/// A single memory cell — the fundamental storage unit in `TardigradeDB`.
///
/// Mirrors the transformer's attention internals: `key` and `value` vectors
/// have the same dimensionality as the model's KV cache at a given layer.
/// Cells are persisted in a custom mmap'd arena (not safetensors) for O(1) access.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryCell {
    /// Globally unique identifier.
    pub id: CellId,
    /// The agent/user that owns this cell.
    pub owner: OwnerId,
    /// Transformer layer this KV pair was captured from.
    pub layer: LayerId,
    /// Key vector (same dimensionality as the model's key projection at `layer`).
    pub key: Vec<f32>,
    /// Value vector (same dimensionality as the model's value projection at `layer`).
    pub value: Vec<f32>,
    /// Token span (start, end) in the original sequence this cell covers.
    pub token_span: (u64, u64),
    /// Position encoding captured alongside the KV pair for safe historical reuse.
    pub pos_encoding: Vec<f32>,
    /// Lifecycle metadata.
    pub meta: CellMeta,
}

/// Builder for constructing `MemoryCell` instances.
///
/// Required fields: `id`, `owner`, `layer`, `key`, `value`.
/// Optional fields default to zero/empty.
#[derive(Debug)]
pub struct MemoryCellBuilder {
    id: CellId,
    owner: OwnerId,
    layer: LayerId,
    key: Vec<f32>,
    value: Vec<f32>,
    token_span: (u64, u64),
    pos_encoding: Vec<f32>,
    meta: CellMeta,
}

impl MemoryCellBuilder {
    pub fn new(id: CellId, owner: OwnerId, layer: LayerId, key: Vec<f32>, value: Vec<f32>) -> Self {
        Self {
            id,
            owner,
            layer,
            key,
            value,
            token_span: (0, 0),
            pos_encoding: Vec::new(),
            meta: CellMeta::default(),
        }
    }

    pub fn token_span(mut self, start: u64, end: u64) -> Self {
        self.token_span = (start, end);
        self
    }

    pub fn pos_encoding(mut self, encoding: Vec<f32>) -> Self {
        self.pos_encoding = encoding;
        self
    }

    pub fn importance(mut self, importance: f32) -> Self {
        self.meta.importance = importance;
        self
    }

    pub fn tags(mut self, tags: TagBits) -> Self {
        self.meta.tags = tags;
        self
    }

    pub fn tier(mut self, tier: Tier) -> Self {
        self.meta.tier = tier;
        self
    }

    pub fn created_at(mut self, ts: u64) -> Self {
        self.meta.created_at = ts;
        self
    }

    pub fn updated_at(mut self, ts: u64) -> Self {
        self.meta.updated_at = ts;
        self
    }

    pub fn build(self) -> MemoryCell {
        MemoryCell {
            id: self.id,
            owner: self.owner,
            layer: self.layer,
            key: self.key,
            value: self.value,
            token_span: self.token_span,
            pos_encoding: self.pos_encoding,
            meta: self.meta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let cell = MemoryCellBuilder::new(1, 42, 12, vec![1.0, 2.0], vec![3.0, 4.0]).build();

        assert_eq!(cell.id, 1);
        assert_eq!(cell.owner, 42);
        assert_eq!(cell.layer, 12);
        assert_eq!(cell.key, vec![1.0, 2.0]);
        assert_eq!(cell.value, vec![3.0, 4.0]);
        assert_eq!(cell.token_span, (0, 0));
        assert!(cell.pos_encoding.is_empty());
        assert_eq!(cell.meta.tier, Tier::Draft);
        assert!((cell.meta.importance - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder_with_all_fields() {
        let cell = MemoryCellBuilder::new(1, 42, 12, vec![1.0], vec![2.0])
            .token_span(10, 20)
            .pos_encoding(vec![0.5])
            .importance(75.0)
            .tags(0b1010)
            .tier(Tier::Validated)
            .created_at(1000)
            .updated_at(2000)
            .build();

        assert_eq!(cell.token_span, (10, 20));
        assert_eq!(cell.pos_encoding, vec![0.5]);
        assert!((cell.meta.importance - 75.0).abs() < f32::EPSILON);
        assert_eq!(cell.meta.tags, 0b1010);
        assert_eq!(cell.meta.tier, Tier::Validated);
        assert_eq!(cell.meta.created_at, 1000);
        assert_eq!(cell.meta.updated_at, 2000);
    }
}
