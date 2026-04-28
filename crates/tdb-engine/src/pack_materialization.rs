//! Pack-read materialization value objects.
//!
//! `Engine::mem_read_pack` retrieves pack key cells first, then materializes
//! complete packs. These types keep that boundary explicit so ranking, pack
//! deduplication, layer hydration, and final result construction remain separate
//! responsibilities.

use std::collections::HashSet;

use tdb_core::kv_pack::{KVLayerPayload, KVPack, PackId, PackReadResult};
use tdb_core::{CellId, OwnerId, Tier};

#[derive(Debug, Clone)]
pub(crate) struct PackCandidate {
    pub(crate) pack_id: PackId,
    pub(crate) retrieval_cell_id: CellId,
    pub(crate) owner: OwnerId,
    pub(crate) score: f32,
    pub(crate) cell_ids: Vec<CellId>,
}

impl PackCandidate {
    pub(crate) fn new(
        pack_id: PackId,
        retrieval_cell_id: CellId,
        owner: OwnerId,
        score: f32,
        cell_ids: Vec<CellId>,
    ) -> Self {
        Self { pack_id, retrieval_cell_id, owner, score, cell_ids }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PackAccessSnapshot {
    pub(crate) tier: Tier,
    pub(crate) decay_factor: f32,
    pub(crate) tier_boost: f32,
}

#[cfg(test)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PackMaterializationCounters {
    pub(crate) candidate_count: usize,
    pub(crate) unique_pack_count: usize,
    pub(crate) hydrated_layer_count: usize,
    pub(crate) returned_pack_count: usize,
}

#[cfg(test)]
impl PackMaterializationCounters {
    pub(crate) fn total_observed_items(self) -> usize {
        self.candidate_count
            + self.unique_pack_count
            + self.hydrated_layer_count
            + self.returned_pack_count
    }
}

#[cfg(test)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PackMaterializationPhaseProfile {
    pub(crate) candidate_retrieval_nanos: u64,
    pub(crate) candidate_sort_dedup_nanos: u64,
    pub(crate) pack_lookup_nanos: u64,
    pub(crate) layer_hydration_nanos: u64,
    pub(crate) governance_nanos: u64,
    pub(crate) result_construction_nanos: u64,
    pub(crate) counters: PackMaterializationCounters,
}

#[cfg(test)]
impl PackMaterializationPhaseProfile {
    pub(crate) fn total_nanos(self) -> u64 {
        self.candidate_retrieval_nanos
            + self.candidate_sort_dedup_nanos
            + self.pack_lookup_nanos
            + self.layer_hydration_nanos
            + self.governance_nanos
            + self.result_construction_nanos
    }
}

pub(crate) fn keep_first_ranked_pack_candidates(
    candidates: Vec<PackCandidate>,
    limit: usize,
) -> Vec<PackCandidate> {
    let mut seen_packs = HashSet::new();
    let mut selected = Vec::with_capacity(limit.min(candidates.len()));

    for candidate in candidates {
        if seen_packs.insert(candidate.pack_id) {
            selected.push(candidate);
        }
        if selected.len() >= limit {
            break;
        }
    }

    selected
}

pub(crate) fn build_pack_read_result(
    candidate: &PackCandidate,
    layers: Vec<KVLayerPayload>,
    access: PackAccessSnapshot,
) -> PackReadResult {
    PackReadResult {
        pack: KVPack {
            id: candidate.pack_id,
            owner: candidate.owner,
            retrieval_key: Vec::new(),
            layers,
            salience: 0.0,
            text: None,
        },
        score: candidate.score * access.decay_factor * access.tier_boost,
        tier: access.tier,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANDIDATE_COUNT: usize = 9;
    const UNIQUE_PACK_COUNT: usize = 4;
    const HYDRATED_LAYER_COUNT: usize = 12;
    const RETURNED_PACK_COUNT: usize = 3;
    const EXPECTED_TOTAL_ITEMS: usize = 28;
    const CANDIDATE_RETRIEVAL_NANOS: u64 = 1_000;
    const SORT_DEDUP_NANOS: u64 = 200;
    const LOOKUP_NANOS: u64 = 300;
    const HYDRATION_NANOS: u64 = 700;
    const GOVERNANCE_NANOS: u64 = 100;
    const RESULT_NANOS: u64 = 150;
    const EXPECTED_TOTAL_NANOS: u64 = 2_450;
    const FIRST_PACK_ID: PackId = 7;
    const SECOND_PACK_ID: PackId = 11;
    const FIRST_RETRIEVAL_CELL: CellId = 70;
    const DUPLICATE_RETRIEVAL_CELL: CellId = 71;
    const SECOND_RETRIEVAL_CELL: CellId = 110;
    const OWNER: OwnerId = 1;
    const FIRST_SCORE: f32 = 9.0;
    const DUPLICATE_SCORE: f32 = 8.0;
    const SECOND_SCORE: f32 = 7.0;
    const DEDUP_LIMIT: usize = 2;

    #[test]
    fn test_pack_materialization_profile_sums_phase_counts() {
        let counters = PackMaterializationCounters {
            candidate_count: CANDIDATE_COUNT,
            unique_pack_count: UNIQUE_PACK_COUNT,
            hydrated_layer_count: HYDRATED_LAYER_COUNT,
            returned_pack_count: RETURNED_PACK_COUNT,
        };
        let profile = PackMaterializationPhaseProfile {
            candidate_retrieval_nanos: CANDIDATE_RETRIEVAL_NANOS,
            candidate_sort_dedup_nanos: SORT_DEDUP_NANOS,
            pack_lookup_nanos: LOOKUP_NANOS,
            layer_hydration_nanos: HYDRATION_NANOS,
            governance_nanos: GOVERNANCE_NANOS,
            result_construction_nanos: RESULT_NANOS,
            counters,
        };

        assert_eq!(profile.counters.total_observed_items(), EXPECTED_TOTAL_ITEMS);
        assert_eq!(profile.total_nanos(), EXPECTED_TOTAL_NANOS);
    }

    #[test]
    fn test_pack_candidate_dedup_keeps_first_ranked_pack() {
        let candidates = vec![
            PackCandidate::new(
                FIRST_PACK_ID,
                FIRST_RETRIEVAL_CELL,
                OWNER,
                FIRST_SCORE,
                vec![FIRST_RETRIEVAL_CELL],
            ),
            PackCandidate::new(
                FIRST_PACK_ID,
                DUPLICATE_RETRIEVAL_CELL,
                OWNER,
                DUPLICATE_SCORE,
                vec![DUPLICATE_RETRIEVAL_CELL],
            ),
            PackCandidate::new(
                SECOND_PACK_ID,
                SECOND_RETRIEVAL_CELL,
                OWNER,
                SECOND_SCORE,
                vec![SECOND_RETRIEVAL_CELL],
            ),
        ];

        let deduped = keep_first_ranked_pack_candidates(candidates, DEDUP_LIMIT);

        assert_eq!(deduped.len(), DEDUP_LIMIT);
        assert_eq!(deduped[0].pack_id, FIRST_PACK_ID);
        assert_eq!(deduped[0].retrieval_cell_id, FIRST_RETRIEVAL_CELL);
        assert_eq!(deduped[1].pack_id, SECOND_PACK_ID);
    }
}
