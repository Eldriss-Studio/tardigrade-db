//! Pack-read profiling value objects.
//!
//! These types keep phase math and benchmark-label formatting explicit. They
//! are test-only implementation details, so the formatting remains locked
//! without changing the public engine API.

#[derive(Debug, Clone, Copy)]
pub(crate) struct PackReadPhaseProfile {
    pub(crate) retrieval_nanos: u64,
    pub(crate) pack_read_nanos: u64,
    pub(crate) hydration_nanos: u64,
}

impl PackReadPhaseProfile {
    pub(crate) fn reconstruction_delta_nanos(self) -> u64 {
        self.pack_read_nanos.saturating_sub(self.retrieval_nanos)
    }

    pub(crate) fn storage_gap_nanos(self) -> u64 {
        self.pack_read_nanos.saturating_sub(self.hydration_nanos)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PackReadProfileLabel {
    pub(crate) target_top1: bool,
    pub(crate) dedup_ok: bool,
    pub(crate) layer_count: usize,
    pub(crate) payload_dim: usize,
    pub(crate) indexed_cells_per_pack: usize,
}

impl PackReadProfileLabel {
    pub(crate) fn as_benchmark_id(self) -> String {
        format!(
            "target-{}-dedup-{}-layers-{}-payload-{}-indexed-{}",
            self.target_top1,
            self.dedup_ok,
            self.layer_count,
            self.payload_dim,
            self.indexed_cells_per_pack
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StorageHydrationLabel {
    pub(crate) payload_dim: usize,
    pub(crate) cell_count: usize,
}

impl StorageHydrationLabel {
    pub(crate) fn as_benchmark_id(self) -> String {
        format!("payload-{}-cells-{}", self.payload_dim, self.cell_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RETRIEVAL_NANOS: u64 = 1_000;
    const PACK_READ_NANOS: u64 = 4_500;
    const HYDRATION_NANOS: u64 = 2_000;
    const EXPECTED_RECONSTRUCTION_DELTA_NANOS: u64 = 3_500;
    const EXPECTED_STORAGE_GAP_NANOS: u64 = 2_500;
    const PROFILE_LAYER_COUNT: usize = 4;
    const PROFILE_PAYLOAD_DIM: usize = 128;
    const PROFILE_INDEXED_CELLS_PER_PACK: usize = 1;
    const PROFILE_CELL_COUNT: usize = 10_000;

    #[test]
    fn test_pack_read_profile_metrics_match_known_timings() {
        let profile = PackReadPhaseProfile {
            retrieval_nanos: RETRIEVAL_NANOS,
            pack_read_nanos: PACK_READ_NANOS,
            hydration_nanos: HYDRATION_NANOS,
        };

        assert_eq!(profile.reconstruction_delta_nanos(), EXPECTED_RECONSTRUCTION_DELTA_NANOS);
        assert_eq!(profile.storage_gap_nanos(), EXPECTED_STORAGE_GAP_NANOS);
    }

    #[test]
    fn test_pack_profile_label_includes_layer_count_and_payload_dim() {
        let label = PackReadProfileLabel {
            target_top1: true,
            dedup_ok: true,
            layer_count: PROFILE_LAYER_COUNT,
            payload_dim: PROFILE_PAYLOAD_DIM,
            indexed_cells_per_pack: PROFILE_INDEXED_CELLS_PER_PACK,
        }
        .as_benchmark_id();

        assert!(label.contains("target-true"));
        assert!(label.contains("dedup-true"));
        assert!(label.contains("layers-4"));
        assert!(label.contains("payload-128"));
        assert!(label.contains("indexed-1"));
    }

    #[test]
    fn test_storage_hydration_label_includes_payload_dim_and_cell_count() {
        let label = StorageHydrationLabel {
            payload_dim: PROFILE_PAYLOAD_DIM,
            cell_count: PROFILE_CELL_COUNT,
        }
        .as_benchmark_id();

        assert!(label.contains("payload-128"));
        assert!(label.contains("cells-10000"));
    }
}
