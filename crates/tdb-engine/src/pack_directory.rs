//! KV pack directory — Value Object for pack membership and reverse lookup.
//!
//! `Engine::mem_read_pack` starts from retrieval results, which are cell IDs.
//! The engine then needs to reconstruct complete packs. Keeping both
//! `pack_id -> cell_ids` and `cell_id -> pack_id` in one private value object
//! makes that mapping explicit and keeps reverse lookup O(1).

use std::collections::{HashMap, HashSet};

use tdb_core::CellId;
use tdb_core::kv_pack::PackId;

const FIRST_ASSIGNED_PACK_ID: PackId = 1;

#[derive(Debug, Default)]
pub(crate) struct PackDirectory {
    cells_by_pack: HashMap<PackId, Vec<CellId>>,
    pack_by_cell: HashMap<CellId, PackId>,
}

impl PackDirectory {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn from_cells<I>(cells: I) -> Self
    where
        I: IntoIterator<Item = (PackId, CellId)>,
    {
        let mut directory = Self::new();
        for (pack_id, cell_id) in cells {
            directory.add_cell(pack_id, cell_id);
        }
        directory.sort_pack_cells();
        directory
    }

    pub(crate) fn insert_pack(&mut self, pack_id: PackId, cell_ids: Vec<CellId>) {
        if let Some(previous_cells) = self.cells_by_pack.remove(&pack_id) {
            for cell_id in previous_cells {
                self.pack_by_cell.remove(&cell_id);
            }
        }

        let mut unique_cells = Vec::with_capacity(cell_ids.len());
        let mut seen = HashSet::with_capacity(cell_ids.len());
        for cell_id in cell_ids {
            if seen.insert(cell_id) {
                unique_cells.push(cell_id);
                self.pack_by_cell.insert(cell_id, pack_id);
            }
        }

        unique_cells.sort_unstable();
        self.cells_by_pack.insert(pack_id, unique_cells);
    }

    pub(crate) fn pack_for_cell(&self, cell_id: CellId) -> Option<PackId> {
        self.pack_by_cell.get(&cell_id).copied()
    }

    pub(crate) fn cell_ids(&self, pack_id: PackId) -> Option<&[CellId]> {
        self.cells_by_pack.get(&pack_id).map(Vec::as_slice)
    }

    pub(crate) fn len(&self) -> usize {
        self.cells_by_pack.len()
    }

    /// Remove a pack and all its cell mappings from the directory.
    ///
    /// Returns the cell IDs that belonged to the pack (empty if not found).
    pub(crate) fn remove_pack(&mut self, pack_id: PackId) -> Vec<CellId> {
        if let Some(cell_ids) = self.cells_by_pack.remove(&pack_id) {
            for &cell_id in &cell_ids {
                self.pack_by_cell.remove(&cell_id);
            }
            cell_ids
        } else {
            Vec::new()
        }
    }

    pub(crate) fn pack_ids(&self) -> impl Iterator<Item = &PackId> {
        self.cells_by_pack.keys()
    }

    pub(crate) fn all_cell_ids(&self) -> impl Iterator<Item = CellId> + '_ {
        self.pack_by_cell.keys().copied()
    }

    pub(crate) fn next_pack_id(&self) -> PackId {
        self.cells_by_pack
            .keys()
            .copied()
            .max()
            .map_or(FIRST_ASSIGNED_PACK_ID, |pack_id| pack_id + 1)
    }

    pub(crate) fn add_cell(&mut self, pack_id: PackId, cell_id: CellId) {
        self.cells_by_pack.entry(pack_id).or_default().push(cell_id);
        self.pack_by_cell.insert(cell_id, pack_id);
    }

    fn sort_pack_cells(&mut self) {
        for cell_ids in self.cells_by_pack.values_mut() {
            cell_ids.sort_unstable();
            cell_ids.dedup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIRST_TEST_PACK_ID: PackId = 7;
    const SECOND_TEST_PACK_ID: PackId = 11;
    const FIRST_PACK_CELLS: [CellId; 3] = [30, 10, 20];
    const FIRST_PACK_CELLS_SORTED: [CellId; 3] = [10, 20, 30];
    const SECOND_PACK_CELLS: [CellId; 2] = [40, 50];
    const FIRST_PACK_LOOKUP_CELL: CellId = 10;
    const SECOND_PACK_LOOKUP_CELL: CellId = 50;
    const MISSING_PACK_LOOKUP_CELL: CellId = 99;

    #[test]
    fn test_pack_directory_finds_pack_by_cell_id() {
        let mut directory = PackDirectory::new();
        directory.insert_pack(FIRST_TEST_PACK_ID, FIRST_PACK_CELLS.to_vec());
        directory.insert_pack(SECOND_TEST_PACK_ID, SECOND_PACK_CELLS.to_vec());

        assert_eq!(directory.pack_for_cell(FIRST_PACK_LOOKUP_CELL), Some(FIRST_TEST_PACK_ID));
        assert_eq!(directory.pack_for_cell(SECOND_PACK_LOOKUP_CELL), Some(SECOND_TEST_PACK_ID));
        assert_eq!(directory.pack_for_cell(MISSING_PACK_LOOKUP_CELL), None);
    }

    #[test]
    fn test_pack_directory_preserves_layer_cell_membership() {
        let mut directory = PackDirectory::new();

        directory.insert_pack(FIRST_TEST_PACK_ID, FIRST_PACK_CELLS.to_vec());

        assert_eq!(
            directory.cell_ids(FIRST_TEST_PACK_ID),
            Some(FIRST_PACK_CELLS_SORTED.as_slice())
        );
    }
}
