//! KV pack directory — Value Object for pack membership and reverse lookup.
//!
//! `Engine::mem_read_pack` starts from retrieval results, which are cell IDs.
//! The engine then needs to reconstruct complete packs. Keeping
//! `pack_id -> cell_ids`, `cell_id -> pack_id`, and `pack_id -> owner` in one
//! private value object makes those mappings explicit and keeps every
//! lookup O(1).
//!
//! The owner index in particular lets `Engine::list_packs` answer metadata
//! queries in pure in-memory iteration — without decompressing each pack's
//! retrieval cell from the block pool just to read its owner field.

use std::collections::{HashMap, HashSet};

use tdb_core::kv_pack::PackId;
use tdb_core::{CellId, OwnerId};

const FIRST_ASSIGNED_PACK_ID: PackId = 1;

#[derive(Debug, Default)]
pub(crate) struct PackDirectory {
    cells_by_pack: HashMap<PackId, Vec<CellId>>,
    pack_by_cell: HashMap<CellId, PackId>,
    owners_by_pack: HashMap<PackId, OwnerId>,
}

impl PackDirectory {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Build a directory from a flat iterator of `(pack_id, cell_id, owner)`.
    ///
    /// Used during `Engine::open` and `Engine::refresh` when the segment
    /// trail is the authoritative source. The owner is captured here so the
    /// rebuilt directory can answer owner queries without re-reading cells.
    pub(crate) fn from_cells<I>(cells: I) -> Self
    where
        I: IntoIterator<Item = (PackId, CellId, OwnerId)>,
    {
        let mut directory = Self::new();
        for (pack_id, cell_id, owner) in cells {
            directory.add_cell(pack_id, cell_id);
            directory.owners_by_pack.insert(pack_id, owner);
        }
        directory.sort_pack_cells();
        directory
    }

    /// Insert (or replace) the cell set for a pack, recording its owner.
    pub(crate) fn insert_pack(&mut self, pack_id: PackId, cell_ids: Vec<CellId>, owner: OwnerId) {
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
        self.owners_by_pack.insert(pack_id, owner);
    }

    pub(crate) fn pack_for_cell(&self, cell_id: CellId) -> Option<PackId> {
        self.pack_by_cell.get(&cell_id).copied()
    }

    pub(crate) fn cell_ids(&self, pack_id: PackId) -> Option<&[CellId]> {
        self.cells_by_pack.get(&pack_id).map(Vec::as_slice)
    }

    /// Owner of a pack, in O(1). Returns `None` for an unknown pack.
    pub(crate) fn owner_for_pack(&self, pack_id: PackId) -> Option<OwnerId> {
        self.owners_by_pack.get(&pack_id).copied()
    }

    pub(crate) fn len(&self) -> usize {
        self.cells_by_pack.len()
    }

    /// Remove a pack, its cell mappings, and its owner record.
    ///
    /// Returns the cell IDs that belonged to the pack (empty if not found).
    pub(crate) fn remove_pack(&mut self, pack_id: PackId) -> Vec<CellId> {
        self.owners_by_pack.remove(&pack_id);
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
    const FIRST_OWNER: OwnerId = 100;
    const SECOND_OWNER: OwnerId = 200;
    const FIRST_PACK_CELLS: [CellId; 3] = [30, 10, 20];
    const FIRST_PACK_CELLS_SORTED: [CellId; 3] = [10, 20, 30];
    const SECOND_PACK_CELLS: [CellId; 2] = [40, 50];
    const FIRST_PACK_LOOKUP_CELL: CellId = 10;
    const SECOND_PACK_LOOKUP_CELL: CellId = 50;
    const MISSING_PACK_LOOKUP_CELL: CellId = 99;

    #[test]
    fn test_pack_directory_finds_pack_by_cell_id() {
        let mut directory = PackDirectory::new();
        directory.insert_pack(FIRST_TEST_PACK_ID, FIRST_PACK_CELLS.to_vec(), FIRST_OWNER);
        directory.insert_pack(SECOND_TEST_PACK_ID, SECOND_PACK_CELLS.to_vec(), SECOND_OWNER);

        assert_eq!(directory.pack_for_cell(FIRST_PACK_LOOKUP_CELL), Some(FIRST_TEST_PACK_ID));
        assert_eq!(directory.pack_for_cell(SECOND_PACK_LOOKUP_CELL), Some(SECOND_TEST_PACK_ID));
        assert_eq!(directory.pack_for_cell(MISSING_PACK_LOOKUP_CELL), None);
    }

    #[test]
    fn test_pack_directory_preserves_layer_cell_membership() {
        let mut directory = PackDirectory::new();

        directory.insert_pack(FIRST_TEST_PACK_ID, FIRST_PACK_CELLS.to_vec(), FIRST_OWNER);

        assert_eq!(
            directory.cell_ids(FIRST_TEST_PACK_ID),
            Some(FIRST_PACK_CELLS_SORTED.as_slice())
        );
    }

    #[test]
    fn test_pack_directory_records_owner() {
        let mut directory = PackDirectory::new();
        directory.insert_pack(FIRST_TEST_PACK_ID, FIRST_PACK_CELLS.to_vec(), FIRST_OWNER);
        directory.insert_pack(SECOND_TEST_PACK_ID, SECOND_PACK_CELLS.to_vec(), SECOND_OWNER);

        assert_eq!(directory.owner_for_pack(FIRST_TEST_PACK_ID), Some(FIRST_OWNER));
        assert_eq!(directory.owner_for_pack(SECOND_TEST_PACK_ID), Some(SECOND_OWNER));
        assert_eq!(directory.owner_for_pack(999), None);
    }

    #[test]
    fn test_remove_pack_drops_owner_record() {
        let mut directory = PackDirectory::new();
        directory.insert_pack(FIRST_TEST_PACK_ID, FIRST_PACK_CELLS.to_vec(), FIRST_OWNER);

        directory.remove_pack(FIRST_TEST_PACK_ID);

        assert_eq!(directory.owner_for_pack(FIRST_TEST_PACK_ID), None);
    }
}
