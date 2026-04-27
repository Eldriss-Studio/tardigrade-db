//! Block pool — the Repository abstraction over segmented storage.
//!
//! Provides `append` and `get` operations over a collection of segment files,
//! with an in-memory index mapping `CellId` → (`segment_id`, `byte_offset`).
//! The index is rebuilt from segment files on open (recovery).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use tdb_core::CellId;
use tdb_core::error::{Result, TardigradeError};
use tdb_core::memory_cell::MemoryCell;

use crate::segment::{RecordLocation, Segment, list_segments, scan_segment};

/// Default segment size threshold: 256 MB.
const DEFAULT_SEGMENT_SIZE: u64 = 256 * 1024 * 1024;

/// Repository over segmented, append-only storage for memory cells.
///
/// The index is held in memory (`BTreeMap`) and rebuilt from segment files on open.
/// Segments are append-only; when the active segment exceeds the size threshold,
/// a new segment is created.
#[derive(Debug)]
pub struct BlockPool {
    dir: PathBuf,
    segments: Vec<Segment>,
    index: BTreeMap<CellId, RecordLocation>,
    segment_size_threshold: u64,
}

impl BlockPool {
    /// Open or create a block pool at the given directory path.
    /// Rebuilds the in-memory index by scanning existing segments.
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with_segment_size(dir, DEFAULT_SEGMENT_SIZE)
    }

    /// Open with a custom segment size threshold (useful for testing).
    pub fn open_with_segment_size(dir: &Path, segment_size_threshold: u64) -> Result<Self> {
        std::fs::create_dir_all(dir)?;

        let segment_ids = list_segments(dir)?;
        let mut segments = Vec::new();
        let mut index = BTreeMap::new();

        for &seg_id in &segment_ids {
            let segment = Segment::open(dir, seg_id)?;
            let entries = scan_segment(dir, seg_id)?;
            for (cell_id, byte_offset) in entries {
                index.insert(cell_id, RecordLocation { segment_id: seg_id, byte_offset });
            }
            segments.push(segment);
        }

        // If no segments exist, create the first one.
        if segments.is_empty() {
            segments.push(Segment::create(dir, 0)?);
        }

        Ok(Self { dir: dir.to_path_buf(), segments, index, segment_size_threshold })
    }

    /// Re-scan segment files on disk and merge any new entries into the
    /// in-memory index. Picks up cells written by another `BlockPool`
    /// handle at the same path.
    ///
    /// Idempotent: re-scanning a segment that hasn't grown leaves the
    /// index unchanged. New segment files (if another writer rolled over)
    /// are opened and added to `self.segments`. Existing segment file
    /// handles are not re-opened.
    pub fn refresh_index(&mut self) -> Result<()> {
        let segment_ids = list_segments(&self.dir)?;

        // Pick up any newly-created segments (other writer rolled over).
        let known: std::collections::HashSet<u32> = self.segments.iter().map(Segment::id).collect();
        for &seg_id in &segment_ids {
            if !known.contains(&seg_id) {
                self.segments.push(Segment::open(&self.dir, seg_id)?);
            }
        }

        // Re-scan every segment (cheap — header reads only) and merge into
        // the index. `scan_segment` is idempotent on append-only files.
        for &seg_id in &segment_ids {
            let entries = scan_segment(&self.dir, seg_id)?;
            for (cell_id, byte_offset) in entries {
                self.index
                    .entry(cell_id)
                    .or_insert(RecordLocation { segment_id: seg_id, byte_offset });
            }
        }

        Ok(())
    }

    /// Append a memory cell to the pool. Returns the cell ID on success.
    pub fn append(&mut self, cell: &MemoryCell) -> Result<CellId> {
        self.ensure_active_segment_has_capacity()?;

        let active = self.active_segment_mut();
        let seg_id = active.id();
        let byte_offset = active.append(cell)?;

        self.index.insert(cell.id, RecordLocation { segment_id: seg_id, byte_offset });

        Ok(cell.id)
    }

    /// Append multiple cells in a single write + single fsync (Write-Behind Buffer).
    ///
    /// All cells are written to the active segment and durably committed with
    /// one `sync_data()` call. Returns the cell IDs.
    pub fn append_batch(&mut self, cells: &[MemoryCell]) -> Result<Vec<CellId>> {
        if cells.is_empty() {
            return Ok(Vec::new());
        }

        self.ensure_active_segment_has_capacity()?;

        let active = self.active_segment_mut();
        let seg_id = active.id();
        let offsets = active.append_batch(cells)?;

        let mut ids = Vec::with_capacity(cells.len());
        for (cell, byte_offset) in cells.iter().zip(offsets) {
            self.index.insert(cell.id, RecordLocation { segment_id: seg_id, byte_offset });
            ids.push(cell.id);
        }

        Ok(ids)
    }

    /// Retrieve a memory cell by its ID.
    pub fn get(&self, cell_id: CellId) -> Result<MemoryCell> {
        let loc = self.index.get(&cell_id).ok_or(TardigradeError::CellNotFound(cell_id))?;

        let segment = self
            .segments
            .get(loc.segment_id as usize)
            .ok_or(TardigradeError::CellNotFound(cell_id))?;

        Ok(segment.read_at(loc.byte_offset)?)
    }

    /// Number of segment files in this pool.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Number of cells tracked in the index.
    pub fn cell_count(&self) -> usize {
        self.index.len()
    }

    /// Iterate over all persisted cell IDs (sorted, from the in-memory index).
    /// Used by `Engine::open()` to rebuild derived state from disk (Memento pattern).
    pub fn iter_cell_ids(&self) -> impl Iterator<Item = CellId> + '_ {
        self.index.keys().copied()
    }

    /// If the active segment exceeds the threshold, create a new one.
    fn ensure_active_segment_has_capacity(&mut self) -> Result<()> {
        let needs_rollover =
            self.segments.last().is_some_and(|s| s.size() >= self.segment_size_threshold);

        if needs_rollover {
            let new_id = self.segments.last().unwrap().id() + 1;
            let new_segment = Segment::create(&self.dir, new_id)?;
            self.segments.push(new_segment);
        }
        Ok(())
    }

    fn active_segment_mut(&mut self) -> &mut Segment {
        self.segments.last_mut().expect("pool always has at least one segment")
    }
}
