//! Block pool — the Repository abstraction over segmented storage.
//!
//! Provides `append` and `get` operations over a collection of segment files,
//! with an in-memory index mapping `CellId` → (`segment_id`, `byte_offset`).
//! The index is rebuilt from segment files on open (recovery).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use tdb_core::CellId;
use tdb_core::error::{Result, TardigradeError};
use tdb_core::memory_cell::MemoryCell;

use crate::segment::{RecordLocation, Segment, list_segments, scan_segment, segment_path};

/// Default segment size threshold: 256 MB.
const DEFAULT_SEGMENT_SIZE: u64 = 256 * 1024 * 1024;

/// Segments with a live-cell ratio below this threshold are candidates for compaction.
const COMPACTION_LIVE_RATIO_THRESHOLD: f64 = 0.5;

/// Result of a compaction operation.
#[derive(Debug, Clone, Default)]
pub struct CompactionResult {
    pub segments_compacted: usize,
    pub cells_moved: usize,
    pub bytes_reclaimed: u64,
}

struct CompactJob {
    seg_id: u32,
    cells: Vec<MemoryCell>,
    file_size: u64,
}

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

        let active = self.active_segment_mut()?;
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

        let active = self.active_segment_mut()?;
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
            .iter()
            .find(|s| s.id() == loc.segment_id)
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

    /// Total on-disk bytes across every segment file in this pool.
    ///
    /// Used by `EngineStatus::arena_bytes` for footprint reporting. Sums
    /// the active segment plus all sealed segments; does not subtract
    /// space reclaimable by compaction (call `compact()` first if you
    /// want a tight figure).
    pub fn arena_bytes(&self) -> u64 {
        self.segments.iter().map(Segment::size).sum()
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
            let last = self
                .segments
                .last()
                .ok_or(TardigradeError::SegmentFull { path: self.dir.display().to_string() })?;
            let new_id = last.id() + 1;
            let new_segment = Segment::create(&self.dir, new_id)?;
            self.segments.push(new_segment);
        }
        Ok(())
    }

    /// Compact segments by rewriting live cells and deleting dead ones (Mark-Sweep).
    ///
    /// Non-active segments where the ratio of live cells falls below
    /// `COMPACTION_LIVE_RATIO_THRESHOLD` are rewritten: live cells are
    /// appended to the active segment, then the old segment file is deleted.
    ///
    /// Crash-safe: new cells are fsynced before old segment deletion. If a
    /// crash occurs between write and delete, the next `open()` rebuilds
    /// from all segments — duplicates are harmless (index deduplicates).
    pub fn compact(&mut self, live_cell_ids: &HashSet<CellId>) -> Result<CompactionResult> {
        let mut result = CompactionResult::default();

        if self.segments.len() <= 1 {
            return Ok(result);
        }

        let active_seg_id = self.segments.last().map_or(0, Segment::id);

        let mut jobs: Vec<CompactJob> = Vec::new();

        for seg_idx in 0..self.segments.len() {
            let seg_id = self.segments[seg_idx].id();
            if seg_id == active_seg_id {
                continue;
            }

            let entries =
                scan_segment(&self.dir, seg_id).map_err(|e| TardigradeError::Io { source: e })?;
            if entries.is_empty() {
                continue;
            }

            let live_count = entries.iter().filter(|(cid, _)| live_cell_ids.contains(cid)).count();
            let total = entries.len();
            let live_ratio = live_count as f64 / total as f64;

            if live_ratio >= COMPACTION_LIVE_RATIO_THRESHOLD {
                continue;
            }

            let mut cells = Vec::with_capacity(live_count);
            for (cell_id, byte_offset) in &entries {
                if live_cell_ids.contains(cell_id) {
                    let cell = self.segments[seg_idx]
                        .read_at(*byte_offset)
                        .map_err(|e| TardigradeError::Io { source: e })?;
                    cells.push(cell);
                }
            }

            let seg_path = segment_path(&self.dir, seg_id);
            let file_size = std::fs::metadata(&seg_path).map_or(0, |m| m.len());

            jobs.push(CompactJob { seg_id, cells, file_size });
        }

        if jobs.is_empty() {
            return Ok(result);
        }

        for job in &jobs {
            if !job.cells.is_empty() {
                self.ensure_active_segment_has_capacity()?;
                let active = self.active_segment_mut()?;
                let new_seg_id = active.id();
                let offsets = active
                    .append_batch(&job.cells)
                    .map_err(|e| TardigradeError::Io { source: e })?;

                for (cell, offset) in job.cells.iter().zip(offsets) {
                    self.index.insert(
                        cell.id,
                        RecordLocation { segment_id: new_seg_id, byte_offset: offset },
                    );
                }
                result.cells_moved += job.cells.len();
            }

            result.bytes_reclaimed += job.file_size;
        }

        let compacted_ids: HashSet<u32> = jobs.iter().map(|j| j.seg_id).collect();
        self.index.retain(|_, loc| !compacted_ids.contains(&loc.segment_id));

        for seg_id in &compacted_ids {
            let path = segment_path(&self.dir, *seg_id);
            std::fs::remove_file(&path).map_err(|e| TardigradeError::Io { source: e })?;
        }

        self.segments.retain(|s| !compacted_ids.contains(&s.id()));
        result.segments_compacted = compacted_ids.len();

        Ok(result)
    }

    fn active_segment_mut(&mut self) -> Result<&mut Segment> {
        self.segments
            .last_mut()
            .ok_or(TardigradeError::SegmentFull { path: self.dir.display().to_string() })
    }
}
