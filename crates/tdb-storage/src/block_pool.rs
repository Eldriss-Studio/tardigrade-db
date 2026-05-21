//! Block pool — the Repository abstraction over segmented storage.
//!
//! Provides `append` and `get` operations over a collection of segment files,
//! with an in-memory index mapping `CellId` → (`segment_id`, `byte_offset`).
//! The index is rebuilt from segment files on open (recovery).
//!
//! # Concurrency
//!
//! Every public method takes `&self`. The segment list and the cell index
//! both live behind [`arc_swap::ArcSwap`], so readers ([`BlockPool::get`],
//! [`BlockPool::cell_count`], [`BlockPool::iter_cell_ids`]) take cheap
//! atomic snapshots that never block writers. Writers ([`BlockPool::append`],
//! [`BlockPool::append_batch`], [`BlockPool::compact`], [`BlockPool::refresh_index`])
//! serialize behind a single internal `Mutex<()>` that protects the file
//! I/O *and* the index/segments swap so two writers can't interleave
//! records on disk or race the snapshot replacement. Readers holding an
//! older snapshot keep returning the state they saw at acquisition time
//! even while a writer installs a newer one — the snapshot is what makes
//! pack hydration able to drop the engine mutex.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
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
///
/// See the module docs for the concurrency contract — every method is
/// `&self`-safe; writers serialize behind an internal mutex while readers
/// take lock-free snapshots.
#[derive(Debug)]
pub struct BlockPool {
    dir: PathBuf,
    segments: ArcSwap<Vec<Segment>>,
    index: ArcSwap<BTreeMap<CellId, RecordLocation>>,
    segment_size_threshold: u64,
    /// Serializes file appends and snapshot installation so concurrent
    /// writers can't interleave records or race the segments/index swap.
    write_lock: Mutex<()>,
}

impl BlockPool {
    /// Open or create a block pool at the given directory path.
    /// Rebuilds the in-memory index by scanning existing segments.
    ///
    /// # Errors
    /// Returns [`TardigradeError::Io`] if the directory cannot be created,
    /// segment files cannot be opened, or an existing segment's header
    /// fails the magic/version check during recovery.
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with_segment_size(dir, DEFAULT_SEGMENT_SIZE)
    }

    /// Open with a custom segment size threshold (useful for testing).
    ///
    /// # Errors
    /// Same conditions as [`Self::open`].
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

        Ok(Self {
            dir: dir.to_path_buf(),
            segments: ArcSwap::from_pointee(segments),
            index: ArcSwap::from_pointee(index),
            segment_size_threshold,
            write_lock: Mutex::new(()),
        })
    }

    /// Re-scan segment files on disk and merge any new entries into the
    /// in-memory index. Picks up cells written by another `BlockPool`
    /// handle at the same path.
    ///
    /// Idempotent: re-scanning a segment that hasn't grown leaves the
    /// index unchanged. New segment files (if another writer rolled over)
    /// are opened and added to `self.segments`. Existing segment file
    /// handles are not re-opened.
    ///
    /// # Errors
    /// Returns [`TardigradeError::Io`] if directory enumeration fails or
    /// a newly-discovered segment cannot be opened or scanned.
    ///
    /// # Panics
    /// Panics if the internal write mutex is poisoned — only possible if
    /// a previous writer panicked while holding it.
    pub fn refresh_index(&self) -> Result<()> {
        let segment_ids = list_segments(&self.dir)?;
        let _guard = self.write_lock.lock().expect("block-pool write_lock poisoned");

        // Stage a new segments list and a new index built atop the current
        // snapshots; install them atomically at the end so readers never
        // see a partially-applied refresh.
        let mut new_segments: Vec<Segment> = (**self.segments.load()).clone();
        let known: std::collections::HashSet<u32> = new_segments.iter().map(Segment::id).collect();
        for &seg_id in &segment_ids {
            if !known.contains(&seg_id) {
                new_segments.push(Segment::open(&self.dir, seg_id)?);
            }
        }

        let mut new_index: BTreeMap<CellId, RecordLocation> = (**self.index.load()).clone();
        for &seg_id in &segment_ids {
            let entries = scan_segment(&self.dir, seg_id)?;
            for (cell_id, byte_offset) in entries {
                new_index
                    .entry(cell_id)
                    .or_insert(RecordLocation { segment_id: seg_id, byte_offset });
            }
        }

        self.segments.store(Arc::new(new_segments));
        self.index.store(Arc::new(new_index));

        Ok(())
    }

    /// Append a memory cell to the pool. Returns the cell ID on success.
    ///
    /// # Errors
    /// Returns [`TardigradeError::Io`] on segment rollover or write failure,
    /// or [`TardigradeError::SegmentFull`] if the active segment cannot be located.
    ///
    /// # Panics
    /// Panics if the internal write mutex is poisoned — see [`Self::refresh_index`].
    pub fn append(&self, cell: &MemoryCell) -> Result<CellId> {
        let _guard = self.write_lock.lock().expect("block-pool write_lock poisoned");
        let mut segs = (**self.segments.load()).clone();
        Self::ensure_active_segment_has_capacity(
            &mut segs,
            &self.dir,
            self.segment_size_threshold,
        )?;

        let active = segs
            .last_mut()
            .ok_or_else(|| TardigradeError::SegmentFull { path: self.dir.display().to_string() })?;
        let seg_id = active.id();
        let byte_offset = active.append(cell)?;

        let mut idx = (**self.index.load()).clone();
        idx.insert(cell.id, RecordLocation { segment_id: seg_id, byte_offset });

        self.segments.store(Arc::new(segs));
        self.index.store(Arc::new(idx));

        Ok(cell.id)
    }

    /// Append multiple cells in a single write + single fsync (Write-Behind Buffer).
    ///
    /// All cells are written to the active segment and durably committed with
    /// one `sync_data()` call. Returns the cell IDs.
    ///
    /// # Errors
    /// Same as [`Self::append`]; on partial-batch failure the segment may be
    /// rolled over but the index is not updated with the failed slice.
    ///
    /// # Panics
    /// Panics if the internal write mutex is poisoned — see [`Self::refresh_index`].
    pub fn append_batch(&self, cells: &[MemoryCell]) -> Result<Vec<CellId>> {
        if cells.is_empty() {
            return Ok(Vec::new());
        }

        let _guard = self.write_lock.lock().expect("block-pool write_lock poisoned");
        let mut segs = (**self.segments.load()).clone();
        Self::ensure_active_segment_has_capacity(
            &mut segs,
            &self.dir,
            self.segment_size_threshold,
        )?;

        let active = segs
            .last_mut()
            .ok_or_else(|| TardigradeError::SegmentFull { path: self.dir.display().to_string() })?;
        let seg_id = active.id();
        let offsets = active.append_batch(cells)?;

        let mut idx = (**self.index.load()).clone();
        let mut ids = Vec::with_capacity(cells.len());
        for (cell, byte_offset) in cells.iter().zip(offsets) {
            idx.insert(cell.id, RecordLocation { segment_id: seg_id, byte_offset });
            ids.push(cell.id);
        }

        self.segments.store(Arc::new(segs));
        self.index.store(Arc::new(idx));

        Ok(ids)
    }

    /// Retrieve a memory cell by its ID.
    ///
    /// Takes atomic snapshots of the index and segments list; a concurrent
    /// writer installing a newer state doesn't affect the read in progress.
    ///
    /// # Errors
    /// Returns [`TardigradeError::CellNotFound`] if the cell is unknown
    /// to the in-memory index or its segment is missing, or
    /// [`TardigradeError::Io`] if the underlying segment read fails.
    pub fn get(&self, cell_id: CellId) -> Result<MemoryCell> {
        let index = self.index.load();
        let loc = index.get(&cell_id).ok_or(TardigradeError::CellNotFound(cell_id))?;
        let segments = self.segments.load();
        let segment = segments
            .iter()
            .find(|s| s.id() == loc.segment_id)
            .ok_or(TardigradeError::CellNotFound(cell_id))?;

        Ok(segment.read_at(loc.byte_offset)?)
    }

    /// Number of segment files in this pool.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.load().len()
    }

    /// Number of cells tracked in the index.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.index.load().len()
    }

    /// Total on-disk bytes across every segment file in this pool.
    ///
    /// Used by `EngineStatus::arena_bytes` for footprint reporting. Sums
    /// the active segment plus all sealed segments; does not subtract
    /// space reclaimable by compaction (call `compact()` first if you
    /// want a tight figure).
    #[must_use]
    pub fn arena_bytes(&self) -> u64 {
        self.segments.load().iter().map(Segment::size).sum()
    }

    /// All persisted cell IDs in a freshly-snapshotted owned `Vec`, sorted.
    ///
    /// Used by `Engine::open()` to rebuild derived state from disk (Memento
    /// pattern). Returns owned data because the snapshot's lifetime ends
    /// with the load; an iterator borrowing from the snapshot would dangle.
    #[must_use]
    pub fn iter_cell_ids(&self) -> std::vec::IntoIter<CellId> {
        let snapshot = self.index.load();
        let ids: Vec<CellId> = snapshot.keys().copied().collect();
        ids.into_iter()
    }

    /// If the active segment exceeds the threshold, create a new one.
    ///
    /// Operates on a caller-owned `Vec<Segment>` (the writer's staged
    /// clone) rather than `&mut self` so the rollover composes with the
    /// `CoW` write path.
    fn ensure_active_segment_has_capacity(
        segs: &mut Vec<Segment>,
        dir: &Path,
        segment_size_threshold: u64,
    ) -> Result<()> {
        let needs_rollover = segs.last().is_some_and(|s| s.size() >= segment_size_threshold);

        if needs_rollover {
            let last = segs
                .last()
                .ok_or_else(|| TardigradeError::SegmentFull { path: dir.display().to_string() })?;
            let new_id = last.id() + 1;
            let new_segment = Segment::create(dir, new_id)?;
            segs.push(new_segment);
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
    ///
    /// # Errors
    /// Returns [`TardigradeError::Io`] on segment scan, read, append, or
    /// fsync failure.
    ///
    /// # Panics
    /// Panics if the internal write mutex is poisoned — see [`Self::refresh_index`].
    pub fn compact(&self, live_cell_ids: &HashSet<CellId>) -> Result<CompactionResult> {
        let mut result = CompactionResult::default();

        let _guard = self.write_lock.lock().expect("block-pool write_lock poisoned");
        let mut segs: Vec<Segment> = (**self.segments.load()).clone();

        if segs.len() <= 1 {
            return Ok(result);
        }

        let active_seg_id = segs.last().map_or(0, Segment::id);

        let mut jobs: Vec<CompactJob> = Vec::new();

        for seg in &segs {
            let seg_id = seg.id();
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
                    let cell =
                        seg.read_at(*byte_offset).map_err(|e| TardigradeError::Io { source: e })?;
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

        let mut idx: BTreeMap<CellId, RecordLocation> = (**self.index.load()).clone();

        for job in &jobs {
            if !job.cells.is_empty() {
                Self::ensure_active_segment_has_capacity(
                    &mut segs,
                    &self.dir,
                    self.segment_size_threshold,
                )?;
                let active = segs.last_mut().ok_or_else(|| TardigradeError::SegmentFull {
                    path: self.dir.display().to_string(),
                })?;
                let new_seg_id = active.id();
                let offsets = active
                    .append_batch(&job.cells)
                    .map_err(|e| TardigradeError::Io { source: e })?;

                for (cell, offset) in job.cells.iter().zip(offsets) {
                    idx.insert(
                        cell.id,
                        RecordLocation { segment_id: new_seg_id, byte_offset: offset },
                    );
                }
                result.cells_moved += job.cells.len();
            }

            result.bytes_reclaimed += job.file_size;
        }

        let compacted_ids: HashSet<u32> = jobs.iter().map(|j| j.seg_id).collect();
        idx.retain(|_, loc| !compacted_ids.contains(&loc.segment_id));

        for seg_id in &compacted_ids {
            let path = segment_path(&self.dir, *seg_id);
            std::fs::remove_file(&path).map_err(|e| TardigradeError::Io { source: e })?;
        }

        segs.retain(|s| !compacted_ids.contains(&s.id()));
        result.segments_compacted = compacted_ids.len();

        self.segments.store(Arc::new(segs));
        self.index.store(Arc::new(idx));

        Ok(result)
    }
}
