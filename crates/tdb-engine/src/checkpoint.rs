//! Labeled checkpoint repository.
//!
//! Pattern: **Repository**. Wraps the snapshot/restore API
//! with a labelled, sequence-numbered slot layout so consumers
//! can think in terms of `"autosave"` / `"chapter-3-end"` instead
//! of bespoke tarball paths.
//!
//! ## On-disk layout
//!
//! ```text
//! <root>/
//!   <label>/
//!     0001.tar      ← first checkpoint with this label
//!     0002.tar      ← second checkpoint with this label
//!     ...
//!   <other-label>/
//!     0001.tar
//! ```
//!
//! Sequence numbers are zero-padded to four digits so filesystem
//! lexicographic order matches numeric order — useful when an
//! external tool (CLI, file browser) lists the directory.
//!
//! ## Reliability contract
//!
//! Each save delegates to [`crate::engine::Engine::snapshot`],
//! which itself calls `flush()` first — the checkpoint therefore
//! captures durable state at flush completion. Restore is
//! identical to [`crate::engine::Engine::restore_from`] of the
//! same tar.

use std::path::{Path, PathBuf};

use tdb_core::error::{Result, TardigradeError};

use crate::engine::Engine;
use crate::snapshot::SnapshotManifest;

/// File-name format width for sequence numbers — four digits gives
/// space for ten thousand checkpoints per label before the
/// padding overflows (after which the format still sorts
/// correctly numerically, just not lexicographically).
const SEQ_PAD_WIDTH: usize = 4;

/// Extension used for the tarball each checkpoint produces.
const CHECKPOINT_EXTENSION: &str = "tar";

/// Sentinel "no entries yet" sequence — the first save under a
/// label becomes `LAST_SEQ_BEFORE_FIRST + 1 == 1`.
const LAST_SEQ_BEFORE_FIRST: u32 = 0;

/// A single saved checkpoint produced by
/// [`CheckpointRepository::save_from`].
#[derive(Debug, Clone)]
pub struct CheckpointEntry {
    /// Human-readable label this checkpoint belongs to.
    pub label: String,
    /// Monotonic sequence within the label — first save is `1`.
    pub seq: u32,
    /// Absolute path of the tarball on disk.
    pub path: PathBuf,
    /// Manifest as returned by the underlying snapshot.
    pub manifest: SnapshotManifest,
}

/// Repository pattern: durable, label-scoped checkpoint storage.
///
/// Construct once with a root directory (created on demand on
/// first save). Each `save_from` call yields a fresh
/// [`CheckpointEntry`] with a monotonically increasing `seq` for
/// its label. Listing and restoring are pure filesystem walks +
/// manifest parsing — no in-memory state to keep coherent.
#[derive(Debug, Clone)]
pub struct CheckpointRepository {
    root: PathBuf,
}

impl CheckpointRepository {
    /// Build a repository rooted at `root`. The directory does not
    /// have to exist yet — it's created lazily on the first save.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Root directory this repository writes into.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Save the engine's current state into `<root>/<label>/<seq>.tar`.
    ///
    /// The next sequence number is computed by listing existing
    /// `<seq>.tar` files in the label's directory; the highest
    /// observed value + 1 is used. Empty label directory ⇒ seq 1.
    pub fn save_from(&self, engine: &mut Engine, label: &str) -> Result<CheckpointEntry> {
        let label_dir = self.label_dir(label);
        std::fs::create_dir_all(&label_dir).map_err(|e| TardigradeError::Io { source: e })?;

        let next_seq = self.next_seq_for(label)?;
        let path = label_dir.join(format_seq_filename(next_seq));
        let manifest = engine.snapshot(&path)?;
        Ok(CheckpointEntry { label: label.to_string(), seq: next_seq, path, manifest })
    }

    /// List all checkpoints, optionally filtered by label.
    ///
    /// Results are sorted by `(label, seq)` ascending so consumers
    /// can pattern-match on the head/last for the latest under a
    /// specific label without re-sorting.
    pub fn list(&self, label: Option<&str>) -> Result<Vec<CheckpointEntry>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }

        let mut entries = Vec::new();
        for label_dir in self.label_dirs(label)? {
            let label_name =
                label_dir.file_name().and_then(|s| s.to_str()).map(str::to_string).ok_or_else(
                    || {
                        TardigradeError::SnapshotIntegrity(format!(
                            "non-UTF8 checkpoint label directory: {}",
                            label_dir.display()
                        ))
                    },
                )?;

            for (seq, path) in Self::numbered_files(&label_dir)? {
                let manifest = crate::snapshot::read_manifest_only(&path)?;
                entries.push(CheckpointEntry { label: label_name.clone(), seq, path, manifest });
            }
        }

        entries.sort_by(|a, b| a.label.cmp(&b.label).then(a.seq.cmp(&b.seq)));
        Ok(entries)
    }

    /// Latest checkpoint for `label`, or `None` if none exist.
    pub fn latest(&self, label: &str) -> Result<Option<CheckpointEntry>> {
        Ok(self.list(Some(label))?.pop())
    }

    /// Restore the latest checkpoint for `label` into `target_dir`.
    ///
    /// Errors:
    /// - `TardigradeError::SnapshotIntegrity` if no checkpoint
    ///   matches `label`.
    /// - Whatever [`Engine::restore_from`] returns for tar/manifest
    ///   corruption.
    pub fn restore_latest(&self, label: &str, target_dir: &Path) -> Result<Engine> {
        let entry = self.latest(label)?.ok_or_else(|| {
            TardigradeError::SnapshotIntegrity(format!("no checkpoint found for label {label:?}"))
        })?;
        Engine::restore_from(&entry.path, target_dir)
    }

    // ── Internals ────────────────────────────────────────────────

    fn label_dir(&self, label: &str) -> PathBuf {
        self.root.join(label)
    }

    fn label_dirs(&self, only: Option<&str>) -> Result<Vec<PathBuf>> {
        let mut dirs = Vec::new();
        let read = std::fs::read_dir(&self.root).map_err(|e| TardigradeError::Io { source: e })?;
        for entry in read {
            let entry = entry.map_err(|e| TardigradeError::Io { source: e })?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if let Some(filter) = only
                && path.file_name().and_then(|s| s.to_str()) != Some(filter)
            {
                continue;
            }
            dirs.push(path);
        }
        dirs.sort();
        Ok(dirs)
    }

    fn next_seq_for(&self, label: &str) -> Result<u32> {
        let label_dir = self.label_dir(label);
        if !label_dir.exists() {
            return Ok(LAST_SEQ_BEFORE_FIRST + 1);
        }
        let max = Self::numbered_files(&label_dir)?
            .into_iter()
            .map(|(seq, _)| seq)
            .max()
            .unwrap_or(LAST_SEQ_BEFORE_FIRST);
        Ok(max + 1)
    }

    /// Return `(seq, path)` pairs for every `<NNNN>.tar` in `dir`,
    /// skipping files that don't match the format.
    fn numbered_files(dir: &Path) -> Result<Vec<(u32, PathBuf)>> {
        let mut out = Vec::new();
        let read = std::fs::read_dir(dir).map_err(|e| TardigradeError::Io { source: e })?;
        for entry in read {
            let entry = entry.map_err(|e| TardigradeError::Io { source: e })?;
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
            let ext = path.extension().and_then(|s| s.to_str());
            if ext != Some(CHECKPOINT_EXTENSION) {
                continue;
            }
            if let Ok(seq) = stem.parse::<u32>() {
                out.push((seq, path));
            }
        }
        out.sort_by_key(|(seq, _)| *seq);
        Ok(out)
    }
}

fn format_seq_filename(seq: u32) -> String {
    let padded = format!("{seq:0>SEQ_PAD_WIDTH$}");
    format!("{padded}.{CHECKPOINT_EXTENSION}")
}
