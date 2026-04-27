//! Append-only deletion log — durable persistence for pack deletions.
//!
//! Records `PackId` values that have been deleted. On open, the file is
//! replayed into a `HashSet` which serves as a negative filter over the
//! pack directory.
//!
//! # File format
//!
//! ```text
//! ┌────────────────────────────┐
//! │ pack_id: u64 (8 bytes, LE) │
//! ├────────────────────────────┤
//! │ … next record …            │
//! └────────────────────────────┘
//! ```
//!
//! Each record is exactly 8 bytes. Truncated trailing bytes (< 8) are
//! silently discarded.
//!
//! # Durability contract
//!
//! [`DeletionLog::mark_deleted`] appends and fsyncs before returning.
//! A crash after `mark_deleted` returns guarantees the deletion is on disk.
//!
//! # Recovery contract
//!
//! [`DeletionLog::open`] rebuilds the `HashSet` from the file. On engine
//! open, the deleted set is applied as a filter to the pack directory.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use tdb_core::kv_pack::PackId;

const DELETION_LOG_FILENAME: &str = "deletion_log.bin";
const RECORD_SIZE: usize = 8; // u64 pack_id

/// Append-only log of deleted pack IDs.
#[derive(Debug)]
pub struct DeletionLog {
    path: PathBuf,
    deleted: HashSet<PackId>,
}

impl DeletionLog {
    /// Open or create the deletion log in the given directory.
    ///
    /// Scans the file to rebuild the deleted set. Truncated trailing
    /// bytes are silently discarded.
    pub fn open(dir: &Path) -> io::Result<Self> {
        let path = dir.join(DELETION_LOG_FILENAME);
        let deleted = if path.exists() { Self::replay(&path)? } else { HashSet::new() };
        Ok(Self { path, deleted })
    }

    /// Re-read the on-disk file and rebuild the deleted set.
    ///
    /// Used by [`Engine::refresh`] to pick up deletions performed by another
    /// `Engine` handle at the same path. Idempotent: repeated calls with no
    /// on-disk changes leave the set unchanged.
    pub fn refresh(&mut self) -> io::Result<()> {
        self.deleted = if self.path.exists() {
            Self::replay(&self.path)?
        } else {
            HashSet::new()
        };
        Ok(())
    }

    /// Mark a pack as deleted. Appends to the file and fsyncs.
    pub fn mark_deleted(&mut self, pack_id: PackId) -> io::Result<()> {
        let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        file.write_all(&pack_id.to_le_bytes())?;
        file.sync_all()?;
        self.deleted.insert(pack_id);
        Ok(())
    }

    /// Check if a pack has been deleted.
    pub fn is_deleted(&self, pack_id: PackId) -> bool {
        self.deleted.contains(&pack_id)
    }

    /// The complete set of deleted pack IDs.
    pub fn deleted_set(&self) -> &HashSet<PackId> {
        &self.deleted
    }

    /// Replay the deletion log file, rebuilding the deleted set.
    fn replay(path: &Path) -> io::Result<HashSet<PackId>> {
        let file_len = std::fs::metadata(path)?.len() as usize;
        let mut data = vec![0u8; file_len];
        let mut file = File::open(path)?;
        file.read_exact(&mut data)?;

        let mut deleted = HashSet::new();
        let mut cursor = 0;

        while cursor + RECORD_SIZE <= data.len() {
            let pack_id =
                u64::from_le_bytes(data[cursor..cursor + RECORD_SIZE].try_into().unwrap());
            deleted.insert(pack_id);
            cursor += RECORD_SIZE;
        }

        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deletion_log_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let mut log = DeletionLog::open(dir.path()).unwrap();

        log.mark_deleted(1).unwrap();
        log.mark_deleted(5).unwrap();

        assert!(log.is_deleted(1));
        assert!(log.is_deleted(5));
        assert!(!log.is_deleted(3));
    }

    #[test]
    fn test_deletion_log_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();

        {
            let mut log = DeletionLog::open(dir.path()).unwrap();
            log.mark_deleted(42).unwrap();
        }

        let log = DeletionLog::open(dir.path()).unwrap();
        assert!(log.is_deleted(42));
    }

    #[test]
    fn test_deletion_log_opens_fresh_directory() {
        let dir = tempfile::tempdir().unwrap();
        let log = DeletionLog::open(dir.path()).unwrap();
        assert!(log.deleted_set().is_empty());
    }

    #[test]
    fn test_deletion_log_duplicate_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let mut log = DeletionLog::open(dir.path()).unwrap();

        log.mark_deleted(7).unwrap();
        log.mark_deleted(7).unwrap();

        assert!(log.is_deleted(7));
        assert_eq!(log.deleted_set().len(), 1);
    }

    #[test]
    fn test_deletion_log_truncated_trailing_bytes_discarded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DELETION_LOG_FILENAME);

        // Write a valid record.
        {
            let mut log = DeletionLog::open(dir.path()).unwrap();
            log.mark_deleted(1).unwrap();
        }

        // Append 3 garbage bytes (< 8, so incomplete record).
        {
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(&[0xFF, 0xFF, 0xFF]).unwrap();
            file.sync_all().unwrap();
        }

        let log = DeletionLog::open(dir.path()).unwrap();
        assert!(log.is_deleted(1));
        assert_eq!(log.deleted_set().len(), 1);
    }
}
