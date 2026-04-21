//! Write-Ahead Log for crash-recoverable graph mutations.
//!
//! Append-only log file. Every Trace graph mutation is logged before execution.
//! Checkpoint = flush all in-memory state + truncate WAL.
//!
//! ## Record format (binary, little-endian)
//!
//! ```text
//! [entry_type: u8]    — 1 = AddEdge
//! [src: u64]          — source cell ID
//! [dst: u64]          — destination cell ID
//! [edge_type: u8]     — edge type enum
//! [timestamp: u64]    — event timestamp
//! ```

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// A single WAL entry representing a graph mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalEntry {
    AddEdge { src: u64, dst: u64, edge_type: u8, timestamp: u64 },
}

/// Size of a serialized `AddEdge` entry: type(1) + src(8) + dst(8) + `edge_type(1)` + timestamp(8) = 26
const ADD_EDGE_RECORD_SIZE: usize = 26;

const ENTRY_TYPE_ADD_EDGE: u8 = 1;

/// Write-Ahead Log for Trace graph mutations.
#[derive(Debug)]
pub struct Wal {
    path: PathBuf,
}

impl Wal {
    /// Open or create a WAL at the given directory.
    pub fn open(dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("trace.wal");
        // Touch the file if it doesn't exist.
        OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self { path })
    }

    /// Append an entry to the WAL. Fsyncs for durability.
    pub fn append(&mut self, entry: &WalEntry) -> io::Result<()> {
        let file = OpenOptions::new().append(true).open(&self.path)?;
        let mut w = BufWriter::new(file);

        match entry {
            WalEntry::AddEdge { src, dst, edge_type, timestamp } => {
                w.write_all(&[ENTRY_TYPE_ADD_EDGE])?;
                w.write_all(&src.to_le_bytes())?;
                w.write_all(&dst.to_le_bytes())?;
                w.write_all(&[*edge_type])?;
                w.write_all(&timestamp.to_le_bytes())?;
            }
        }

        let inner = w.into_inner().map_err(std::io::IntoInnerError::into_error)?;
        inner.sync_data()?;
        Ok(())
    }

    /// Replay all entries from the WAL (for recovery).
    pub fn replay(&self) -> io::Result<Vec<WalEntry>> {
        let mut file = File::open(&self.path)?;
        let file_len = file.metadata()?.len();

        if file_len == 0 {
            return Ok(Vec::new());
        }

        file.seek(SeekFrom::Start(0))?;
        let mut entries = Vec::new();
        let mut pos = 0u64;

        while pos + ADD_EDGE_RECORD_SIZE as u64 <= file_len {
            let mut type_buf = [0u8; 1];
            if file.read_exact(&mut type_buf).is_err() {
                break;
            }

            match type_buf[0] {
                ENTRY_TYPE_ADD_EDGE => {
                    let mut buf = [0u8; 8];

                    // Use break-on-error for all reads — lenient during crash recovery.
                    // A partial record at EOF is discarded rather than aborting replay.
                    if file.read_exact(&mut buf).is_err() {
                        break;
                    }
                    let src = u64::from_le_bytes(buf);

                    if file.read_exact(&mut buf).is_err() {
                        break;
                    }
                    let dst = u64::from_le_bytes(buf);

                    let mut et_buf = [0u8; 1];
                    if file.read_exact(&mut et_buf).is_err() {
                        break;
                    }
                    let edge_type = et_buf[0];

                    if file.read_exact(&mut buf).is_err() {
                        break;
                    }
                    let timestamp = u64::from_le_bytes(buf);

                    entries.push(WalEntry::AddEdge { src, dst, edge_type, timestamp });
                }
                _ => break, // Unknown entry type — stop replay (possible corruption)
            }

            pos += ADD_EDGE_RECORD_SIZE as u64;
        }

        Ok(entries)
    }

    /// Checkpoint: truncate the WAL (all mutations have been applied to durable state).
    pub fn checkpoint(&mut self) -> io::Result<()> {
        let file = OpenOptions::new().write(true).truncate(true).open(&self.path)?;
        file.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_and_replay() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(dir.path()).unwrap();

        wal.append(&WalEntry::AddEdge { src: 1, dst: 2, edge_type: 0, timestamp: 1000 }).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], WalEntry::AddEdge { src: 1, dst: 2, edge_type: 0, timestamp: 1000 });
    }

    #[test]
    fn test_empty_replay() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path()).unwrap();
        let entries = wal.replay().unwrap();
        assert!(entries.is_empty());
    }
}
