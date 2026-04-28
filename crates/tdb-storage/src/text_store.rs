//! Append-only text store — durable persistence for KV pack fact text.
//!
//! Maps `PackId → String` with crash-safe append semantics. Text is written
//! alongside tensor data so both survive restarts and crashes together.
//!
//! # File format
//!
//! ```text
//! ┌─────────────────────────────┐
//! │ pack_id: u64 (8 bytes, LE)  │
//! │ text_len: u32 (4 bytes, LE) │
//! │ text: u8[text_len]          │
//! ├─────────────────────────────┤
//! │ … next record …             │
//! └─────────────────────────────┘
//! ```
//!
//! All integers are little-endian. On open, the file is scanned sequentially
//! to rebuild the in-memory `HashMap`. Truncated trailing records (from a
//! crash mid-write) are silently discarded — the same recovery model as
//! [`BlockPool`](crate::block_pool::BlockPool).
//!
//! # Durability contract
//!
//! [`TextStore::store`] appends and fsyncs before returning. A crash after
//! `store` returns guarantees the text is on disk.
//!
//! # Recovery contract
//!
//! [`TextStore::open`] rebuilds the `HashMap` from the file. If the last
//! record is truncated, it is skipped. No data is served that wasn't fsynced.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use tdb_core::kv_pack::PackId;

const TEXT_STORE_FILENAME: &str = "text_store.bin";
const RECORD_HEADER_SIZE: usize = 8 + 4; // pack_id (u64) + text_len (u32)

/// Append-only store mapping `PackId` to original fact text.
#[derive(Debug)]
pub struct TextStore {
    path: PathBuf,
    texts: HashMap<PackId, String>,
}

impl TextStore {
    /// Open or create the text store in the given directory.
    ///
    /// Scans the file to rebuild the in-memory index. Truncated trailing
    /// records are silently discarded.
    pub fn open(dir: &Path) -> io::Result<Self> {
        let path = dir.join(TEXT_STORE_FILENAME);
        let texts = if path.exists() { Self::replay(&path)? } else { HashMap::new() };
        Ok(Self { path, texts })
    }

    /// Re-read the on-disk file and rebuild the in-memory index.
    ///
    /// Used by `Engine::refresh` (in `tdb-engine`) to pick up writes from another `Engine`
    /// handle at the same path. Idempotent: repeated calls with no on-disk
    /// changes leave the index unchanged.
    pub fn refresh(&mut self) -> io::Result<()> {
        self.texts = if self.path.exists() { Self::replay(&self.path)? } else { HashMap::new() };
        Ok(())
    }

    /// Store text for a pack. Appends to the file and fsyncs.
    ///
    /// Thin wrapper over [`store_batch`](Self::store_batch) for the single-entry
    /// case — the batch path is the canonical implementation.
    pub fn store(&mut self, pack_id: PackId, text: &str) -> io::Result<()> {
        self.store_batch(&[(pack_id, text)])
    }

    /// Store many entries in a single append + fsync.
    ///
    /// Builds the full record buffer in memory, then performs one `write_all`
    /// followed by one `sync_all`. For N entries this collapses N fsyncs into
    /// one — orders of magnitude faster on fsync-bound workloads (migration,
    /// bulk import).
    ///
    /// Last-writer-wins within the batch: if the same `pack_id` appears twice,
    /// the later entry's text is what reads return after this call returns.
    ///
    /// # Recovery
    ///
    /// A crash between `write_all` and `sync_all` may leave the file with a
    /// partial trailing record. [`Self::open`]'s replay discards trailing
    /// records whose declared length exceeds the remaining bytes — durable
    /// state is always a valid record prefix.
    pub fn store_batch(&mut self, entries: &[(PackId, &str)]) -> io::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let total_capacity: usize =
            entries.iter().map(|(_, text)| RECORD_HEADER_SIZE + text.len()).sum();
        let mut buffer = Vec::with_capacity(total_capacity);
        for (pack_id, text) in entries {
            let text_bytes = text.as_bytes();
            buffer.extend_from_slice(&pack_id.to_le_bytes());
            buffer.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
            buffer.extend_from_slice(text_bytes);
        }

        let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        file.write_all(&buffer)?;
        file.sync_all()?;

        for (pack_id, text) in entries {
            self.texts.insert(*pack_id, (*text).to_owned());
        }
        Ok(())
    }

    /// Look up text for a pack.
    pub fn get(&self, pack_id: PackId) -> Option<&str> {
        self.texts.get(&pack_id).map(String::as_str)
    }

    /// Remove a pack's text from the in-memory index.
    ///
    /// The on-disk record remains (append-only) but is shadowed by the
    /// deletion log. On next open, the deletion log prevents this entry
    /// from being loaded.
    pub fn remove(&mut self, pack_id: PackId) {
        self.texts.remove(&pack_id);
    }

    /// Number of text entries.
    pub fn len(&self) -> usize {
        self.texts.len()
    }

    /// Whether the store has no entries.
    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Replay the text store file, rebuilding the in-memory index.
    ///
    /// Duplicate `PackId`s are last-writer-wins (same as re-storing text).
    /// Truncated trailing records are silently skipped.
    fn replay(path: &Path) -> io::Result<HashMap<PackId, String>> {
        let file_len = std::fs::metadata(path)?.len() as usize;
        let mut data = vec![0u8; file_len];
        let mut file = File::open(path)?;
        file.read_exact(&mut data)?;

        let mut texts = HashMap::new();
        let mut cursor = 0;

        while cursor + RECORD_HEADER_SIZE <= data.len() {
            let pack_id = u64::from_le_bytes(data[cursor..cursor + 8].try_into().unwrap());
            let text_len =
                u32::from_le_bytes(data[cursor + 8..cursor + 12].try_into().unwrap()) as usize;

            let record_end = cursor + RECORD_HEADER_SIZE + text_len;
            if record_end > data.len() {
                // Truncated trailing record — discard.
                break;
            }

            let text_bytes = &data[cursor + RECORD_HEADER_SIZE..record_end];
            if let Ok(text) = std::str::from_utf8(text_bytes) {
                texts.insert(pack_id, text.to_owned());
            }
            // Invalid UTF-8 is silently skipped (corrupted record).

            cursor = record_end;
        }

        Ok(texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_store_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = TextStore::open(dir.path()).unwrap();

        store.store(1, "Nyx's favorite star is Vega").unwrap();
        store.store(2, "Corvus collects ancient maps").unwrap();

        assert_eq!(store.get(1), Some("Nyx's favorite star is Vega"));
        assert_eq!(store.get(2), Some("Corvus collects ancient maps"));
        assert_eq!(store.get(99), None);
    }

    #[test]
    fn test_text_store_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();

        {
            let mut store = TextStore::open(dir.path()).unwrap();
            store.store(1, "Memory persists").unwrap();
        }

        let store = TextStore::open(dir.path()).unwrap();
        assert_eq!(store.get(1), Some("Memory persists"));
    }

    #[test]
    fn test_text_store_empty_string() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = TextStore::open(dir.path()).unwrap();

        store.store(1, "").unwrap();
        assert_eq!(store.get(1), Some(""));
    }

    #[test]
    fn test_text_store_remove_is_in_memory() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = TextStore::open(dir.path()).unwrap();

        store.store(1, "Will be removed").unwrap();
        store.remove(1);
        assert_eq!(store.get(1), None);

        // But on-disk record survives (remove is in-memory only).
        let store2 = TextStore::open(dir.path()).unwrap();
        assert_eq!(store2.get(1), Some("Will be removed"));
    }

    #[test]
    fn test_text_store_last_writer_wins() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = TextStore::open(dir.path()).unwrap();

        store.store(1, "First version").unwrap();
        store.store(1, "Updated version").unwrap();

        assert_eq!(store.get(1), Some("Updated version"));

        // Reopen also picks the latest.
        let store2 = TextStore::open(dir.path()).unwrap();
        assert_eq!(store2.get(1), Some("Updated version"));
    }

    #[test]
    fn test_store_batch_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = TextStore::open(dir.path()).unwrap();

        store.store_batch(&[(1, "first"), (2, "second"), (3, "third")]).unwrap();

        assert_eq!(store.get(1), Some("first"));
        assert_eq!(store.get(2), Some("second"));
        assert_eq!(store.get(3), Some("third"));
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_store_batch_empty_is_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(TEXT_STORE_FILENAME);
        let mut store = TextStore::open(dir.path()).unwrap();

        store.store_batch(&[]).unwrap();

        // Empty batch must not create the file (avoids gratuitous I/O).
        assert!(!path.exists());
        assert!(store.is_empty());
    }

    #[test]
    fn test_store_batch_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut store = TextStore::open(dir.path()).unwrap();
            store.store_batch(&[(10, "a"), (20, "b"), (30, "c")]).unwrap();
        }

        let store = TextStore::open(dir.path()).unwrap();
        assert_eq!(store.get(10), Some("a"));
        assert_eq!(store.get(20), Some("b"));
        assert_eq!(store.get(30), Some("c"));
    }

    #[test]
    fn test_store_batch_last_writer_wins_within_batch() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = TextStore::open(dir.path()).unwrap();

        store.store_batch(&[(1, "first"), (1, "overwritten")]).unwrap();

        assert_eq!(store.get(1), Some("overwritten"));

        // Reopen reads the file and the second record wins on replay.
        let store2 = TextStore::open(dir.path()).unwrap();
        assert_eq!(store2.get(1), Some("overwritten"));
    }

    #[test]
    fn test_text_store_truncated_record_discarded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(TEXT_STORE_FILENAME);

        // Write a valid record, then a truncated one.
        {
            let mut store = TextStore::open(dir.path()).unwrap();
            store.store(1, "Valid record").unwrap();
        }

        // Append garbage that looks like a header claiming 1000 bytes of text.
        {
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(&42u64.to_le_bytes()).unwrap();
            file.write_all(&1000u32.to_le_bytes()).unwrap(); // text_len=1000 but no text follows
            file.sync_all().unwrap();
        }

        let store = TextStore::open(dir.path()).unwrap();
        assert_eq!(store.get(1), Some("Valid record"));
        assert_eq!(store.get(42), None); // Truncated record was discarded.
    }

    #[test]
    fn test_text_store_opens_fresh_directory() {
        let dir = tempfile::tempdir().unwrap();
        let store = TextStore::open(dir.path()).unwrap();
        assert!(store.is_empty());
    }
}
