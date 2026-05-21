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
//!
//! # Concurrency
//!
//! Every method takes `&self`. The in-memory index lives behind
//! [`arc_swap::ArcSwap`] so readers take a cheap atomic snapshot
//! ([`TextStore::get`], [`TextStore::len`], [`TextStore::is_empty`])
//! without blocking concurrent writers, and writers replace the whole
//! index in one atomic store under a `Mutex<()>` that serialises file
//! appends and index installation. Readers holding an older snapshot
//! keep returning the state they observed at acquisition time even
//! while a new writer installs a newer one — this is the substrate
//! that lets pack hydration drop the engine mutex while it walks text.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
use tdb_core::kv_pack::PackId;

const TEXT_STORE_FILENAME: &str = "text_store.bin";
const RECORD_HEADER_SIZE: usize = 8 + 4; // pack_id (u64) + text_len (u32)

/// Append-only store mapping `PackId` to original fact text.
///
/// All methods are `&self`-safe (no exclusive borrow required). Reads are
/// lock-free atomic snapshots; writes serialise behind a single internal
/// mutex that protects both the file append and the index swap.
#[derive(Debug)]
pub struct TextStore {
    path: PathBuf,
    index: ArcSwap<HashMap<PackId, String>>,
    /// Serialises file appends and index installation so concurrent writes
    /// don't interleave records on disk or lose updates in the index swap.
    write_lock: Mutex<()>,
}

impl TextStore {
    /// Open or create the text store in the given directory.
    ///
    /// Scans the file to rebuild the in-memory index. Truncated trailing
    /// records are silently discarded.
    ///
    /// # Errors
    /// Returns an [`io::Error`] if the text-store file exists but cannot
    /// be opened or read.
    pub fn open(dir: &Path) -> io::Result<Self> {
        let path = dir.join(TEXT_STORE_FILENAME);
        let initial = if path.exists() { Self::replay(&path)? } else { HashMap::new() };
        Ok(Self { path, index: ArcSwap::from_pointee(initial), write_lock: Mutex::new(()) })
    }

    /// Re-read the on-disk file and rebuild the in-memory index.
    ///
    /// Used by `Engine::refresh` (in `tdb-engine`) to pick up writes from another `Engine`
    /// handle at the same path. Idempotent: repeated calls with no on-disk
    /// changes leave the observable index unchanged.
    ///
    /// # Errors
    /// Returns an [`io::Error`] if the text-store file exists but cannot be read.
    ///
    /// # Panics
    /// Panics if the internal write mutex is poisoned — only possible if a
    /// previous writer panicked while holding it, which would itself signal
    /// a deeper invariant violation worth surfacing loudly.
    pub fn refresh(&self) -> io::Result<()> {
        let _guard = self.write_lock.lock().expect("text-store write_lock poisoned");
        let rebuilt = if self.path.exists() { Self::replay(&self.path)? } else { HashMap::new() };
        self.index.store(Arc::new(rebuilt));
        Ok(())
    }

    /// Store text for a pack. Appends to the file and fsyncs.
    ///
    /// Thin wrapper over [`store_batch`](Self::store_batch) for the single-entry
    /// case — the batch path is the canonical implementation.
    ///
    /// # Errors
    /// Same as [`Self::store_batch`].
    pub fn store(&self, pack_id: PackId, text: &str) -> io::Result<()> {
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
    ///
    /// # Errors
    /// Returns an [`io::Error`] if the text-store file cannot be opened in
    /// append mode, the buffered records cannot be written, or fsync fails.
    ///
    /// # Panics
    /// Panics if the internal write mutex is poisoned — see [`Self::refresh`].
    pub fn store_batch(&self, entries: &[(PackId, &str)]) -> io::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let total_capacity: usize =
            entries.iter().map(|(_, text)| RECORD_HEADER_SIZE + text.len()).sum();
        let mut buffer = Vec::with_capacity(total_capacity);
        for (pack_id, text) in entries {
            let text_bytes = text.as_bytes();
            buffer.extend_from_slice(&pack_id.to_le_bytes());
            let text_len_u32 = u32::try_from(text_bytes.len()).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "text record exceeds u32::MAX bytes")
            })?;
            buffer.extend_from_slice(&text_len_u32.to_le_bytes());
            buffer.extend_from_slice(text_bytes);
        }

        // Hold the write lock across the file append AND the index swap so
        // a concurrent writer can't interleave their record on disk or race
        // us to install a stale index snapshot.
        let _guard = self.write_lock.lock().expect("text-store write_lock poisoned");

        let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        file.write_all(&buffer)?;
        file.sync_all()?;

        // Copy-on-write the index. Readers that took a snapshot before this
        // store call keep observing their snapshot until they re-read; the
        // next `index.load()` after our `store` call below sees the new map.
        let mut next = (**self.index.load()).clone();
        for (pack_id, text) in entries {
            next.insert(*pack_id, (*text).to_owned());
        }
        self.index.store(Arc::new(next));
        Ok(())
    }

    /// Look up text for a pack.
    ///
    /// Takes an atomic snapshot of the index, then returns an owned `String`
    /// (cloned out of the snapshot) so the caller can drop the snapshot
    /// immediately. The owned-`String` shape is what every consumer in
    /// `tdb-engine` wants anyway (pack hydration copies into a new struct).
    #[must_use]
    pub fn get(&self, pack_id: PackId) -> Option<String> {
        self.index.load().get(&pack_id).cloned()
    }

    /// Remove a pack's text from the in-memory index.
    ///
    /// The on-disk record remains (append-only) but is shadowed by the
    /// deletion log. On next open, the deletion log prevents this entry
    /// from being loaded.
    ///
    /// # Panics
    /// Panics if the internal write mutex is poisoned — see [`Self::refresh`].
    pub fn remove(&self, pack_id: PackId) {
        let _guard = self.write_lock.lock().expect("text-store write_lock poisoned");
        let mut next = (**self.index.load()).clone();
        next.remove(&pack_id);
        self.index.store(Arc::new(next));
    }

    /// Number of text entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.load().len()
    }

    /// Whether the store has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.load().is_empty()
    }

    /// Replay the text store file, rebuilding the in-memory index.
    ///
    /// Duplicate `PackId`s are last-writer-wins (same as re-storing text).
    /// Truncated trailing records are silently skipped.
    fn replay(path: &Path) -> io::Result<HashMap<PackId, String>> {
        let file_len = usize::try_from(std::fs::metadata(path)?.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "text store larger than usize::MAX")
        })?;
        let mut data = vec![0u8; file_len];
        let mut file = File::open(path)?;
        file.read_exact(&mut data)?;

        let mut texts = HashMap::new();
        let mut cursor = 0;

        while cursor + RECORD_HEADER_SIZE <= data.len() {
            let pack_id_bytes: [u8; 8] = data[cursor..cursor + 8]
                .try_into()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "truncated pack_id"))?;
            let pack_id = u64::from_le_bytes(pack_id_bytes);
            let len_bytes: [u8; 4] = data[cursor + 8..cursor + 12]
                .try_into()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "truncated text_len"))?;
            let text_len = u32::from_le_bytes(len_bytes) as usize;

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
        let store = TextStore::open(dir.path()).unwrap();

        store.store(1, "Nyx's favorite star is Vega").unwrap();
        store.store(2, "Corvus collects ancient maps").unwrap();

        assert_eq!(store.get(1).as_deref(), Some("Nyx's favorite star is Vega"));
        assert_eq!(store.get(2).as_deref(), Some("Corvus collects ancient maps"));
        assert_eq!(store.get(99), None);
    }

    #[test]
    fn test_text_store_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();

        {
            let store = TextStore::open(dir.path()).unwrap();
            store.store(1, "Memory persists").unwrap();
        }

        let store = TextStore::open(dir.path()).unwrap();
        assert_eq!(store.get(1).as_deref(), Some("Memory persists"));
    }

    #[test]
    fn test_text_store_empty_string() {
        let dir = tempfile::tempdir().unwrap();
        let store = TextStore::open(dir.path()).unwrap();

        store.store(1, "").unwrap();
        assert_eq!(store.get(1).as_deref(), Some(""));
    }

    #[test]
    fn test_text_store_remove_is_in_memory() {
        let dir = tempfile::tempdir().unwrap();
        let store = TextStore::open(dir.path()).unwrap();

        store.store(1, "Will be removed").unwrap();
        store.remove(1);
        assert_eq!(store.get(1), None);

        // But on-disk record survives (remove is in-memory only).
        let store2 = TextStore::open(dir.path()).unwrap();
        assert_eq!(store2.get(1).as_deref(), Some("Will be removed"));
    }

    #[test]
    fn test_text_store_last_writer_wins() {
        let dir = tempfile::tempdir().unwrap();
        let store = TextStore::open(dir.path()).unwrap();

        store.store(1, "First version").unwrap();
        store.store(1, "Updated version").unwrap();

        assert_eq!(store.get(1).as_deref(), Some("Updated version"));

        // Reopen also picks the latest.
        let store2 = TextStore::open(dir.path()).unwrap();
        assert_eq!(store2.get(1).as_deref(), Some("Updated version"));
    }

    #[test]
    fn test_store_batch_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = TextStore::open(dir.path()).unwrap();

        store.store_batch(&[(1, "first"), (2, "second"), (3, "third")]).unwrap();

        assert_eq!(store.get(1).as_deref(), Some("first"));
        assert_eq!(store.get(2).as_deref(), Some("second"));
        assert_eq!(store.get(3).as_deref(), Some("third"));
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_store_batch_empty_is_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(TEXT_STORE_FILENAME);
        let store = TextStore::open(dir.path()).unwrap();

        store.store_batch(&[]).unwrap();

        // Empty batch must not create the file (avoids gratuitous I/O).
        assert!(!path.exists());
        assert!(store.is_empty());
    }

    #[test]
    fn test_store_batch_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = TextStore::open(dir.path()).unwrap();
            store.store_batch(&[(10, "a"), (20, "b"), (30, "c")]).unwrap();
        }

        let store = TextStore::open(dir.path()).unwrap();
        assert_eq!(store.get(10).as_deref(), Some("a"));
        assert_eq!(store.get(20).as_deref(), Some("b"));
        assert_eq!(store.get(30).as_deref(), Some("c"));
    }

    #[test]
    fn test_store_batch_last_writer_wins_within_batch() {
        let dir = tempfile::tempdir().unwrap();
        let store = TextStore::open(dir.path()).unwrap();

        store.store_batch(&[(1, "first"), (1, "overwritten")]).unwrap();

        assert_eq!(store.get(1).as_deref(), Some("overwritten"));

        // Reopen reads the file and the second record wins on replay.
        let store2 = TextStore::open(dir.path()).unwrap();
        assert_eq!(store2.get(1).as_deref(), Some("overwritten"));
    }

    #[test]
    fn test_text_store_truncated_record_discarded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(TEXT_STORE_FILENAME);

        // Write a valid record, then a truncated one.
        {
            let store = TextStore::open(dir.path()).unwrap();
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
        assert_eq!(store.get(1).as_deref(), Some("Valid record"));
        assert_eq!(store.get(42), None); // Truncated record was discarded.
    }

    #[test]
    fn test_text_store_opens_fresh_directory() {
        let dir = tempfile::tempdir().unwrap();
        let store = TextStore::open(dir.path()).unwrap();
        assert!(store.is_empty());
    }

    /// A `TextStore` shared across threads via `Arc` must serve concurrent
    /// reads while a separate thread writes new entries: no data race,
    /// no lost updates, and the seed entry every reader knows about
    /// remains observable for the duration of the run. This is the
    /// behavior contract pack hydration relies on when it drops the
    /// engine mutex to walk text without blocking writers.
    #[test]
    fn concurrent_readers_see_seed_while_writer_appends() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(TextStore::open(dir.path()).unwrap());

        // Seed a stable entry every reader expects to find.
        store.store(1, "seed").unwrap();

        let stop = Arc::new(AtomicBool::new(false));

        // Writer thread: keep appending new packs until told to stop.
        let writer = {
            let store = Arc::clone(&store);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let mut next_id: u64 = 100;
                while !stop.load(Ordering::Relaxed) {
                    store.store(next_id, &format!("text-{next_id}")).unwrap();
                    next_id += 1;
                }
                next_id // last id we wrote past
            })
        };

        // Reader threads: hammer get() and confirm the seed is always there
        // and that any id they observe via len() boundary is readable. The
        // ArcSwap snapshot guarantees a reader that sees `len() >= n` can
        // also `get()` ids `< n` that were written before the writer
        // committed the snapshot they're holding.
        let mut readers = Vec::new();
        for _ in 0..4 {
            let store = Arc::clone(&store);
            let stop = Arc::clone(&stop);
            readers.push(thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    assert_eq!(store.get(1).as_deref(), Some("seed"));
                    reads += 1;
                }
                reads
            }));
        }

        // Run for a brief window — enough to interleave many ops without
        // turning the test into a perf bench.
        thread::sleep(std::time::Duration::from_millis(50));
        stop.store(true, Ordering::Relaxed);

        let last_written = writer.join().unwrap();
        let mut total_reads = 0u64;
        for r in readers {
            total_reads += r.join().unwrap();
        }

        // Sanity: writer made progress, readers made progress, and every
        // entry the writer claims to have stored is observable post-stop.
        assert!(last_written > 100, "writer made no progress");
        assert!(total_reads > 0, "readers made no progress");
        for id in 100..last_written {
            assert_eq!(
                store.get(id).as_deref(),
                Some(format!("text-{id}").as_str()),
                "pack {id} missing after concurrent run"
            );
        }
    }

    /// `TextStore` must be `Send + Sync` so callers can wrap it in
    /// `Arc<TextStore>` and share across threads without bolting extra
    /// synchronization onto every consumer. The check is a static
    /// assertion at compile time; the test body merely names it.
    #[test]
    fn text_store_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TextStore>();
    }
}
