//! Append-only file store for `SynapticBankEntry` (`LoRA` adapters).
//!
//! Repository pattern: mirrors `BlockPool` conventions — binary records,
//! length-prefixed, `sync_data()` for durability, scan-on-open for recovery.
//!
//! ## Record format (binary, little-endian)
//!
//! ```text
//! [record_len: u32]    — total bytes excluding this field
//! [id: u64]            — synaptic entry ID
//! [owner: u64]         — agent/user owner
//! [rank: u32]          — LoRA rank
//! [d_model: u32]       — model dimension
//! [scale: u16]         — f16 scaling factor
//! [last_used: u64]     — timestamp
//! [quality: f32]       — quality score
//! [lora_a: ...]        — rank * d_model * 2 bytes (f16)
//! [lora_b: ...]        — d_model * rank * 2 bytes (f16)
//! ```

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use half::f16;
use tdb_core::synaptic_bank::SynapticBankEntry;
use tdb_core::types::OwnerId;

const SYNAPTIC_FILE_NAME: &str = "synaptic.tdb";

/// Repository for persisting and loading `SynapticBankEntry` records.
#[derive(Debug)]
pub struct SynapticStore {
    path: PathBuf,
    /// In-memory index: (id, owner, `byte_offset`).
    index: Vec<(u64, OwnerId, u64)>,
}

impl SynapticStore {
    /// Open or create a synaptic store at the given directory.
    pub fn open(dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(SYNAPTIC_FILE_NAME);

        // Touch file if it doesn't exist.
        OpenOptions::new().create(true).append(true).open(&path)?;

        // Scan to build index.
        let index = scan_entries(&path)?;

        Ok(Self { path, index })
    }

    /// Append a `SynapticBankEntry` to the store.
    pub fn append(&mut self, entry: &SynapticBankEntry) -> io::Result<()> {
        let file = OpenOptions::new().append(true).open(&self.path)?;
        let offset = file.metadata()?.len();
        let mut w = BufWriter::new(file);

        let lora_bytes = entry.lora_a.len() * 2 + entry.lora_b.len() * 2;
        // Fixed: id(8) + owner(8) + rank(4) + d_model(4) + scale(2) + last_used(8) + quality(4) = 38
        let record_len: u32 = (38 + lora_bytes)
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "record too large"))?;

        w.write_all(&record_len.to_le_bytes())?;
        w.write_all(&entry.id.to_le_bytes())?;
        w.write_all(&entry.owner.to_le_bytes())?;
        w.write_all(&entry.rank.to_le_bytes())?;
        w.write_all(&entry.d_model.to_le_bytes())?;
        w.write_all(&entry.scale.to_le_bytes())?;
        w.write_all(&entry.last_used.to_le_bytes())?;
        w.write_all(&entry.quality.to_le_bytes())?;

        write_f16_slice(&mut w, &entry.lora_a)?;
        write_f16_slice(&mut w, &entry.lora_b)?;

        let inner = w.into_inner().map_err(std::io::IntoInnerError::into_error)?;
        inner.sync_data()?;

        self.index.push((entry.id, entry.owner, offset));
        Ok(())
    }

    /// Load all entries belonging to a specific owner.
    pub fn load_by_owner(&self, owner: OwnerId) -> io::Result<Vec<SynapticBankEntry>> {
        let offsets: Vec<u64> = self
            .index
            .iter()
            .filter(|(_, o, _)| *o == owner)
            .map(|(_, _, offset)| *offset)
            .collect();

        let mut results = Vec::with_capacity(offsets.len());
        for offset in offsets {
            results.push(read_entry_at(&self.path, offset)?);
        }
        Ok(results)
    }

    /// Number of entries in the store.
    pub fn entry_count(&self) -> usize {
        self.index.len()
    }
}

fn scan_entries(path: &Path) -> io::Result<Vec<(u64, OwnerId, u64)>> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut entries = Vec::new();
    let mut pos = 0u64;

    while pos + 4 <= file_len {
        let record_len = match read_u32(&mut file) {
            Ok(len) => len as u64,
            Err(_) => break,
        };

        if pos + 4 + record_len > file_len {
            break;
        }

        let Ok(id) = read_u64(&mut file) else {
            break;
        };
        let Ok(owner) = read_u64(&mut file) else {
            break;
        };

        entries.push((id, owner, pos));

        // Skip rest of record: already read 4 (record_len) + 8 (id) + 8 (owner) = 20.
        let remaining = record_len.saturating_sub(16);
        if file.seek(SeekFrom::Current(i64::try_from(remaining).unwrap_or(i64::MAX))).is_err() {
            break;
        }
        pos += 4 + record_len;
    }

    Ok(entries)
}

fn read_entry_at(path: &Path, offset: u64) -> io::Result<SynapticBankEntry> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;

    let _record_len = read_u32(&mut file)?;
    let id = read_u64(&mut file)?;
    let owner = read_u64(&mut file)?;
    let rank = read_u32(&mut file)?;
    let d_model = read_u32(&mut file)?;

    let mut scale_buf = [0u8; 2];
    file.read_exact(&mut scale_buf)?;
    let scale = f16::from_le_bytes(scale_buf);

    let last_used = read_u64(&mut file)?;
    let quality = read_f32(&mut file)?;

    let lora_len = (rank as usize) * (d_model as usize);
    let lora_a = read_f16_vec(&mut file, lora_len)?;
    let lora_b = read_f16_vec(&mut file, lora_len)?;

    Ok(SynapticBankEntry { id, owner, lora_a, lora_b, scale, rank, d_model, last_used, quality })
}

fn write_f16_slice(w: &mut impl Write, values: &[f16]) -> io::Result<()> {
    for &v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_f16_vec(r: &mut impl Read, count: usize) -> io::Result<Vec<f16>> {
    let mut buf = vec![0u8; count * 2];
    r.read_exact(&mut buf)?;
    Ok(buf.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]])).collect())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
