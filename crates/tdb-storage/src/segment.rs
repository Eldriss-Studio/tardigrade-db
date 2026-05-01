//! Segment file management — append-only binary files containing serialized memory cells.
//!
//! Each segment file has a small header followed by variable-length records.
//! New segments are created when the current segment exceeds the size threshold.
//!
//! ## Record Layout (binary, little-endian)
//!
//! ```text
//! [record_len: u32]       — total bytes of this record (excluding this field)
//! [cell_id: u64]          — globally unique cell ID
//! [owner: u64]            — owner/agent ID
//! [layer: u16]            — transformer layer index
//! [key_dim: u32]          — number of f32 elements in key vector
//! [value_dim: u32]        — number of f32 elements in value vector
//! [pos_dim: u32]          — number of f32 elements in pos_encoding
//! [token_span_start: u64] — token span start
//! [token_span_end: u64]   — token span end
//! [created_at: u64]       — creation timestamp (nanos)
//! [updated_at: u64]       — last update timestamp (nanos)
//! [importance: f32]       — importance score
//! [tags: u32]             — tag bitfield
//! [tier: u8]              — maturity tier enum
//! [key_scales_len: u32]   — number of Q4 scale floats for key
//! [key_data_len: u32]     — number of Q4 packed bytes for key
//! [val_scales_len: u32]   — number of Q4 scale floats for value
//! [val_data_len: u32]     — number of Q4 packed bytes for value
//! [key_scales: ...]       — f32 scale factors
//! [key_data: ...]         — packed Q4 bytes
//! [val_scales: ...]       — f32 scale factors
//! [val_data: ...]         — packed Q4 bytes
//! [pos_encoding: ...]     — f32 values (unquantized)
//! ```

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use tdb_core::memory_cell::{CellMeta, MemoryCell};
use tdb_core::types::Tier;
use tdb_core::{CellId, LayerId, OwnerId};

use crate::quantization::{DequantizeStrategy, Q4, QuantizeStrategy, QuantizedTensor};

/// File header magic bytes to identify segment files.
const SEGMENT_MAGIC: &[u8; 4] = b"TDBS";
/// Current segment format version.
const SEGMENT_VERSION: u32 = 1;
/// Size of the file header: magic(4) + version(4) = 8 bytes.
const FILE_HEADER_SIZE: u64 = 8;

/// Location of a record within a segment.
#[derive(Debug, Clone, Copy)]
pub struct RecordLocation {
    pub segment_id: u32,
    pub byte_offset: u64,
}

/// A single segment file — append-only storage for serialized memory cells.
#[derive(Debug)]
pub struct Segment {
    id: u32,
    path: PathBuf,
    size: u64,
}

impl Segment {
    /// Create a new empty segment file.
    pub fn create(dir: &Path, id: u32) -> io::Result<Self> {
        let path = segment_path(dir, id);
        let mut file = File::create(&path)?;
        file.write_all(SEGMENT_MAGIC)?;
        file.write_all(&SEGMENT_VERSION.to_le_bytes())?;
        file.flush()?;
        Ok(Self { id, path, size: FILE_HEADER_SIZE })
    }

    /// Open an existing segment file.
    pub fn open(dir: &Path, id: u32) -> io::Result<Self> {
        let path = segment_path(dir, id);
        let meta = fs::metadata(&path)?;
        Ok(Self { id, path, size: meta.len() })
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Append a `MemoryCell` (quantized to Q4) and return its byte offset.
    pub fn append(&mut self, cell: &MemoryCell) -> io::Result<u64> {
        let file = OpenOptions::new().append(true).open(&self.path)?;
        let mut w = BufWriter::new(file);

        let offset = self.size;
        let record_bytes = write_cell_record(&mut w, cell)?;
        self.size = offset + 4 + record_bytes as u64;

        let inner = w.into_inner().map_err(std::io::IntoInnerError::into_error)?;
        inner.sync_data()?;
        Ok(offset)
    }

    /// Append multiple cells in a single write + single fsync (Write-Behind Buffer).
    ///
    /// Returns a `Vec` of (`byte_offset`, `record_size`) for each cell written.
    /// All cells are durably committed after the single `sync_data()` call.
    pub fn append_batch(&mut self, cells: &[MemoryCell]) -> io::Result<Vec<u64>> {
        if cells.is_empty() {
            return Ok(Vec::new());
        }

        let file = OpenOptions::new().append(true).open(&self.path)?;
        let mut w = BufWriter::new(file);

        let mut offsets = Vec::with_capacity(cells.len());
        for cell in cells {
            let offset = self.size;
            let record_bytes = write_cell_record(&mut w, cell)?;
            self.size = offset + 4 + record_bytes as u64;
            offsets.push(offset);
        }

        // Single fsync for all cells — this is the key optimization.
        let inner = w.into_inner().map_err(std::io::IntoInnerError::into_error)?;
        inner.sync_data()?;
        Ok(offsets)
    }

    /// Read a `MemoryCell` from a specific byte offset.
    pub fn read_at(&self, byte_offset: u64) -> io::Result<MemoryCell> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(byte_offset))?;

        let record_len = read_u32(&mut file)?;
        let _ = record_len; // We read field-by-field, not by total length.

        let cell_id: CellId = read_u64(&mut file)?;
        let owner: OwnerId = read_u64(&mut file)?;
        let layer: LayerId = read_u16(&mut file)?;
        let key_dim = read_u32(&mut file)? as usize;
        let value_dim = read_u32(&mut file)? as usize;
        let pos_dim = read_u32(&mut file)? as usize;
        let token_span_start = read_u64(&mut file)?;
        let token_span_end = read_u64(&mut file)?;
        let created_at = read_u64(&mut file)?;
        let updated_at = read_u64(&mut file)?;
        let importance = read_f32(&mut file)?;
        let tags = read_u32(&mut file)?;
        let tier_byte = read_u8(&mut file)?;

        let key_scales_len = read_u32(&mut file)? as usize;
        let key_data_len = read_u32(&mut file)? as usize;
        let val_scales_len = read_u32(&mut file)? as usize;
        let val_data_len = read_u32(&mut file)? as usize;

        let key_scales = read_f32_vec(&mut file, key_scales_len)?;
        let key_data = read_bytes(&mut file, key_data_len)?;
        let val_scales = read_f32_vec(&mut file, val_scales_len)?;
        let val_data = read_bytes(&mut file, val_data_len)?;
        let pos_encoding = read_f32_vec(&mut file, pos_dim)?;

        let key_q = QuantizedTensor { data: key_data, scales: key_scales, original_len: key_dim };
        let val_q = QuantizedTensor { data: val_data, scales: val_scales, original_len: value_dim };

        let key = Q4::dequantize(&key_q);
        let value = Q4::dequantize(&val_q);

        let tier = match tier_byte {
            0 => Tier::Draft,
            1 => Tier::Validated,
            2 => Tier::Core,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid tier byte {tier_byte} in cell {cell_id}"),
                ));
            }
        };

        Ok(MemoryCell {
            id: cell_id,
            owner,
            layer,
            key,
            value,
            token_span: (token_span_start, token_span_end),
            pos_encoding,
            meta: CellMeta { created_at, updated_at, importance, tags, tier },
        })
    }
}

/// Scan a segment to rebuild the index (used on recovery).
pub fn scan_segment(dir: &Path, segment_id: u32) -> io::Result<Vec<(CellId, u64)>> {
    let path = segment_path(dir, segment_id);
    let mut file = File::open(&path)?;
    let file_len = file.metadata()?.len();

    // Validate file header.
    if file_len < FILE_HEADER_SIZE {
        return Ok(Vec::new());
    }
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != SEGMENT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid segment magic in {}", path.display()),
        ));
    }
    // Skip version (4 bytes) — already past magic.
    file.seek(SeekFrom::Start(FILE_HEADER_SIZE))?;

    let mut entries = Vec::new();
    let mut pos = FILE_HEADER_SIZE;

    while pos < file_len {
        let record_len = match read_u32(&mut file) {
            Ok(len) => len as u64,
            Err(_) => break, // truncated record_len — stop scanning
        };

        // Bounds check: ensure the full record fits within the file.
        if pos + 4 + record_len > file_len {
            break; // partial record at EOF — discard it
        }

        let Ok(cell_id) = read_u64(&mut file) else {
            break; // partial record — stop scanning
        };
        entries.push((cell_id, pos));

        // Skip the rest of this record.
        let remaining = record_len.saturating_sub(8);
        let Ok(seek_offset) = i64::try_from(remaining) else {
            break;
        };
        if file.seek(SeekFrom::Current(seek_offset)).is_err() {
            break;
        }
        pos += 4 + record_len;
    }

    Ok(entries)
}

pub(crate) fn segment_path(dir: &Path, id: u32) -> PathBuf {
    dir.join(format!("segment_{id:06}.tdb"))
}

/// List segment IDs found in a directory, sorted.
pub fn list_segments(dir: &Path) -> io::Result<Vec<u32>> {
    let mut ids = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(rest) = name.strip_prefix("segment_")
            && let Some(num_str) = rest.strip_suffix(".tdb")
            && let Ok(id) = num_str.parse::<u32>()
        {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

fn compute_record_size(key_q: &QuantizedTensor, val_q: &QuantizedTensor, pos_len: usize) -> usize {
    // Fixed fields: id(8) + owner(8) + layer(2) + key_dim(4) + value_dim(4) + pos_dim(4)
    //   + token_span(16) + created_at(8) + updated_at(8) + importance(4) + tags(4) + tier(1)
    //   + key_scales_len(4) + key_data_len(4) + val_scales_len(4) + val_data_len(4)
    let fixed = 8 + 8 + 2 + 4 + 4 + 4 + 16 + 8 + 8 + 4 + 4 + 1 + 4 + 4 + 4 + 4;
    let variable = key_q.scales.len() * 4
        + key_q.data.len()
        + val_q.scales.len() * 4
        + val_q.data.len()
        + pos_len * 4;
    fixed + variable
}

// --- Write helpers ---

/// Serialize a single `MemoryCell` to a writer (Q4 quantized).
/// Returns the record body size (excluding the 4-byte `record_len` prefix).
fn write_cell_record(w: &mut impl Write, cell: &MemoryCell) -> io::Result<u64> {
    let key_q = Q4::quantize(&cell.key);
    let val_q = Q4::quantize(&cell.value);

    let record_bytes = compute_record_size(&key_q, &val_q, cell.pos_encoding.len());
    let record_len: u32 = record_bytes.try_into().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("record size {record_bytes} exceeds u32::MAX"),
        )
    })?;
    w.write_all(&record_len.to_le_bytes())?;

    // Fixed fields.
    w.write_all(&cell.id.to_le_bytes())?;
    w.write_all(&cell.owner.to_le_bytes())?;
    w.write_all(&cell.layer.to_le_bytes())?;
    w.write_all(&(cell.key.len() as u32).to_le_bytes())?;
    w.write_all(&(cell.value.len() as u32).to_le_bytes())?;
    w.write_all(&(cell.pos_encoding.len() as u32).to_le_bytes())?;
    w.write_all(&cell.token_span.0.to_le_bytes())?;
    w.write_all(&cell.token_span.1.to_le_bytes())?;
    w.write_all(&cell.meta.created_at.to_le_bytes())?;
    w.write_all(&cell.meta.updated_at.to_le_bytes())?;
    w.write_all(&cell.meta.importance.to_le_bytes())?;
    w.write_all(&cell.meta.tags.to_le_bytes())?;
    w.write_all(&[cell.meta.tier as u8])?;

    // Quantized key.
    w.write_all(&(key_q.scales.len() as u32).to_le_bytes())?;
    w.write_all(&(key_q.data.len() as u32).to_le_bytes())?;
    // Quantized value.
    w.write_all(&(val_q.scales.len() as u32).to_le_bytes())?;
    w.write_all(&(val_q.data.len() as u32).to_le_bytes())?;

    // Variable-length data.
    write_f32_slice(w, &key_q.scales)?;
    w.write_all(&key_q.data)?;
    write_f32_slice(w, &val_q.scales)?;
    w.write_all(&val_q.data)?;
    write_f32_slice(w, &cell.pos_encoding)?;

    Ok(record_bytes as u64)
}

// --- I/O helpers (little-endian) ---

fn write_f32_slice(w: &mut impl Write, values: &[f32]) -> io::Result<()> {
    for &v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16(r: &mut impl Read) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
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

fn read_f32_vec(r: &mut impl Read, count: usize) -> io::Result<Vec<f32>> {
    let mut buf = vec![0u8; count * 4];
    r.read_exact(&mut buf)?;
    Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn read_bytes(r: &mut impl Read, count: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; count];
    r.read_exact(&mut buf)?;
    Ok(buf)
}
