//! Portable snapshot/restore for the engine.
//!
//! Bundles the engine's persistent state into a single tar archive
//! with a forensic-friendly manifest header. The format is adapted
//! from `SpacetimeDB`'s `crates/snapshot` (magic bytes + version +
//! offset-indexed lookup) with `TardigradeDB`-specific additions
//! (codec versioning so snapshots survive future compressor swaps).
//!
//! ## On-disk layout
//!
//! ```text
//! <out_path>.tar
//!   manifest.json           — versioned manifest (see SnapshotManifest)
//!   engine_state/           — verbatim copy of the engine's working dir
//!     <block-pool segments, WAL, text store, deletion log, ...>
//! ```
//!
//! ## Reliability contract
//!
//! - **Durability.** [`crate::engine::Engine::snapshot`] calls [`crate::engine::Engine::flush`] before
//!   tarring; the archive therefore reflects state at the
//!   `durable_offset` of flush completion.
//! - **Recovery.** [`crate::engine::Engine::restore_from`] is equivalent to a fresh
//!   [`crate::engine::Engine::open`] of the same directory after extraction. All
//!   existing replay rules apply unchanged.
//! - **Versioning.** Restore refuses unknown `format_version` and
//!   incompatible `codecs` with structured errors. Forward-compat is
//!   an explicit conversion tool, not implicit.
//! - **Integrity.** A SHA-256 of the `engine_state/` payload is
//!   stored in the manifest; restore verifies it before extracting.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tar::{Archive, Builder};
use tdb_core::error::{Result, TardigradeError};

// ─── Format constants ───────────────────────────────────────────────────

/// Magic marker placed at the head of every `TardigradeDB` snapshot
/// manifest. Lets external tooling (and human forensics) distinguish
/// our archives from arbitrary tarballs.
pub const SNAPSHOT_MAGIC: &str = "tdb!";

/// Current snapshot format version. Increment on any breaking change
/// to the manifest schema, archive layout, or required state files.
/// Older readers refuse newer versions; newer readers may accept
/// older versions when the migration is trivial.
pub const SNAPSHOT_FORMAT_VERSION: u8 = 1;

/// Versioned identifier for the quantization codec currently in use.
/// Snapshots embed this so a future engine that drops Q4 support can
/// refuse Q4 snapshots cleanly instead of producing garbage reads.
pub const SNAPSHOT_QUANT_CODEC: &str = "q4";

/// Versioned identifier for the per-token retrieval-key codec.
pub const SNAPSHOT_KEY_CODEC: &str = "top5avg";

/// Top-level directory inside the tar archive holding the engine's
/// state files. Kept stable across versions so restore tooling can
/// inspect contents without parsing the manifest.
pub const SNAPSHOT_PAYLOAD_DIR: &str = "engine_state";

/// Filename of the manifest inside the tar archive. Always at the
/// archive root, always read first.
pub const SNAPSHOT_MANIFEST_NAME: &str = "manifest.json";

/// Upper bound (bytes) on the manifest size during restore. The
/// real manifest is ~500 bytes; 1 MiB is a generous ceiling that
/// guards against a corrupt or hostile archive declaring an
/// enormous entry and exhausting RAM via [`Read::read_to_end`].
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

/// Upper bound (bytes) on a single payload file during restore.
/// 16 GiB accommodates the largest plausible `BlockPool` segment
/// (`segment_size` is 256 MiB by default; we leave 64× headroom for
/// future configurations) while still bounding worst-case RAM use.
const MAX_PAYLOAD_FILE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

// ─── Manifest ───────────────────────────────────────────────────────────

/// Versioned metadata header written at the root of every snapshot
/// archive. Read first by [`crate::engine::Engine::restore_from`] to validate
/// compatibility before extracting the payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotManifest {
    /// Magic marker; must equal [`SNAPSHOT_MAGIC`].
    pub magic: String,
    /// Format version; restore refuses unknown values.
    pub format_version: u8,
    /// RFC 3339 timestamp of snapshot creation.
    pub created_at: String,
    /// Codec identifiers for the storage formats embedded in the
    /// payload. Restore refuses snapshots whose codecs are not
    /// supported by the current engine.
    pub codecs: SnapshotCodecs,
    /// Cheap stats useful for forensics + repository lookup.
    pub stats: SnapshotStats,
    /// SHA-256 (hex-encoded) of the tarred ``engine_state/`` payload.
    /// Computed *over the engine state files only* — not over the
    /// manifest itself. Verified by [`crate::engine::Engine::restore_from`].
    pub sha256: String,
}

/// Codec identifiers embedded in a snapshot manifest. Used to detect
/// format swaps between snapshot-write and snapshot-restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotCodecs {
    /// Quantization codec (e.g. ``"q4"``).
    pub quantization: String,
    /// Per-token retrieval-key codec (e.g. ``"top5avg"``).
    pub key: String,
}

/// Snapshot stats — handy for the repository's `latest()` lookup and
/// for displaying in admin UIs without unpacking the archive.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SnapshotStats {
    /// Total packs in the engine at snapshot time.
    pub pack_count: usize,
    /// Distinct owners with at least one pack.
    pub owner_count: usize,
}

impl SnapshotManifest {
    /// Construct a manifest with the current format constants.
    /// `sha256` and `stats` must be filled by the caller.
    #[must_use]
    pub fn current(sha256: String, stats: SnapshotStats) -> Self {
        Self {
            magic: SNAPSHOT_MAGIC.to_string(),
            format_version: SNAPSHOT_FORMAT_VERSION,
            created_at: rfc3339_now(),
            codecs: SnapshotCodecs {
                quantization: SNAPSHOT_QUANT_CODEC.to_string(),
                key: SNAPSHOT_KEY_CODEC.to_string(),
            },
            stats,
            sha256,
        }
    }
}

/// Format an RFC 3339 timestamp without pulling in `chrono`. Returns
/// seconds since the Unix epoch as a string; sufficient for forensic
/// ordering and human readability inside admin tooling.
fn rfc3339_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    // Compact ISO-ish: ``1970-01-01T00:00:00Z + N seconds``. We do
    // not need calendar conversion at this layer; downstream tooling
    // can pretty-print if needed.
    format!("@{secs}")
}

// ─── Public snapshot/restore functions ──────────────────────────────────

/// Bundle the engine's working directory into a tar archive at
/// ``out_path``. Returns the manifest by value so callers can log or
/// persist it without re-reading the archive.
///
/// The caller must have already flushed the engine. [`crate::engine::Engine::snapshot`]
/// in `engine.rs` is the proper public entry point and handles flush.
///
/// # Errors
///
/// Returns [`TardigradeError::Io`] on disk failures and a
/// [`TardigradeError::SnapshotIntegrity`] if the manifest cannot be
/// serialized (which would indicate a bug, not user input).
pub fn write_snapshot(
    engine_dir: &Path,
    out_path: &Path,
    stats: SnapshotStats,
) -> Result<SnapshotManifest> {
    // Refuse the foot-gun where ``out_path`` sits inside
    // ``engine_dir`` — the tar walker would read its own growing
    // output and exhaust disk. Caller must write the snapshot
    // somewhere outside the engine state.
    if let (Ok(out_abs), Ok(engine_abs)) = (
        out_path.canonicalize().or_else(|_| {
            // The output usually doesn't exist yet; resolve its parent.
            out_path.parent().map_or_else(
                || Ok(out_path.to_path_buf()),
                |p| p.canonicalize().map(|abs| abs.join(out_path.file_name().unwrap_or_default())),
            )
        }),
        engine_dir.canonicalize(),
    ) && out_abs.starts_with(&engine_abs)
    {
        return Err(TardigradeError::SnapshotIntegrity(format!(
            "out_path ({}) is inside engine_dir ({}); pick a path outside the engine directory",
            out_path.display(),
            engine_dir.display(),
        )));
    }

    let mut hasher = Sha256::new();

    // Step 1: hash the payload first so we can include the digest
    // in the manifest. We walk the same set of files we'll
    // subsequently tar so the digest matches what restore will see.
    walk_files_for_hash(engine_dir, &mut hasher)?;
    let sha256 = hex_lower(&hasher.finalize());

    let manifest = SnapshotManifest::current(sha256, stats);
    let manifest_json = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| TardigradeError::SnapshotIntegrity(format!("manifest serialize: {e}")))?;

    // Step 2: write the tar archive: manifest.json first, then the
    // engine_state/ payload.
    let file = File::create(out_path).map_err(|e| TardigradeError::Io { source: e })?;
    let mut tar = Builder::new(file);
    tar.mode(tar::HeaderMode::Deterministic);

    // Manifest entry.
    let mut header = tar::Header::new_gnu();
    header.set_path(SNAPSHOT_MANIFEST_NAME).map_err(|e| TardigradeError::Io { source: e })?;
    header.set_size(manifest_json.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    tar.append(&header, &manifest_json[..]).map_err(|e| TardigradeError::Io { source: e })?;

    // Payload: copy every regular file under engine_dir into
    // engine_state/ inside the archive.
    tar.append_dir_all(SNAPSHOT_PAYLOAD_DIR, engine_dir)
        .map_err(|e| TardigradeError::Io { source: e })?;

    tar.finish().map_err(|e| TardigradeError::Io { source: e })?;
    Ok(manifest)
}

/// Read just the manifest from a snapshot archive without
/// extracting the payload or verifying the integrity hash.
///
/// Useful for listing operations that only need metadata (label
/// directories, repository enumeration). Magic and format version
/// are still validated — a tar that lies about being a
/// `TardigradeDB` snapshot is refused here too.
///
/// # Errors
///
/// - [`TardigradeError::NotATardigradeSnapshot`] — magic missing.
/// - [`TardigradeError::UnsupportedFormatVersion`] — unknown version.
/// - [`TardigradeError::Io`] — disk failure.
pub fn read_manifest_only(in_path: &Path) -> Result<SnapshotManifest> {
    let file = File::open(in_path).map_err(|e| TardigradeError::Io { source: e })?;
    let mut archive = Archive::new(file);
    for entry in archive.entries().map_err(|e| TardigradeError::Io { source: e })? {
        let mut entry = entry.map_err(|e| TardigradeError::Io { source: e })?;
        let path = entry.path().map_err(|e| TardigradeError::Io { source: e })?.to_path_buf();
        if path.as_os_str() != SNAPSHOT_MANIFEST_NAME {
            continue;
        }
        let buf = read_entry_bounded(&mut entry, MAX_MANIFEST_BYTES)?;
        let manifest: SnapshotManifest = serde_json::from_slice(&buf).map_err(|e| {
            TardigradeError::NotATardigradeSnapshot { reason: format!("manifest parse: {e}") }
        })?;
        if manifest.magic != SNAPSHOT_MAGIC {
            return Err(TardigradeError::NotATardigradeSnapshot {
                reason: format!("unexpected magic: {:?}", manifest.magic),
            });
        }
        if manifest.format_version != SNAPSHOT_FORMAT_VERSION {
            return Err(TardigradeError::UnsupportedFormatVersion {
                found: manifest.format_version,
                supported: SNAPSHOT_FORMAT_VERSION,
            });
        }
        return Ok(manifest);
    }
    Err(TardigradeError::NotATardigradeSnapshot { reason: "manifest.json missing".to_string() })
}

/// Restore a snapshot archive into ``target_dir`` (which must be
/// empty or non-existent) and validate its manifest. The caller
/// invokes [`crate::engine::Engine::open`] on ``target_dir`` afterwards.
///
/// Returns the parsed manifest so callers can inspect or log the
/// snapshot's metadata.
///
/// # Errors
///
/// - [`TardigradeError::NotATardigradeSnapshot`] — magic missing.
/// - [`TardigradeError::UnsupportedFormatVersion`] — unknown version.
/// - [`TardigradeError::SnapshotCodecMismatch`] — codec incompatible.
/// - [`TardigradeError::SnapshotIntegrity`] — SHA-256 mismatch.
/// - [`TardigradeError::Io`] — disk failure.
pub fn read_snapshot(in_path: &Path, target_dir: &Path) -> Result<SnapshotManifest> {
    let file = File::open(in_path).map_err(|e| TardigradeError::Io { source: e })?;
    let mut archive = Archive::new(file);

    // First entry: manifest. Pull it out independently so we can
    // refuse the snapshot before extracting any payload.
    let mut manifest_bytes: Option<Vec<u8>> = None;
    let mut payload_entries: Vec<(PathBuf, Vec<u8>)> = Vec::new();

    for entry in archive.entries().map_err(|e| TardigradeError::Io { source: e })? {
        let mut entry = entry.map_err(|e| TardigradeError::Io { source: e })?;
        let path = entry.path().map_err(|e| TardigradeError::Io { source: e })?.to_path_buf();

        if path.as_os_str() == SNAPSHOT_MANIFEST_NAME {
            let buf = read_entry_bounded(&mut entry, MAX_MANIFEST_BYTES)?;
            manifest_bytes = Some(buf);
            continue;
        }
        // Only regular files inside engine_state/ are part of the
        // payload; tar Directory entries are skipped (we re-create
        // dirs on extract).
        if !entry.header().entry_type().is_file() {
            continue;
        }
        if !path.starts_with(SNAPSHOT_PAYLOAD_DIR) {
            continue;
        }
        let buf = read_entry_bounded(&mut entry, MAX_PAYLOAD_FILE_BYTES)?;
        payload_entries.push((path, buf));
    }

    let manifest_bytes = manifest_bytes.ok_or_else(|| TardigradeError::NotATardigradeSnapshot {
        reason: "manifest.json missing".to_string(),
    })?;
    let manifest: SnapshotManifest = serde_json::from_slice(&manifest_bytes).map_err(|e| {
        TardigradeError::NotATardigradeSnapshot { reason: format!("manifest parse: {e}") }
    })?;

    if manifest.magic != SNAPSHOT_MAGIC {
        return Err(TardigradeError::NotATardigradeSnapshot {
            reason: format!("unexpected magic: {:?}", manifest.magic),
        });
    }
    if manifest.format_version != SNAPSHOT_FORMAT_VERSION {
        return Err(TardigradeError::UnsupportedFormatVersion {
            found: manifest.format_version,
            supported: SNAPSHOT_FORMAT_VERSION,
        });
    }
    if manifest.codecs.quantization != SNAPSHOT_QUANT_CODEC {
        return Err(TardigradeError::SnapshotCodecMismatch {
            field: "quantization".to_string(),
            snapshot: manifest.codecs.quantization.clone(),
            engine: SNAPSHOT_QUANT_CODEC.to_string(),
        });
    }
    if manifest.codecs.key != SNAPSHOT_KEY_CODEC {
        return Err(TardigradeError::SnapshotCodecMismatch {
            field: "key".to_string(),
            snapshot: manifest.codecs.key.clone(),
            engine: SNAPSHOT_KEY_CODEC.to_string(),
        });
    }

    // Verify the payload SHA-256 against the manifest. Both write
    // and read sides hash in *lexicographic* order of the relative
    // path so the digest is order-independent of the underlying
    // filesystem (OS readdir order leaked through tar entries
    // otherwise and produced spurious mismatches).
    payload_entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = Sha256::new();
    for (path, bytes) in &payload_entries {
        let rel = strip_payload_prefix(path);
        hash_one(&mut hasher, rel.as_os_str(), bytes);
    }
    let computed = hex_lower(&hasher.finalize());
    if computed != manifest.sha256 {
        return Err(TardigradeError::SnapshotIntegrity(format!(
            "sha256 mismatch: manifest={}, computed={}",
            manifest.sha256, computed
        )));
    }

    // Extract payload into target_dir.
    fs::create_dir_all(target_dir).map_err(|e| TardigradeError::Io { source: e })?;
    for (path, bytes) in payload_entries {
        let rel = strip_payload_prefix(&path);
        let dest = target_dir.join(rel);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| TardigradeError::Io { source: e })?;
        }
        let mut f = File::create(&dest).map_err(|e| TardigradeError::Io { source: e })?;
        f.write_all(&bytes).map_err(|e| TardigradeError::Io { source: e })?;
    }

    Ok(manifest)
}

// ─── Internal helpers ───────────────────────────────────────────────────

/// Walk every regular file under ``engine_dir`` in a stable order
/// (lexicographic by relative path) and feed each path + contents
/// into ``hasher``. The order must match the equivalent walk on the
/// read side, otherwise the digest will mismatch.
fn walk_files_for_hash(engine_dir: &Path, hasher: &mut Sha256) -> Result<()> {
    let mut entries: Vec<PathBuf> = Vec::new();
    collect_files(engine_dir, engine_dir, &mut entries)?;
    entries.sort();
    for rel in entries {
        let abs = engine_dir.join(&rel);
        let bytes = fs::read(&abs).map_err(|e| TardigradeError::Io { source: e })?;
        hash_one(hasher, rel.as_os_str(), &bytes);
    }
    Ok(())
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).map_err(|e| TardigradeError::Io { source: e })? {
        let entry = entry.map_err(|e| TardigradeError::Io { source: e })?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, out)?;
        } else if path.is_file() {
            let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
            out.push(rel);
        }
    }
    Ok(())
}

fn strip_payload_prefix(path: &Path) -> PathBuf {
    path.strip_prefix(SNAPSHOT_PAYLOAD_DIR).unwrap_or(path).to_path_buf()
}

/// Read up to `max_bytes` from `r` into a fresh `Vec<u8>`, erroring
/// if more bytes are available. Provides the [`Read::read_to_end`]
/// convenience without the OOM exposure on corrupt archives that
/// declare absurdly large entry sizes.
fn read_entry_bounded<R: Read>(r: &mut R, max_bytes: u64) -> Result<Vec<u8>> {
    // Drain the entry into a Vec via a fixed scratch buffer; refuses
    // (with a typed integrity error) any entry that exceeds
    // `max_bytes`. Hand-rolled rather than `Read::read_to_end` so we
    // never allocate beyond the cap on a corrupt or hostile archive.
    const CHUNK: usize = 64 * 1024;
    let mut buf: Vec<u8> = Vec::new();
    let mut scratch = vec![0u8; CHUNK];
    let max = max_bytes as usize;
    loop {
        let n = r.read(&mut scratch).map_err(|e| TardigradeError::Io { source: e })?;
        if n == 0 {
            return Ok(buf);
        }
        if buf.len().saturating_add(n) > max {
            return Err(TardigradeError::SnapshotIntegrity(format!(
                "entry exceeds maximum allowed size of {max_bytes} bytes",
            )));
        }
        buf.extend_from_slice(&scratch[..n]);
    }
}

fn hash_one(hasher: &mut Sha256, name: &std::ffi::OsStr, bytes: &[u8]) {
    // Mix the path into the hash so reordered renames cannot collide
    // with each other.
    hasher.update(name.as_encoded_bytes());
    hasher.update(b"\0");
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}
