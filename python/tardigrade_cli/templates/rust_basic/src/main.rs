//! TardigradeDB starter — Rust.
//!
//! Opens an engine, writes one pack, reads it back.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_dir = PathBuf::from("./engine_dir");
    let _engine = tdb_engine::Engine::open(&engine_dir)?;

    // TODO: write a pack via tdb_engine::Engine::mem_write_pack and
    // read it back. See https://github.com/Eldriss-Studio/tardigrade-db
    // for the current public API.
    println!("engine opened at {}", engine_dir.display());
    Ok(())
}
