#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;
use tdb_storage::segment::Segment;

fuzz_target!(|data: &[u8]| {
    // Write arbitrary bytes to a temp file and attempt to read as a segment.
    let dir = tempfile::tempdir().unwrap();
    let segment_path = dir.path().join("seg_00000000.tdb");
    {
        let mut file = std::fs::File::create(&segment_path).unwrap();
        file.write_all(data).unwrap();
    }

    // Attempt to open — should either succeed or return an error, never panic.
    match Segment::open(dir.path(), 0) {
        Ok(seg) => {
            // If it opened, try reading at offset 0.
            // Should either return a valid cell or an error.
            let _ = seg.read_at(0);
        }
        Err(_) => {
            // Expected for most fuzzed inputs — corrupt data should be rejected cleanly.
        }
    }
});
