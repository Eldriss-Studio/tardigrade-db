//! Segment file management — 256MB append-only files containing serialized memory cells.
//!
//! New segments are created when the current segment exceeds the size threshold.
