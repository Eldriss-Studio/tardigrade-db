//! Interleaved prefill/decode scheduler.
//!
//! Hides warm-reload latency (~500ms) behind the previous agent's decode phase
//! to keep the pipeline fully saturated.
