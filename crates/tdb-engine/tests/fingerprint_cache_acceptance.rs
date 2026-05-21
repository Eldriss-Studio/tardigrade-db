//! Acceptance tests for the engine-side fingerprint LRU cache.

use std::path::PathBuf;

use tdb_engine::engine::Engine;

fn tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tdb-fp-at-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn it_returns_none_before_capacity_is_configured() {
    let dir = tmp_dir("unconfigured");
    let mut eng = Engine::open(&dir).unwrap();
    assert!(eng.fingerprint_get(42).is_none());
}

#[test]
fn it_evicts_oldest_when_full() {
    let dir = tmp_dir("evict");
    let mut eng = Engine::open(&dir).unwrap();
    eng.set_fingerprint_capacity(3);
    eng.fingerprint_put(1, 100);
    eng.fingerprint_put(2, 200);
    eng.fingerprint_put(3, 300);
    eng.fingerprint_put(4, 400); // evicts 1
    assert!(eng.fingerprint_get(1).is_none());
    assert_eq!(eng.fingerprint_get(2), Some(200));
    assert_eq!(eng.fingerprint_get(4), Some(400));
    assert_eq!(eng.fingerprint_len(), 3);
}

#[test]
fn it_promotes_on_get() {
    let dir = tmp_dir("promote");
    let mut eng = Engine::open(&dir).unwrap();
    eng.set_fingerprint_capacity(3);
    eng.fingerprint_put(1, 100);
    eng.fingerprint_put(2, 200);
    eng.fingerprint_put(3, 300);
    let _ = eng.fingerprint_get(1); // promotes 1
    eng.fingerprint_put(4, 400); // evicts 2 (now oldest)
    assert_eq!(eng.fingerprint_get(1), Some(100));
    assert!(eng.fingerprint_get(2).is_none());
}

#[test]
fn it_release_removes_specific_entry() {
    let dir = tmp_dir("release");
    let mut eng = Engine::open(&dir).unwrap();
    eng.set_fingerprint_capacity(3);
    eng.fingerprint_put(11, 110);
    eng.fingerprint_put(12, 120);
    eng.fingerprint_release(11);
    assert!(eng.fingerprint_get(11).is_none());
    assert_eq!(eng.fingerprint_get(12), Some(120));
}

#[test]
fn it_resize_preserves_recent_entries() {
    let dir = tmp_dir("resize");
    let mut eng = Engine::open(&dir).unwrap();
    eng.set_fingerprint_capacity(4);
    eng.fingerprint_put(1, 100);
    eng.fingerprint_put(2, 200);
    eng.fingerprint_put(3, 300);
    eng.fingerprint_put(4, 400);
    eng.set_fingerprint_capacity(2); // drop 1 and 2 (oldest)
    assert!(eng.fingerprint_get(1).is_none());
    assert!(eng.fingerprint_get(2).is_none());
    assert_eq!(eng.fingerprint_get(3), Some(300));
    assert_eq!(eng.fingerprint_get(4), Some(400));
}

#[test]
fn it_capacity_zero_disables_cache() {
    let dir = tmp_dir("zero");
    let mut eng = Engine::open(&dir).unwrap();
    eng.set_fingerprint_capacity(4);
    eng.fingerprint_put(7, 70);
    eng.set_fingerprint_capacity(0);
    assert!(eng.fingerprint_get(7).is_none());
    assert_eq!(eng.fingerprint_len(), 0);
}
