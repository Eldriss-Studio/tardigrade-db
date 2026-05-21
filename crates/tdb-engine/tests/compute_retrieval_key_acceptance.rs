//! Acceptance tests for `Engine::compute_retrieval_key` and the
//! `RetrievalKeyStrategy` trait dispatch.

use std::path::PathBuf;

use tdb_engine::engine::Engine;

fn tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tdb-rk-at-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn it_returns_none_when_embedding_table_unloaded() {
    let dir = tmp_dir("unloaded");
    let eng = Engine::open(&dir).unwrap();
    let key = eng.compute_retrieval_key(&[1, 2, 3], "last_token").unwrap();
    assert!(key.is_none());
}

#[test]
fn it_dispatches_through_retrieval_key_strategy_trait() {
    // Same input goes through three different strategies and yields
    // three observably different outputs. That's behavioural proof the
    // strategy name routes through different code paths.
    let dir = tmp_dir("dispatch");
    let mut eng = Engine::open(&dir).unwrap();
    let weights = vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    eng.load_embedding_table(weights, 4, 4);

    let token_ids = vec![0i64, 1, 2];

    let last = eng.compute_retrieval_key(&token_ids, "last_token").unwrap().unwrap();
    let mean = eng.compute_retrieval_key(&token_ids, "mean_pool").unwrap().unwrap();

    let projection = vec![
        1.0, 1.0, 1.0, 1.0, //
        2.0, 0.0, 0.0, 0.0,
    ];
    eng.set_projection_matrix(projection, 2, 4);
    let projected = eng.compute_retrieval_key(&token_ids, "projected").unwrap().unwrap();

    assert_eq!(last, vec![0.0, 0.0, 1.0, 0.0]);
    let third = 1.0f32 / 3.0;
    assert!((mean[0] - third).abs() < 1e-6);
    assert_eq!(projected.len(), 2);
    assert!((projected[0] - 1.0).abs() < 1e-6);
    assert!((projected[1] - 0.0).abs() < 1e-6);
}

#[test]
fn it_rejects_unknown_strategy_with_invalid_argument() {
    let dir = tmp_dir("unknown");
    let mut eng = Engine::open(&dir).unwrap();
    eng.load_embedding_table(vec![1.0, 2.0], 1, 2);
    let err = eng.compute_retrieval_key(&[0], "made_up").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("made_up") || msg.contains("strategy"), "got: {msg}");
}

#[test]
fn it_increments_load_count_once_per_call() {
    let dir = tmp_dir("counter");
    let mut eng = Engine::open(&dir).unwrap();
    assert_eq!(eng.embedding_table_load_count(), 0);
    eng.load_embedding_table(vec![1.0; 8], 2, 4);
    assert_eq!(eng.embedding_table_load_count(), 1);
    // Many computes do not bump the counter.
    for _ in 0..25 {
        let _ = eng.compute_retrieval_key(&[0], "last_token").unwrap();
    }
    assert_eq!(eng.embedding_table_load_count(), 1);
    eng.load_embedding_table(vec![2.0; 8], 2, 4);
    assert_eq!(eng.embedding_table_load_count(), 2);
}

#[test]
fn it_filters_negative_and_out_of_range_ids() {
    let dir = tmp_dir("filter");
    let mut eng = Engine::open(&dir).unwrap();
    eng.load_embedding_table(
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0,
        ],
        3,
        2,
    );
    // Last in-range token is id=1 → row [3.0, 4.0].
    let key = eng.compute_retrieval_key(&[-1, 1, 999, 50], "last_token").unwrap().unwrap();
    assert_eq!(key, vec![3.0, 4.0]);
}
