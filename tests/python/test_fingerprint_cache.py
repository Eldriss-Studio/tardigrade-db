"""Acceptance tests for the engine-side fingerprint LRU cache.

The connector previously held an `OrderedDict[int, int]` capped at
`max_num_seqs`. Phase 9 lifts that into the engine so the cache state
is owned in one place; behavioural contract is unchanged.
"""

import pytest

import tardigrade_db


@pytest.fixture
def engine(tmp_path):
    eng = tardigrade_db.Engine(str(tmp_path / "engine"))
    eng.set_fingerprint_capacity(4)
    return eng


def test_get_returns_none_for_unknown_fingerprint(engine):
    assert engine.fingerprint_get(12345) is None


def test_put_then_get_round_trip(engine):
    engine.fingerprint_put(7, 101)
    assert engine.fingerprint_get(7) == 101


def test_insert_beyond_capacity_evicts_oldest(engine):
    # capacity=4; insert 5 → first one (1) gets dropped
    for fp, pid in [(1, 100), (2, 200), (3, 300), (4, 400), (5, 500)]:
        engine.fingerprint_put(fp, pid)
    assert engine.fingerprint_get(1) is None  # evicted
    assert engine.fingerprint_get(2) == 200
    assert engine.fingerprint_get(5) == 500


def test_get_promotes_entry_to_most_recent(engine):
    for fp, pid in [(1, 100), (2, 200), (3, 300), (4, 400)]:
        engine.fingerprint_put(fp, pid)
    # touch fp=1 so it becomes most-recent; then insert fp=5 → fp=2 evicts
    assert engine.fingerprint_get(1) == 100
    engine.fingerprint_put(5, 500)
    assert engine.fingerprint_get(2) is None
    assert engine.fingerprint_get(1) == 100


def test_release_removes_specific_entry(engine):
    engine.fingerprint_put(11, 110)
    engine.fingerprint_put(12, 120)
    engine.fingerprint_release(11)
    assert engine.fingerprint_get(11) is None
    assert engine.fingerprint_get(12) == 120


def test_release_unknown_fingerprint_is_noop(engine):
    engine.fingerprint_release(99999)  # must not raise


def test_repeated_put_updates_pack_id_without_growing_size(engine):
    engine.fingerprint_put(3, 30)
    engine.fingerprint_put(3, 31)
    engine.fingerprint_put(3, 32)
    assert engine.fingerprint_get(3) == 32
    # cache still has room for 3 more entries
    for fp, pid in [(10, 100), (11, 110), (12, 120)]:
        engine.fingerprint_put(fp, pid)
    assert engine.fingerprint_get(3) is not None  # not evicted


def test_capacity_is_configurable(tmp_path):
    eng = tardigrade_db.Engine(str(tmp_path / "engine"))
    eng.set_fingerprint_capacity(2)
    eng.fingerprint_put(1, 10)
    eng.fingerprint_put(2, 20)
    eng.fingerprint_put(3, 30)
    assert eng.fingerprint_get(1) is None  # evicted
    assert eng.fingerprint_get(3) == 30


def test_block_id_reused_by_new_request_does_not_return_old_pack(engine):
    """The cache is keyed by fingerprint (= block_id). When a request
    finishes and a new request reuses the same block_id, the connector
    must release first; lookup after release returns None even if the
    new request has the same fingerprint."""
    engine.fingerprint_put(42, 999)
    engine.fingerprint_release(42)
    assert engine.fingerprint_get(42) is None
