# ATDD acceptance tests for Path 2: MemoryPrefixBuilder.
#
# Design pattern: Facade (MemoryPrefixBuilder wraps engine enumeration +
# governance filtering + format composition).
#
# Uses the Rust engine directly — no model needed.

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.prefix_builder import MemoryPrefixBuilder, PrefixResult
from tardigrade_hooks.prefix_format import BulletListFormat, TierAnnotatedFormat

KEY_DIM = 16


@pytest.fixture
def engine(tmp_path):
    return tardigrade_db.Engine(str(tmp_path))


def _store(engine, text, owner=1, salience=80.0):
    """Store a pack with the given text. salience=80 → Core, 60 → Validated, 20 → Draft."""
    key = np.random.randn(KEY_DIM).astype(np.float32)
    payload = np.zeros(KEY_DIM, dtype=np.float32)
    return engine.mem_write_pack(owner, key, [(0, payload)], salience, text)


# -- 1: empty engine ----------------------------------------------------------


def test_empty_engine_returns_empty_prefix(engine):
    builder = MemoryPrefixBuilder(engine, owner=1)
    result = builder.build()
    assert result.text == ""
    assert result.pack_ids == []
    assert result.token_estimate >= 0


# -- 2: core memories included ------------------------------------------------


def test_core_memories_included(engine):
    _store(engine, "Zyphlox-9 is the capital", salience=80.0)
    builder = MemoryPrefixBuilder(engine, owner=1)
    result = builder.build()
    assert "Zyphlox-9 is the capital" in result.text
    assert len(result.pack_ids) == 1


# -- 3: draft memories excluded -----------------------------------------------


def test_draft_memories_excluded(engine):
    _store(engine, "draft fact", salience=20.0)
    _store(engine, "core fact", salience=80.0)
    builder = MemoryPrefixBuilder(engine, owner=1)
    result = builder.build()
    assert "core fact" in result.text
    assert "draft fact" not in result.text
    assert len(result.pack_ids) == 1


# -- 4: validated memories optional -------------------------------------------


def test_validated_memories_included_when_enabled(engine):
    _store(engine, "validated fact", salience=60.0)
    builder_with = MemoryPrefixBuilder(engine, owner=1, include_validated=True)
    builder_without = MemoryPrefixBuilder(engine, owner=1, include_validated=False)
    assert "validated fact" in builder_with.build().text
    assert "validated fact" not in builder_without.build().text


# -- 5: deterministic ---------------------------------------------------------


def test_prefix_is_deterministic(engine):
    _store(engine, "fact A", salience=80.0)
    _store(engine, "fact B", salience=80.0)
    builder = MemoryPrefixBuilder(engine, owner=1)
    r1 = builder.build()
    r2 = builder.build()
    assert r1.text == r2.text
    assert r1.version == r2.version


# -- 6: ordered by importance -------------------------------------------------


def test_prefix_ordered_by_importance(engine):
    pid_low = _store(engine, "low importance", salience=80.0)
    pid_high = _store(engine, "high importance", salience=80.0)
    # Boost the second pack's importance by accessing it.
    for _ in range(10):
        engine.load_pack_by_id(pid_high)

    builder = MemoryPrefixBuilder(engine, owner=1)
    result = builder.build()
    assert result.pack_ids[0] == pid_high


# -- 7: token budget ----------------------------------------------------------


def test_token_budget_truncates(engine):
    for i in range(20):
        _store(engine, f"memory number {i} with enough words to use tokens", salience=80.0)

    builder_unlimited = MemoryPrefixBuilder(engine, owner=1)
    builder_limited = MemoryPrefixBuilder(engine, owner=1, token_budget=50)

    r_all = builder_unlimited.build()
    r_limited = builder_limited.build()
    assert len(r_limited.pack_ids) < len(r_all.pack_ids)
    assert r_limited.token_estimate <= 50


# -- 8: version increments on change ------------------------------------------


def test_version_increments_on_change(engine):
    _store(engine, "fact one", salience=80.0)
    builder = MemoryPrefixBuilder(engine, owner=1)
    v1 = builder.build().version

    _store(engine, "fact two", salience=80.0)
    v2 = builder.build().version
    assert v1 != v2


# -- 9: version stable when unchanged ----------------------------------------


def test_version_stable_when_unchanged(engine):
    _store(engine, "stable fact", salience=80.0)
    builder = MemoryPrefixBuilder(engine, owner=1)
    v1 = builder.build().version
    v2 = builder.build().version
    assert v1 == v2


# -- 10: format strategy swappable -------------------------------------------


def test_format_strategy_swappable(engine):
    _store(engine, "some fact", salience=80.0)
    bullet = MemoryPrefixBuilder(engine, owner=1, format=BulletListFormat())
    tier = MemoryPrefixBuilder(engine, owner=1, format=TierAnnotatedFormat())
    r_bullet = bullet.build()
    r_tier = tier.build()
    assert r_bullet.text != r_tier.text
    assert "some fact" in r_bullet.text
    assert "[Core]" in r_tier.text


# -- 11: newlines in fact text escaped ----------------------------------------


def test_newlines_in_fact_text_escaped(engine):
    _store(engine, "line one\nline two\nline three", salience=80.0)
    builder = MemoryPrefixBuilder(engine, owner=1)
    result = builder.build()
    lines = result.text.split("\n")
    for line in lines[1:]:
        assert line.startswith("- "), f"malformed line: {line}"
