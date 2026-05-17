"""ATDD: Builder pattern for ``TardigradeClient``.

Today's constructor is positional/opaque: ``TardigradeClient(db_path,
tokenizer=..., owner=..., kv_capture_fn=..., vamana_threshold=...)``.
Adding many optional knobs without forcing every consumer to re-read
the constructor signature is what the builder buys.

ATs:

- builder builds with only the required field set
- missing required field raises a typed ``BuilderIncomplete``
  with the missing field named in the message
- chained setters return the builder (fluent contract)
- optional fields fall back to documented defaults
- builder produces a working client (round-trip store/query)
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_engine_dir(tmp_path: Path) -> Path:
    return tmp_path / "engine"


class TestTardigradeClientBuilder:
    def test_builder_returns_builder_instance(self):
        from tardigrade_hooks import TardigradeClient
        builder = TardigradeClient.builder()
        assert builder is not None
        # The builder should have a ``build`` method.
        assert callable(getattr(builder, "build", None))

    def test_chained_setters_return_builder_for_fluent_use(self, tmp_engine_dir):
        from tardigrade_hooks import TardigradeClient
        b = TardigradeClient.builder()
        # Each setter returns ``self`` so calls can chain.
        assert b.with_engine_dir(tmp_engine_dir) is b
        assert b.with_owner(7) is b

    def test_build_succeeds_with_only_required_fields(self, tmp_engine_dir):
        from tardigrade_hooks import TardigradeClient
        client = (
            TardigradeClient.builder()
            .with_engine_dir(tmp_engine_dir)
            .build()
        )
        assert client is not None
        # The built client exposes the same surface as the
        # positional constructor.
        assert callable(getattr(client, "store", None))
        assert callable(getattr(client, "query", None))

    def test_build_without_engine_dir_raises_typed_error(self):
        from tardigrade_hooks import TardigradeClient
        from tardigrade_hooks.builder import BuilderIncomplete
        with pytest.raises(BuilderIncomplete) as exc:
            TardigradeClient.builder().build()
        assert "engine_dir" in str(exc.value)

    def test_optional_fields_fall_back_to_defaults(self, tmp_engine_dir):
        from tardigrade_hooks import TardigradeClient
        client = (
            TardigradeClient.builder()
            .with_engine_dir(tmp_engine_dir)
            .build()
        )
        # Owner defaults to 1 (mirrors the positional ctor); cell
        # count starts at 0 on a fresh engine.
        assert client._owner == 1
        assert client.pack_count() == 0

    def test_with_engine_shares_state_across_clients(self, tmp_engine_dir):
        """Multiple clients can share one underlying ``Engine``.

        Critical for multi-agent demos where N owners need one
        engine — opening the same ``engine_dir`` from N separate
        ``Engine`` instances produces N isolated states, not one
        shared one. ``with_engine`` lets the consumer build one
        engine and inject it into each client.
        """
        from tardigrade_hooks import TardigradeClient
        import tardigrade_db

        engine = tardigrade_db.Engine(str(tmp_engine_dir))
        client_a = (
            TardigradeClient.builder()
            .with_engine(engine)
            .with_owner(1)
            .build()
        )
        client_b = (
            TardigradeClient.builder()
            .with_engine(engine)
            .with_owner(2)
            .build()
        )
        client_a.store("fact for owner 1")
        client_b.store("fact for owner 2")
        # The shared engine sees both owners.
        assert sorted(engine.list_owners()) == [1, 2]

    def test_with_engine_and_with_engine_dir_are_mutually_exclusive(
        self, tmp_engine_dir,
    ):
        from tardigrade_hooks import TardigradeClient
        from tardigrade_hooks.builder import BuilderIncomplete
        import tardigrade_db

        engine = tardigrade_db.Engine(str(tmp_engine_dir))
        with pytest.raises(BuilderIncomplete) as exc:
            (
                TardigradeClient.builder()
                .with_engine(engine)
                .with_engine_dir(tmp_engine_dir)
                .build()
            )
        msg = str(exc.value).lower()
        assert "engine_dir" in msg and "engine" in msg

    def test_builder_round_trip_stores_and_queries(self, tmp_engine_dir):
        from tardigrade_hooks import TardigradeClient
        client = (
            TardigradeClient.builder()
            .with_engine_dir(tmp_engine_dir)
            .with_owner(42)
            .build()
        )
        text = "the builder works end to end"
        pid = client.store(text)
        results = client.query(text, k=5)
        assert any(r["pack_id"] == pid for r in results)
