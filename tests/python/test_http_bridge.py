"""ATDD: HTTP/REST bridge.

The bridge is an Adapter over :class:`tardigrade_db.Engine` that
exposes the foundation surface (store/query/owners/status/snapshot/
restore) over HTTP. Pydantic models drive request validation and
OpenAPI generation; FastAPI's TestClient drives the integration
tests below.

Test classes:

- ``TestOpenAPISchema`` — route surface + problem+json content-type
  on validation errors.
- ``TestStoreAndQuery`` / ``TestOwners`` / ``TestStatus`` /
  ``TestSnapshotRestore`` — full parity with the direct Python
  surface.
- ``TestErrorResponses`` — RFC 7807 envelope on validation failures
  and engine errors.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_and_engine(tmp_path):
    import tardigrade_db

    from tardigrade_http.server import create_app

    engine = tardigrade_db.Engine(str(tmp_path / "engine"))
    app = create_app(engine)
    return app, engine


@pytest.fixture
def client(app_and_engine):
    app, _ = app_and_engine
    return TestClient(app)


class TestOpenAPISchema:
    def test_exposes_all_required_routes(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        for route in (
            "/mem/store",
            "/mem/query",
            "/mem/owners",
            "/mem/status",
            "/mem/save",
            "/mem/restore",
        ):
            assert route in paths, f"missing {route} in OpenAPI paths"

    def test_advertises_problem_json_for_validation_errors(self, client):
        r = client.post("/mem/store", json={"owner": -1, "fact_text": "x"})
        assert r.status_code == 422
        assert r.headers["content-type"].startswith("application/problem+json")


class TestStoreAndQuery:
    def test_store_returns_pack_id(self, client):
        r = client.post(
            "/mem/store",
            json={"owner": 1, "fact_text": "hello world"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "pack_id" in body
        assert isinstance(body["pack_id"], int)

    def test_query_retrieves_stored_fact(self, client):
        text = "the sun is bright today"
        r1 = client.post("/mem/store", json={"owner": 1, "fact_text": text})
        assert r1.status_code == 200
        pid = r1.json()["pack_id"]

        r2 = client.post(
            "/mem/query",
            json={"owner": 1, "query_text": text, "k": 5},
        )
        assert r2.status_code == 200
        results = r2.json()["results"]
        assert any(r["pack_id"] == pid for r in results)

    def test_query_respects_owner_scope(self, client):
        client.post("/mem/store", json={"owner": 1, "fact_text": "alpha"})
        client.post("/mem/store", json={"owner": 7, "fact_text": "beta"})
        r = client.post(
            "/mem/query",
            json={"owner": 1, "query_text": "alpha", "k": 5},
        )
        assert r.status_code == 200
        for res in r.json()["results"]:
            # Owner 7's pack should never appear in an owner-1 query.
            assert "beta" not in (res.get("text") or "")


class TestOwners:
    def test_lists_owners_after_stores(self, client):
        for o in (1, 7, 42):
            client.post(
                "/mem/store",
                json={"owner": o, "fact_text": f"owner-{o} fact"},
            )
        r = client.get("/mem/owners")
        assert r.status_code == 200
        assert sorted(r.json()["owners"]) == [1, 7, 42]

    def test_empty_engine_returns_empty_owners(self, client):
        r = client.get("/mem/owners")
        assert r.status_code == 200
        assert r.json()["owners"] == []


class TestStatus:
    def test_returns_engine_state_fields(self, client):
        client.post("/mem/store", json={"owner": 1, "fact_text": "x"})
        r = client.get("/mem/status")
        assert r.status_code == 200
        body = r.json()
        for k in (
            "cell_count",
            "pack_count",
            "segment_count",
            "slb_occupancy",
            "slb_capacity",
            "vamana_active",
            "pipeline_stages",
            "governance_entries",
            "trace_edges",
            "arena_bytes",
            "arena_bytes_per_cell",
        ):
            assert k in body, f"missing {k} in status response"


class TestSnapshotRestore:
    def test_save_and_restore_roundtrip(self, client, tmp_path):
        for o in (1, 7, 42):
            client.post(
                "/mem/store",
                json={"owner": o, "fact_text": f"fact-{o}"},
            )

        snap_path = tmp_path / "snap.tar"
        r = client.post(
            "/mem/save",
            json={"snapshot_path": str(snap_path)},
        )
        assert r.status_code == 200
        manifest = r.json()["manifest"]
        assert manifest["magic"] == "tdb!"
        assert manifest["pack_count"] == 3
        assert manifest["owner_count"] == 3

        target = tmp_path / "restored"
        r = client.post(
            "/mem/restore",
            json={
                "snapshot_path": str(snap_path),
                "target_dir": str(target),
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["pack_count"] == 3


class TestCheckedInSchema:
    """The exported ``schema.yaml`` is the public contract.

    If anyone changes a route, request body, or response shape in
    ``server.py`` / ``models.py`` without re-running
    ``python -m tardigrade_http.export_schema``, this test fails
    loud — catching drift between the running code and the
    committed contract before consumers see it.
    """

    def test_checked_in_schema_matches_live_app(self, client):
        from pathlib import Path

        import yaml

        live = client.get("/openapi.json").json()
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "python"
            / "tardigrade_http"
            / "schema.yaml"
        )
        checked_in = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        assert live == checked_in, (
            "Live OpenAPI schema drifted from checked-in schema.yaml. "
            "Run `python -m tardigrade_http.export_schema` to regenerate."
        )


class TestErrorResponses:
    def test_validation_error_uses_problem_json_envelope(self, client):
        r = client.post(
            "/mem/store",
            json={"owner": "not-an-int", "fact_text": "x"},
        )
        assert r.status_code == 422
        assert r.headers["content-type"].startswith("application/problem+json")
        body = r.json()
        assert body["status"] == 422
        assert body["title"]
        assert "detail" in body

    def test_restore_corrupt_snapshot_returns_problem_json(self, client, tmp_path):
        bogus = tmp_path / "bogus.tar"
        bogus.write_bytes(b"definitely not a snapshot")
        r = client.post(
            "/mem/restore",
            json={
                "snapshot_path": str(bogus),
                "target_dir": str(tmp_path / "restored"),
            },
        )
        assert r.status_code == 400
        assert r.headers["content-type"].startswith("application/problem+json")
        body = r.json()
        assert body["status"] == 400
