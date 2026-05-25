"""Acceptance tests for the HTTP bridge's confirmed-read surface.

POST /mem/query accepts ``?wait=durable&timeout=<ms>`` query
parameters that map to the engine-side ``mode="confirmed"`` /
``timeout_ms=...`` call. Closes the HTTP-side half of the
confirmed-vs-unconfirmed contract (AT #5 in the plan).

Spins up the bridge in a background uvicorn thread on a free port,
issues real HTTP requests, asserts the documented contract:

- ``?wait=durable&timeout=<ms>``: blocks until durable or returns 504.
- ``?wait=durable`` without timeout: HTTP 400.
- ``?timeout=0`` or above the server cap: HTTP 400.
- No ``?wait`` (default): unchanged behaviour, immediate return.

All 504s are RFC 7807 ``application/problem+json`` with a stable
``type`` URI so consumers can pattern-match without parsing prose.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from contextlib import closing
from urllib.error import HTTPError

import pytest

import tardigrade_db
from tardigrade_http import create_app


def _free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def bridge(tmp_path):
    """Start a uvicorn server in a background thread; tear it down on exit."""
    import uvicorn

    engine = tardigrade_db.Engine.open_with_write_buffer(
        str(tmp_path / "engine"),
        max_batch_size=100,
        max_idle_ms=60000,
    )
    # The bridge needs a key-vector function. For these tests we
    # don't care about retrieval quality — only the wait semantics —
    # so a stub kv_fn returning a fixed-shape key is fine.
    import numpy as np

    def kv_fn(text: str):
        return (
            np.ones(64, dtype=np.float32),
            [(0, np.ones(64, dtype=np.float32))],
        )

    port = _free_port()
    app = create_app(engine, kv_fn=kv_fn)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Poll until the server accepts connections.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with closing(socket.create_connection(("127.0.0.1", port), timeout=0.1)):
                break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("uvicorn server did not start")
    yield (engine, port)
    server.should_exit = True
    thread.join(timeout=5.0)


def _post_query(port: int, body: dict, query: str = "") -> tuple[int, dict | str]:
    """POST to /mem/query, optionally with a query string. Returns (status, body)."""
    url = f"http://127.0.0.1:{port}/mem/query"
    if query:
        url = f"{url}?{query}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body_bytes = exc.read()
        try:
            return exc.code, json.loads(body_bytes.decode("utf-8"))
        except json.JSONDecodeError:
            return exc.code, body_bytes.decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────
# AT #5 — HTTP confirmed-read surface
# ─────────────────────────────────────────────────────────────────────────


class TestDefaultModeIsUnchanged:
    """``POST /mem/query`` with no ``?wait`` param behaves exactly as before."""

    def test_default_query_returns_200_quickly(self, bridge) -> None:
        _engine, port = bridge
        start = time.monotonic()
        status, body = _post_query(
            port,
            body={"owner": 1, "query_text": "anything", "k": 5},
        )
        elapsed = time.monotonic() - start
        assert status == 200
        assert "results" in body
        assert elapsed < 0.5, (
            f"Default-mode HTTP query took {elapsed*1000:.0f}ms — should "
            "return immediately."
        )


class TestConfirmedHttpBlocksAndReturns:
    """``?wait=durable&timeout=<ms>`` blocks the response until flush."""

    def test_confirmed_wait_unblocks_on_flush_within_window(self, bridge) -> None:
        engine, port = bridge
        import numpy as np

        # Buffered write that won't be visible to retrieval until flushed.
        engine.mem_write_pack(
            owner=1,
            retrieval_key=np.ones(64, dtype=np.float32),
            layer_payloads=[(0, np.ones(64, dtype=np.float32))],
            salience=50.0,
        )

        result_holder: list = []

        def issue_confirmed() -> None:
            status, body = _post_query(
                port,
                body={"owner": 1, "query_text": "anything", "k": 5},
                query="wait=durable&timeout=5000",
            )
            result_holder.append((status, body))

        client = threading.Thread(target=issue_confirmed)
        client.start()
        # Give the HTTP request time to land at the engine and enter the wait.
        time.sleep(0.2)
        assert not result_holder, (
            "Confirmed HTTP request returned before flush — "
            "either the bridge didn't honour ?wait=durable or the "
            "write was already durable."
        )
        flush_time = time.monotonic()
        engine.flush_buffer()
        client.join(timeout=3.0)
        unblock_latency = time.monotonic() - flush_time

        assert result_holder, "Confirmed HTTP request never returned."
        status, body = result_holder[0]
        assert status == 200, f"Expected 200, got {status}: {body!r}"
        assert unblock_latency < 1.0, (
            f"HTTP unblock latency {unblock_latency*1000:.0f}ms — should "
            "be sub-second (Condvar wakeup + uvicorn round-trip)."
        )


class TestConfirmedHttpTimeoutReturns504:
    """When the wait exceeds the supplied timeout, return RFC 7807 504."""

    def test_504_with_rfc7807_problem_body_on_timeout(self, bridge) -> None:
        engine, port = bridge
        import numpy as np

        engine.mem_write_pack(
            owner=1,
            retrieval_key=np.ones(64, dtype=np.float32),
            layer_payloads=[(0, np.ones(64, dtype=np.float32))],
            salience=50.0,
        )
        # No flush — the confirmed wait must time out.
        status, body = _post_query(
            port,
            body={"owner": 1, "query_text": "anything", "k": 5},
            query="wait=durable&timeout=100",
        )
        assert status == 504, f"Expected 504, got {status}: {body!r}"
        assert isinstance(body, dict), f"Expected JSON body, got {body!r}"
        # RFC 7807 mandates `type`, `title`, `status` at minimum.
        assert body.get("type") == "tdb:durability:read_timeout", (
            f"Expected stable problem type URI; got {body!r}"
        )
        assert body.get("status") == 504


class TestConfirmedHttpRequiresTimeout:
    """``?wait=durable`` without ``?timeout`` is a programmer error → 400."""

    def test_400_when_wait_durable_has_no_timeout(self, bridge) -> None:
        _engine, port = bridge
        status, body = _post_query(
            port,
            body={"owner": 1, "query_text": "anything", "k": 5},
            query="wait=durable",
        )
        assert status == 400, f"Expected 400, got {status}: {body!r}"
        assert isinstance(body, dict)
        assert body.get("type", "").startswith("tdb:")

    def test_400_when_timeout_is_zero(self, bridge) -> None:
        _engine, port = bridge
        status, _body = _post_query(
            port,
            body={"owner": 1, "query_text": "anything", "k": 5},
            query="wait=durable&timeout=0",
        )
        assert status == 400

    def test_400_when_timeout_exceeds_server_cap(self, bridge) -> None:
        _engine, port = bridge
        # MAX_CONFIRMED_TIMEOUT_MS is 300000 (5 minutes); anything above
        # is rejected to prevent unbounded server-held connections.
        status, _body = _post_query(
            port,
            body={"owner": 1, "query_text": "anything", "k": 5},
            query="wait=durable&timeout=999999",
        )
        assert status == 400
