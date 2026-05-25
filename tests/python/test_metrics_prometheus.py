"""Acceptance tests for the Prometheus-format metrics layer.

Closes the last item on the SpacetimeDB research report's §5.1
"Adopt Now" list: startup/replay metrics + general operational
visibility. Engine emits metrics into a process-global Prometheus
recorder; ``engine.metrics_prometheus_text()`` renders the current
state into the Prometheus text exposition format; the HTTP bridge
serves it at ``GET /metrics`` so any Prometheus scraper can consume.

The tests use absolute presence + relative-delta assertions: since
the recorder is process-global, individual test runs accumulate
state. We check that metric families exist and that counters change
in the expected direction, not that they hit specific absolute
values.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.request
from contextlib import closing

import numpy as np
import pytest

import tardigrade_db


KEY_DIM = 64


def _key(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(KEY_DIM, dtype=np.float32)


def _layer_payload(seed: int) -> list[tuple[int, np.ndarray]]:
    return [(0, _key(seed + 1000))]


def _parse_prometheus_text(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Tiny Prometheus text-format parser.

    Returns ``{family_name: [(labels_dict, value), ...]}``. Good
    enough for our ATs — we only need to find a metric by name and
    inspect its value. A real consumer would use
    ``prometheus_client.parser.text_string_to_metric_families``.
    """
    families: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Format: <name>{labels} <value>  OR  <name> <value>
        # Labels are optional. Value may be int or float.
        if "{" in line:
            name, rest = line.split("{", 1)
            labels_str, value_str = rest.rsplit("}", 1)
            labels: dict[str, str] = {}
            for kv in labels_str.split(","):
                kv = kv.strip()
                if not kv:
                    continue
                k, v = kv.split("=", 1)
                labels[k.strip()] = v.strip().strip('"')
            value = float(value_str.strip())
        else:
            parts = line.split()
            name, value_str = parts[0], parts[-1]
            labels = {}
            value = float(value_str)
        families.setdefault(name, []).append((labels, value))
    return families


# ─────────────────────────────────────────────────────────────────────────
# Engine-side metrics rendering
# ─────────────────────────────────────────────────────────────────────────


class TestMetricsTextRenders:
    """``engine.metrics_prometheus_text()`` returns parseable output."""

    def test_render_returns_non_empty_string(self, tmp_path) -> None:
        engine = tardigrade_db.Engine(str(tmp_path / "engine"))
        text = engine.metrics_prometheus_text()
        assert isinstance(text, str)
        assert text, "metrics text should not be empty"

    def test_render_is_prometheus_parseable(self, tmp_path) -> None:
        engine = tardigrade_db.Engine(str(tmp_path / "engine"))
        # Do some work so something gets emitted.
        engine.mem_write_pack(
            owner=1,
            retrieval_key=_key(0),
            layer_payloads=_layer_payload(0),
            salience=50.0,
        )
        engine.flush()
        text = engine.metrics_prometheus_text()
        families = _parse_prometheus_text(text)
        # At minimum, the durable_offset gauge should exist after a
        # write+flush cycle. Other families may or may not depending
        # on which paths were exercised.
        assert "tdb_durable_offset" in families, (
            f"tdb_durable_offset gauge missing from render. Got families: "
            f"{sorted(families.keys())}"
        )


class TestDurableOffsetGauge:
    """``tdb_durable_offset`` gauge tracks the engine's durable offset."""

    def test_gauge_advances_after_writes(self, tmp_path) -> None:
        engine = tardigrade_db.Engine(str(tmp_path / "engine"))
        # Three writes, each fsynced immediately on the non-buffered path.
        for i in range(3):
            engine.mem_write_pack(
                owner=1,
                retrieval_key=_key(i),
                layer_payloads=_layer_payload(i),
                salience=50.0,
            )
            engine.flush()
        text = engine.metrics_prometheus_text()
        families = _parse_prometheus_text(text)
        rows = families.get("tdb_durable_offset", [])
        assert rows, "tdb_durable_offset missing after writes"
        # Last value should match the engine's actual durable_offset.
        gauge_value = rows[-1][1]
        actual = float(engine.durable_offset())
        assert gauge_value == actual, (
            f"gauge value {gauge_value} != engine.durable_offset() {actual}"
        )


class TestConfirmedReadCounters:
    """``tdb_confirmed_read_total`` counts confirmed-mode outcomes."""

    def test_ok_outcome_counter_bumps_on_success(self, tmp_path) -> None:
        engine = tardigrade_db.Engine.open_with_write_buffer(
            str(tmp_path / "engine"),
            max_batch_size=100,
            max_idle_ms=60000,
        )
        engine.mem_write_pack(
            owner=1,
            retrieval_key=_key(0),
            layer_payloads=_layer_payload(0),
            salience=50.0,
        )

        before_text = engine.metrics_prometheus_text()
        before = _confirmed_read_count(before_text, "ok")

        # Issue a confirmed read in a thread, flush from main.
        result_holder: list = []

        def reader() -> None:
            result_holder.append(
                engine.mem_read_pack(
                    _key(0), k=5, owner=1,
                    mode="confirmed", timeout_ms=5000,
                )
            )

        thread = threading.Thread(target=reader)
        thread.start()
        time.sleep(0.05)
        engine.flush_buffer()
        thread.join(timeout=2.0)
        assert result_holder, "confirmed reader didn't return"

        after_text = engine.metrics_prometheus_text()
        after = _confirmed_read_count(after_text, "ok")
        assert after == before + 1, (
            f"tdb_confirmed_read_total{{outcome=\"ok\"}} expected to "
            f"increase by 1, went from {before} to {after}"
        )

    def test_timeout_outcome_counter_bumps_on_timeout(self, tmp_path) -> None:
        engine = tardigrade_db.Engine.open_with_write_buffer(
            str(tmp_path / "engine"),
            max_batch_size=100,
            max_idle_ms=60000,
        )
        engine.mem_write_pack(
            owner=1,
            retrieval_key=_key(0),
            layer_payloads=_layer_payload(0),
            salience=50.0,
        )

        before = _confirmed_read_count(engine.metrics_prometheus_text(), "timeout")

        with pytest.raises(Exception):
            engine.mem_read_pack(
                _key(0), k=5, owner=1,
                mode="confirmed", timeout_ms=50,
            )

        after = _confirmed_read_count(engine.metrics_prometheus_text(), "timeout")
        assert after == before + 1, (
            f"tdb_confirmed_read_total{{outcome=\"timeout\"}} expected to "
            f"increase by 1, went from {before} to {after}"
        )


def _confirmed_read_count(text: str, outcome: str) -> float:
    """Look up ``tdb_confirmed_read_total{outcome=...}`` in rendered text."""
    families = _parse_prometheus_text(text)
    rows = families.get("tdb_confirmed_read_total", [])
    for labels, value in rows:
        if labels.get("outcome") == outcome:
            return value
    return 0.0


# ─────────────────────────────────────────────────────────────────────────
# HTTP /metrics endpoint
# ─────────────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def bridge(tmp_path):
    import uvicorn
    from tardigrade_http import create_app

    engine = tardigrade_db.Engine(str(tmp_path / "engine"))

    def kv_fn(text: str):
        return (
            np.ones(KEY_DIM, dtype=np.float32),
            [(0, np.ones(KEY_DIM, dtype=np.float32))],
        )

    port = _free_port()
    app = create_app(engine, kv_fn=kv_fn)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
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


class TestHttpMetricsEndpoint:
    """``GET /metrics`` returns the engine's rendered Prometheus text."""

    def test_metrics_endpoint_returns_200(self, bridge) -> None:
        _engine, port = bridge
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5.0) as resp:
            assert resp.status == 200
            content_type = resp.headers.get("Content-Type", "")
            assert content_type.startswith("text/plain"), (
                f"Prometheus scrapers expect text/plain; got {content_type!r}"
            )
            body = resp.read().decode("utf-8")
            assert "tdb_durable_offset" in body, (
                "metrics body should include tdb_durable_offset gauge"
            )
