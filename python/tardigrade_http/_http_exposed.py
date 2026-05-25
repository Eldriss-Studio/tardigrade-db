"""Registry of HTTP routes ↔ engine methods ↔ Pydantic models.

This is the cross-surface contract surface for the HTTP bridge. Every
route declared in :mod:`tardigrade_http.server` has a corresponding
entry here naming:

- The :class:`tardigrade_db.Engine` method(s) the route depends on at
  runtime. A renamed or removed engine method causes the parity test
  in ``tests/python/test_pyo3_http_parity.py`` to fail, surfacing the
  drift before it reaches a consumer.
- The Pydantic request and response models the route binds. Renamed
  or removed models likewise fail the parity test.

This registry is transition scaffolding. The architectural goal — see
``~/.claude/plans/contract-source-of-truth-foundation.md`` §"What this
unlocks" — is to *generate* ``models.py`` from PyO3 attributes so the
registry becomes redundant and goes away. Until that lands, this is
the explicit source of truth for what the HTTP surface covers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteContract:
    """One row of the HTTP↔engine↔models contract."""

    # `tardigrade_db.Engine` methods (instance or classmethod) the route
    # invokes. The parity test asserts each exists.
    engine_methods: tuple[str, ...]
    # Pydantic model class names in ``tardigrade_http.models``. The
    # parity test asserts each exists.
    request_model: str | None
    response_model: str | None
    # Extra Pydantic models the response embeds (e.g. nested types).
    # Same existence check as the request/response pair.
    nested_models: tuple[str, ...] = ()


# Keyed by ``"<METHOD> <path>"`` for human-readable failure messages.
HTTP_EXPOSED: dict[str, RouteContract] = {
    "POST /mem/store": RouteContract(
        engine_methods=("mem_write_pack",),
        request_model="StoreRequest",
        response_model="StoreResponse",
    ),
    "POST /mem/query": RouteContract(
        engine_methods=("mem_read_pack",),
        request_model="QueryRequest",
        response_model="QueryResponse",
        nested_models=("QueryResult",),
    ),
    "GET /mem/owners": RouteContract(
        engine_methods=("list_owners",),
        request_model=None,
        response_model="OwnersResponse",
    ),
    "GET /mem/status": RouteContract(
        engine_methods=("status",),
        request_model=None,
        response_model="StatusResponse",
    ),
    "POST /mem/save": RouteContract(
        engine_methods=("snapshot",),
        request_model="SaveRequest",
        response_model="SaveResponse",
        nested_models=("Manifest",),
    ),
    "GET /metrics": RouteContract(
        # Returns the Prometheus text exposition format. No request
        # body, no Pydantic response model — `PlainTextResponse`
        # returns the engine's render method output verbatim. The
        # parity registry only checks the engine method exists.
        engine_methods=("metrics_prometheus_text",),
        request_model=None,
        response_model=None,
    ),
    "POST /mem/restore": RouteContract(
        # ``restore_from`` is a classmethod / staticmethod-style
        # constructor on ``tardigrade_db.Engine`` rather than an
        # instance method, but it still resolves as an attribute on
        # the class object — the parity test treats both uniformly.
        engine_methods=("restore_from", "pack_count"),
        request_model="RestoreRequest",
        response_model="RestoreResponse",
    ),
}
