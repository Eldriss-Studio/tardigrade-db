"""FastAPI app factory for the TardigradeDB HTTP bridge.

Pattern: **Adapter**. The bridge maps each REST endpoint to one
method on :class:`tardigrade_db.Engine`, normalising error shapes
to RFC 7807 (``application/problem+json``) on the way out.

Why a factory (``create_app``) rather than a module-level ``app``?
Tests construct a fresh engine + app per test in a ``tmp_path``;
the factory keeps that ergonomic without resorting to global state
or lifespan-shaped hacks.

Error handling
--------------

FastAPI's default validation envelope is JSON but advertises
``application/json``. The bridge re-wraps it (and any
:class:`HTTPException`) as a problem-detail envelope so clients
can rely on a single error shape across all failures — see
``models.ProblemDetail``.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import tardigrade_db

from .constants import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    DEFAULT_DB_PATH,
    DEFAULT_HOST,
    DEFAULT_PORT,
    ENV_DB_PATH,
    ENV_HOST,
    ENV_PORT,
    HASH_SEED_MODULUS,
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_GATEWAY_TIMEOUT,
    HTTP_STATUS_VALIDATION_ERROR,
    MAX_CONFIRMED_TIMEOUT_MS,
    PROBLEM_JSON_CONTENT_TYPE,
    PROBLEM_TYPE_ABOUT_BLANK,
    PROBLEM_TYPE_INVALID_REQUEST,
    PROBLEM_TYPE_READ_TIMEOUT,
    STUB_KEY_DIM,
    STUB_LAYER_INDEX,
    STUB_VALUE_DIM,
    WAIT_DURABLE,
)
from .models import (
    Manifest,
    OwnersResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    RestoreRequest,
    RestoreResponse,
    SaveRequest,
    SaveResponse,
    StatusResponse,
    StoreRequest,
    StoreResponse,
)

KvCaptureFn = Callable[[str], tuple]


def _default_kv_fn(text: str) -> tuple:
    """Deterministic hash-based KV stub.

    Same text → same key, so store + query round-trip without a
    real LM. Production deployments inject a real capture function
    (e.g. ``tardigrade_hooks.encoding.encode_per_token``) via
    :func:`create_app`'s ``kv_fn`` parameter.
    """
    rng = np.random.default_rng(abs(hash(text)) % HASH_SEED_MODULUS)
    key = rng.standard_normal(STUB_KEY_DIM).astype(np.float32)
    value = rng.standard_normal(STUB_VALUE_DIM).astype(np.float32)
    return key, [(STUB_LAYER_INDEX, value)]


def _problem_response(
    status: int,
    title: str,
    detail: str | None = None,
    *,
    type_uri: str = PROBLEM_TYPE_ABOUT_BLANK,
) -> JSONResponse:
    """Build an RFC 7807 problem-detail JSON response.

    ``type_uri`` defaults to ``about:blank`` for generic errors but
    callers handling specific failure classes (e.g. durability
    timeout) pass a stable ``tdb:`` URI so consumers can
    pattern-match without parsing prose.
    """
    return JSONResponse(
        status_code=status,
        media_type=PROBLEM_JSON_CONTENT_TYPE,
        content={
            "type": type_uri,
            "title": title,
            "status": status,
            "detail": detail,
        },
    )


def _parse_confirmed_read_params(
    wait: str | None,
    timeout: int | None,
) -> tuple[bool, int | None] | JSONResponse:
    """Validate the ``?wait`` / ``?timeout`` query params for ``/mem/query``.

    Returns either ``(confirmed_mode: bool, timeout_ms: int | None)``
    on success or a problem-detail ``JSONResponse`` when the params
    are inconsistent. Confirmed mode requires ``wait="durable"`` AND
    a ``timeout`` in ``[1, MAX_CONFIRMED_TIMEOUT_MS]``; anything else
    is rejected with HTTP 400 + a stable ``tdb:`` problem-type URI.
    """
    if wait is None:
        # Default: unconfirmed. ``?timeout=`` alone is harmless.
        return (False, None)
    if wait != WAIT_DURABLE:
        return _problem_response(
            HTTP_STATUS_BAD_REQUEST,
            "Invalid wait mode",
            detail=(
                f"wait must be {WAIT_DURABLE!r} or absent; got {wait!r}"
            ),
            type_uri=PROBLEM_TYPE_INVALID_REQUEST,
        )
    if timeout is None:
        return _problem_response(
            HTTP_STATUS_BAD_REQUEST,
            "Confirmed read requires a timeout",
            detail=(
                "wait=durable must be paired with timeout=<ms>; a confirmed "
                "read without a deadline can block indefinitely if the "
                "underlying write never reaches durability"
            ),
            type_uri=PROBLEM_TYPE_INVALID_REQUEST,
        )
    if timeout <= 0 or timeout > MAX_CONFIRMED_TIMEOUT_MS:
        return _problem_response(
            HTTP_STATUS_BAD_REQUEST,
            "Timeout out of range",
            detail=(
                f"timeout must be in [1, {MAX_CONFIRMED_TIMEOUT_MS}] ms; "
                f"got {timeout}"
            ),
            type_uri=PROBLEM_TYPE_INVALID_REQUEST,
        )
    return (True, timeout)


def create_app(engine, kv_fn: KvCaptureFn | None = None) -> FastAPI:
    """Construct a FastAPI app bound to ``engine``.

    ``engine`` is held by reference; the caller retains ownership
    of its lifecycle. ``kv_fn`` defaults to a deterministic hash
    stub — adequate for tests and the OpenAPI surface, but real
    deployments should pass a model-backed capture function.
    """
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
    )
    app.state.engine = engine
    app.state.kv_fn = kv_fn or _default_kv_fn

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(_request: Request, exc: RequestValidationError):
        return _problem_response(
            HTTP_STATUS_VALIDATION_ERROR,
            "Validation error",
            detail=str(exc.errors()),
        )

    @app.exception_handler(HTTPException)
    async def _on_http_error(_request: Request, exc: HTTPException):
        title = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        return _problem_response(exc.status_code, title, detail=title)

    @app.post("/mem/store", response_model=StoreResponse)
    def store(req: StoreRequest) -> StoreResponse:
        key, layers = app.state.kv_fn(req.fact_text)
        pack_id = engine.mem_write_pack(
            req.owner, key, layers, req.salience, req.fact_text,
        )
        return StoreResponse(pack_id=pack_id)

    @app.post("/mem/query")
    def query(
        req: QueryRequest,
        wait: str | None = None,
        timeout: int | None = None,
    ):
        # Confirmed-read contract: `?wait=durable&timeout=<ms>` opts
        # in to durability-blocking semantics; bare POST is the legacy
        # unconfirmed path (immediate return). See
        # docs/guide/performance-contracts.md and the SpacetimeDB-
        # research §8 status update for the rationale.
        parsed = _parse_confirmed_read_params(wait, timeout)
        if isinstance(parsed, JSONResponse):
            return parsed
        confirmed_mode, timeout_ms = parsed

        key, _ = app.state.kv_fn(req.query_text)
        try:
            if confirmed_mode:
                rows = engine.mem_read_pack(
                    key,
                    req.k,
                    req.owner,
                    mode="confirmed",
                    timeout_ms=timeout_ms,
                )
            else:
                rows = engine.mem_read_pack(key, req.k, req.owner)
        except RuntimeError as exc:
            # The PyO3 binding raises `RuntimeError` for the
            # confirmed-read timeout. Pattern-matching on the message
            # is brittle but the alternative (a typed Python
            # exception from PyO3) is a deeper refactor; this is the
            # smallest change that closes the HTTP contract today.
            if "confirmed read timeout" in str(exc).lower():
                return _problem_response(
                    HTTP_STATUS_GATEWAY_TIMEOUT,
                    "Confirmed read exceeded timeout",
                    detail=str(exc),
                    type_uri=PROBLEM_TYPE_READ_TIMEOUT,
                )
            raise

        # `mem_read_pack` always populates the `text` field — it's the
        # pack's stored text or None. No secondary `pack_text` fetch.
        return QueryResponse(
            results=[
                QueryResult(
                    pack_id=int(r["pack_id"]),
                    score=float(r["score"]),
                    text=r["text"],
                )
                for r in rows
            ],
        )

    @app.get("/mem/owners", response_model=OwnersResponse)
    def owners() -> OwnersResponse:
        return OwnersResponse(owners=engine.list_owners())

    @app.get("/mem/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        return StatusResponse(**engine.status())

    @app.post("/mem/save", response_model=SaveResponse)
    def save(req: SaveRequest) -> SaveResponse:
        try:
            manifest = engine.snapshot(req.snapshot_path)
        except (RuntimeError, OSError) as exc:
            raise HTTPException(
                status_code=HTTP_STATUS_BAD_REQUEST, detail=str(exc),
            ) from exc
        return SaveResponse(manifest=Manifest(**manifest))

    @app.post("/mem/restore", response_model=RestoreResponse)
    def restore(req: RestoreRequest) -> RestoreResponse:
        try:
            restored = tardigrade_db.Engine.restore_from(
                req.snapshot_path, req.target_dir,
            )
        except (RuntimeError, OSError) as exc:
            raise HTTPException(
                status_code=HTTP_STATUS_BAD_REQUEST, detail=str(exc),
            ) from exc
        return RestoreResponse(ok=True, pack_count=restored.pack_count())

    return app


def main() -> None:
    """Env-driven entry point.

    Opens an engine from ``TARDIGRADE_HTTP_DB`` (default
    ``./tardigrade-http-engine``) and serves it on
    ``TARDIGRADE_HTTP_HOST``:``TARDIGRADE_HTTP_PORT``.
    """
    import uvicorn

    db_path = os.environ.get(ENV_DB_PATH, DEFAULT_DB_PATH)
    host = os.environ.get(ENV_HOST, DEFAULT_HOST)
    port = int(os.environ.get(ENV_PORT, str(DEFAULT_PORT)))
    engine = tardigrade_db.Engine(db_path)
    app = create_app(engine)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
