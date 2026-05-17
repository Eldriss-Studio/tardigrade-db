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
    HTTP_STATUS_VALIDATION_ERROR,
    PROBLEM_JSON_CONTENT_TYPE,
    PROBLEM_TYPE_ABOUT_BLANK,
    STUB_KEY_DIM,
    STUB_LAYER_INDEX,
    STUB_VALUE_DIM,
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


def _problem_response(status: int, title: str, detail: str | None = None) -> JSONResponse:
    """Build an RFC 7807 problem-detail JSON response."""
    return JSONResponse(
        status_code=status,
        media_type=PROBLEM_JSON_CONTENT_TYPE,
        content={
            "type": PROBLEM_TYPE_ABOUT_BLANK,
            "title": title,
            "status": status,
            "detail": detail,
        },
    )


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

    @app.post("/mem/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        key, _ = app.state.kv_fn(req.query_text)
        rows = engine.mem_read_pack(key, req.k, req.owner)
        return QueryResponse(
            results=[
                QueryResult(
                    pack_id=int(r["pack_id"]),
                    score=float(r["score"]),
                    text=r.get("text") or engine.pack_text(r["pack_id"]),
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
