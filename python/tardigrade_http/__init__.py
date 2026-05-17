"""TardigradeDB HTTP bridge — FastAPI server exposing the engine over REST.

The bridge is an **Adapter** in front of :class:`tardigrade_db.Engine`:
the engine's typed Rust API is rebuilt as a small set of JSON HTTP
endpoints so non-Python consumers (Node, Deno, browsers, other
runtimes) can plug in without going through PyO3.

The public surface here is intentionally tiny:

- :func:`create_app` — build a FastAPI app around an existing
  :class:`Engine`. Useful for tests and for embedding the bridge
  inside a larger Python service.
- :func:`main` — env-driven entry point that opens an engine from
  ``TARDIGRADE_HTTP_DB`` and serves it with uvicorn.

The committed OpenAPI contract lives at
``python/tardigrade_http/schema.yaml`` — regenerate it after any
model/route change with ``python -m tardigrade_http.export_schema``.
A pinned test (``tests/python/test_http_bridge.py::TestCheckedInSchema``)
fails if the live app drifts from the checked-in file.

Security note
-------------

The bridge has **no built-in authentication** and accepts arbitrary
filesystem paths in ``snapshot_path`` / ``target_dir``. The default
``main()`` binds to ``127.0.0.1`` for that reason. Production use
behind a public interface needs a reverse proxy that enforces auth
and a path-allow-list policy.
"""

from .server import create_app, main

__all__ = ["create_app", "main"]
