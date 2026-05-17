"""Export the bridge's OpenAPI schema to a YAML file.

Runnable as ``python -m tardigrade_http.export_schema [out_path]``.

The schema lives at ``python/tardigrade_http/schema.yaml`` in the
repo and is the contract consumers code against. Regenerate it
after any change to :mod:`tardigrade_http.models` or
:mod:`tardigrade_http.server`:

.. code-block:: bash

    python -m tardigrade_http.export_schema

The exporter intentionally avoids opening a real engine — it
builds the FastAPI app against a stub stand-in just long enough
to harvest ``app.openapi()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from .server import create_app


class _StubEngine:
    """Minimal stand-in so ``create_app`` can introspect routes.

    None of these methods are exercised by ``app.openapi()``; we
    only need an object with the right attribute surface so the
    factory's bookkeeping (``app.state.engine = engine``) doesn't
    blow up.
    """

    def mem_write_pack(self, *_a, **_kw):  # pragma: no cover
        raise NotImplementedError

    def mem_read_pack(self, *_a, **_kw):  # pragma: no cover
        raise NotImplementedError

    def list_owners(self):  # pragma: no cover
        return []

    def status(self):  # pragma: no cover
        return {}

    def snapshot(self, *_a, **_kw):  # pragma: no cover
        raise NotImplementedError

    def pack_count(self):  # pragma: no cover
        return 0

    def pack_text(self, _pid):  # pragma: no cover
        return None


def export_schema(out_path: Path) -> None:
    app = create_app(_StubEngine())
    schema = app.openapi()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(schema, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    default_path = Path(__file__).resolve().parent / "schema.yaml"
    out_path = Path(argv[0]) if argv else default_path
    export_schema(out_path)
    print(f"wrote OpenAPI schema → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
