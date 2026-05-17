#!/usr/bin/env python3
"""Pre-commit gate: lint-time imports must not require the native extension.

CI's lint job runs ``python -m tdb_bench.ci_gate`` *without* building
the PyO3 wheel — by design, fast-feedback lint should not require a
Rust toolchain in the loop. That means every module reachable at
import time from the bench/CLI entry points must avoid touching
``tardigrade_db._native`` until it's actually used at runtime.

This script simulates the CI environment locally by hiding the
native extension from the importer, then exercises the CI import
chain. Failures here would otherwise only surface on push (or
worse, after merge).

Run via lefthook pre-commit. Add new entry-point modules to
``_LINT_TIME_IMPORTS`` below as the surface area grows.
"""

from __future__ import annotations

import sys

# Modules CI imports at lint time. If any of these transitively
# touches the native extension, the lint job fails. Keep this list
# in sync with the actual ``python -m`` invocations in CI.
_LINT_TIME_IMPORTS: tuple[str, ...] = (
    "tdb_bench.ci_gate",
    "tdb_bench.cli",
    "tardigrade_cli.main",
    "tardigrade_hooks",
    "tardigrade_hooks.constants",
)


_SENTINEL = "TDB_LAZY_IMPORT_GATE_NATIVE_BLOCKED"


class _HideNative:
    """``sys.meta_path`` shim that pretends the PyO3 extension is missing.

    Raises a ``ModuleNotFoundError`` whose message embeds
    :data:`_SENTINEL` so failures upstream can be attributed to the
    synthetic block (vs. an unrelated missing dev dependency like
    numpy that simply isn't installed).
    """

    @staticmethod
    def find_spec(name, _path=None, _target=None):
        if name == "tardigrade_db._native":
            raise ModuleNotFoundError(
                f"simulated absence of '{name}' [{_SENTINEL}]",
            )
        return None


def main() -> int:
    sys.meta_path.insert(0, _HideNative())

    failures: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for mod_name in _LINT_TIME_IMPORTS:
        # Drop stale cache entries so each import re-runs through
        # the hidden-native simulation.
        for key in list(sys.modules):
            if key.startswith(("tdb_bench", "tardigrade_")):
                del sys.modules[key]
        try:
            __import__(mod_name)
        except Exception as exc:  # noqa: BLE001 — we classify below
            msg = f"{type(exc).__name__}: {exc}"
            # Distinguish "the gate caught a real regression" from
            # "this dev environment is missing an unrelated dep
            # (numpy, etc.)" — only the former should fail the
            # commit. The synthetic block embeds _SENTINEL so we
            # can match it precisely.
            if _SENTINEL in msg:
                failures.append((mod_name, msg))
            else:
                skipped.append((mod_name, msg))

    if failures:
        # `print_stderr` is OK here — this is a CLI script, not a
        # library. We want the output visible in lefthook.
        sys.stderr.write(
            "lint_lazy_imports: one or more lint-time imports require "
            "the native extension. This means the CI lint job will fail.\n",
        )
        for mod, msg in failures:
            sys.stderr.write(f"  - {mod}: {msg}\n")
        sys.stderr.write(
            "\nFix: defer the import of any compiled-extension-requiring\n"
            "module into a function body, or use PEP 562 lazy __getattr__\n"
            "in the package __init__.py.\n",
        )
        return 1
    if skipped:
        sys.stderr.write(
            "lint_lazy_imports: skipped — the dev environment is missing\n"
            "an unrelated dependency. Activate the venv "
            "(``source .venv/bin/activate``) for full coverage. "
            "First skip reason was:\n",
        )
        first_mod, first_msg = skipped[0]
        sys.stderr.write(f"  {first_mod}: {first_msg}\n")
        # Skips are warnings, not failures.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
