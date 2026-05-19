"""JSON-backed persistence for :class:`CalibrationResult` records.

Design pattern: **Repository**. Hides the storage backend (a single
JSON file with model_id → record entries) behind a CRUD interface.

::

    ┌──────────────────────────────────────────────┐
    │           CalibrationRegistry                │
    │   load(model_id)   → CalibrationResult | None│
    │   save(result)                               │
    │   all_keys()       → list[str]               │
    │   clear(model_id=None)                       │
    └──────────────────────────────────────────────┘
                       │
                       ▼
       ~/.tardigrade/calibration.json  (or via
       $TARDIGRADE_CALIBRATION_PATH env var)

Writes are atomic (temp-file + rename), so a crash mid-save cannot
corrupt the existing registry. Reads tolerate (and warn on) a
corrupted file, returning None as if no record existed — calibration
re-runs and re-saves on the next invocation.

# Example

>>> from tardigrade_hooks import CalibrationRegistry, select_query_layer
>>> reg = CalibrationRegistry()  # ~/.tardigrade/calibration.json
>>> result = select_query_layer(model, tok, registry=reg)
>>> # next call returns the cached result without re-running the sweep
>>> reg.load("Qwen/Qwen3-0.6B").best_layer
17
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import tardigrade_db

from .calibrate import CalibrationResult

#: Environment variable that overrides the default registry path.
#: Primarily for testability; production consumers should let the default
#: path under ``~/.tardigrade/`` apply.
CALIBRATION_PATH_ENV: str = "TARDIGRADE_CALIBRATION_PATH"


def _default_path() -> Path:
    """Resolve the default registry path, honouring the env var override.

    Order: ``$TARDIGRADE_CALIBRATION_PATH`` if set, else
    ``~/.tardigrade/calibration.json``.
    """
    override = os.environ.get(CALIBRATION_PATH_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".tardigrade" / "calibration.json"


def _current_version() -> str:
    """Return the currently-installed tardigrade_db version string.

    Wrapped in a function (rather than module-level constant) so tests
    can monkeypatch the version they expect to see.
    """
    return getattr(tardigrade_db, "__version__", "unknown")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically.

    Uses ``tempfile.NamedTemporaryFile`` + ``os.replace`` so the
    destination is either the previous version or the full new
    version — never a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        # If json.dump or anything else blows up, leave the original
        # file untouched and remove our scratch file.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


class CalibrationRegistry:
    """Persistent CRUD store for calibration results, one per model_id.

    Args:
        path: Override the registry's on-disk path. When ``None``, uses
            ``$TARDIGRADE_CALIBRATION_PATH`` if set, else
            ``~/.tardigrade/calibration.json``.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path: Path = Path(path).expanduser() if path is not None else _default_path()

    # ------------------------------------------------------------------ load

    def load(self, model_id: str) -> Optional[CalibrationResult]:
        """Return the cached result for ``model_id`` or ``None``.

        Emits a ``UserWarning`` if the cached record was written by a
        different library version (the cached layer choice may still be
        valid but the library may have changed since — consumers should
        decide whether to re-calibrate).

        Returns ``None`` (with a warning) when the registry file is
        present but corrupted — callers fall through to re-calibration.
        """
        data = self._read_or_none()
        if data is None:
            return None
        entry = data.get(model_id)
        if entry is None:
            return None
        try:
            result = CalibrationResult.from_dict(entry)
        except (KeyError, TypeError, ValueError) as exc:
            warnings.warn(
                f"calibration registry entry for {model_id!r} is malformed: "
                f"{exc!r}; ignoring",
                UserWarning,
                stacklevel=2,
            )
            return None
        current = _current_version()
        if result.tardigrade_db_version != current:
            warnings.warn(
                f"cached calibration for {model_id!r} was written by "
                f"tardigrade-db version {result.tardigrade_db_version!r}; "
                f"this process runs {current!r} — the cached layer choice "
                f"may be stale, consider re-calibrating",
                UserWarning,
                stacklevel=2,
            )
        return result

    # ------------------------------------------------------------------ save

    def save(self, result: CalibrationResult) -> None:
        """Persist ``result``, overwriting any prior entry for the same model_id."""
        data = self._read_or_none() or {}
        data[result.model_id] = result.as_dict()
        _atomic_write_json(self.path, data)

    # ------------------------------------------------------------------ list

    def all_keys(self) -> list[str]:
        """Return all model_ids currently in the registry."""
        data = self._read_or_none()
        if data is None:
            return []
        return list(data.keys())

    # ------------------------------------------------------------------ clear

    def clear(self, model_id: Optional[str] = None) -> None:
        """Remove one entry (when ``model_id`` is given) or all entries."""
        if model_id is None:
            if self.path.exists():
                self.path.unlink()
            return
        data = self._read_or_none()
        if data is None or model_id not in data:
            return
        del data[model_id]
        _atomic_write_json(self.path, data)

    # -------------------------------------------------------------- internal

    def _read_or_none(self) -> Optional[dict]:
        """Read and parse the JSON file. Returns None if missing.

        Emits a ``UserWarning`` and also returns ``None`` if the file
        exists but is unparsable — calibration will re-run and re-save,
        replacing the corrupted file.
        """
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.warn(
                f"calibration registry at {self.path} is corrupt "
                f"({exc!r}); treating as empty",
                UserWarning,
                stacklevel=2,
            )
            return None
