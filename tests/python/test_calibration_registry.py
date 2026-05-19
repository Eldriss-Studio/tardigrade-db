# ATDD acceptance tests for CalibrationRegistry.
#
# Design pattern: Repository (CRUD over a single JSON-backed store).
# Persists CalibrationResult records keyed by model_id, with atomic
# writes and env-var-overridable path for testability.

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tardigrade_hooks.calibrate import CalibrationResult, LayerScore
from tardigrade_hooks.calibration_registry import (
    CALIBRATION_PATH_ENV,
    CalibrationRegistry,
)


import tardigrade_hooks.calibration_registry as _reg_mod


def _make_result(model_id="example/model", best_layer=7,
                 tardigrade_db_version=None) -> CalibrationResult:
    if tardigrade_db_version is None:
        tardigrade_db_version = _reg_mod._current_version()
    return CalibrationResult(
        model_id=model_id,
        tardigrade_db_version=tardigrade_db_version,
        timestamp_iso="2026-05-19T12:00:00",
        n_layers=12,
        hidden_size=512,
        best_layer=best_layer,
        scores=(
            LayerScore(layer=0, kind="embedding", top1=0, top5=0),
            LayerScore(layer=best_layer, kind="attention", top1=18, top5=20),
            LayerScore(layer=11, kind="attention", top1=12, top5=18),
        ),
    )


def test_round_trip_save_then_load_returns_equivalent_record(tmp_path):
    reg = CalibrationRegistry(tmp_path / "calib.json")
    original = _make_result()
    reg.save(original)
    loaded = reg.load(original.model_id)
    assert loaded == original


def test_load_returns_none_for_unknown_model(tmp_path):
    reg = CalibrationRegistry(tmp_path / "calib.json")
    assert reg.load("nobody/has/calibrated/this") is None


def test_env_var_overrides_default_registry_path(tmp_path, monkeypatch):
    monkeypatch.setenv(CALIBRATION_PATH_ENV, str(tmp_path / "from_env.json"))
    reg = CalibrationRegistry()  # no explicit path → reads env var
    reg.save(_make_result())
    assert (tmp_path / "from_env.json").exists()


def test_load_warns_on_version_mismatch(tmp_path, monkeypatch):
    reg = CalibrationRegistry(tmp_path / "calib.json")
    reg.save(_make_result(tardigrade_db_version="0.0.0-ancient"))
    import tardigrade_hooks.calibration_registry as mod
    # Force a known current version so the test is deterministic.
    monkeypatch.setattr(mod, "_current_version", lambda: "9.9.9")
    with pytest.warns(UserWarning, match="calibration.*version"):
        loaded = reg.load("example/model")
    assert loaded is not None
    assert loaded.tardigrade_db_version == "0.0.0-ancient"


def test_clear_removes_one_model(tmp_path):
    reg = CalibrationRegistry(tmp_path / "calib.json")
    reg.save(_make_result(model_id="alpha"))
    reg.save(_make_result(model_id="beta"))
    reg.clear("alpha")
    assert reg.load("alpha") is None
    assert reg.load("beta") is not None


def test_clear_with_no_arg_removes_everything(tmp_path):
    reg = CalibrationRegistry(tmp_path / "calib.json")
    reg.save(_make_result(model_id="alpha"))
    reg.save(_make_result(model_id="beta"))
    reg.clear()
    assert reg.all_keys() == []


def test_all_keys_lists_stored_model_ids(tmp_path):
    reg = CalibrationRegistry(tmp_path / "calib.json")
    reg.save(_make_result(model_id="alpha"))
    reg.save(_make_result(model_id="beta"))
    keys = sorted(reg.all_keys())
    assert keys == ["alpha", "beta"]


def test_save_is_atomic_no_partial_file_on_crash(tmp_path, monkeypatch):
    # If the JSON encoder raises mid-write, the destination file must
    # either be unchanged or be the full new content — never partial.
    reg = CalibrationRegistry(tmp_path / "calib.json")
    reg.save(_make_result(model_id="alpha"))  # known-good baseline
    original_bytes = (tmp_path / "calib.json").read_bytes()

    # Patch json.dump to blow up halfway.
    import tardigrade_hooks.calibration_registry as mod
    real_dump = mod.json.dump

    def boom(*args, **kwargs):
        raise RuntimeError("simulated mid-write crash")

    monkeypatch.setattr(mod.json, "dump", boom)
    with pytest.raises(RuntimeError):
        reg.save(_make_result(model_id="beta"))

    # File on disk should still match the pre-crash state.
    monkeypatch.setattr(mod.json, "dump", real_dump)
    assert (tmp_path / "calib.json").read_bytes() == original_bytes


def test_load_returns_none_on_corrupted_registry_file(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text("{ this is not valid json")
    reg = CalibrationRegistry(path)
    with pytest.warns(UserWarning, match="corrupt"):
        assert reg.load("any-model") is None


def test_save_creates_parent_directory_if_missing(tmp_path):
    nested = tmp_path / "nested" / "subdir" / "calib.json"
    reg = CalibrationRegistry(nested)
    reg.save(_make_result())
    assert nested.exists()
