"""Schema validation for RunResultV1 payloads."""

from __future__ import annotations


class SchemaValidationError(ValueError):
    """Raised for schema contract violations."""


def validate_run_result_v1(payload: dict) -> None:
    required_root = {
        "version": int,
        "manifest": dict,
        "items": list,
        "aggregates": dict,
        "comparisons": dict,
        "status_summary": dict,
    }

    for key, expected in required_root.items():
        if key not in payload:
            raise SchemaValidationError(f"Missing root key: {key}")
        if not isinstance(payload[key], expected):
            raise SchemaValidationError(f"Root key '{key}' should be {expected.__name__}")

    if payload["version"] != 1:
        raise SchemaValidationError("Unsupported schema version")

    manifest = payload["manifest"]
    if "repeats" in manifest and (not isinstance(manifest["repeats"], int) or manifest["repeats"] < 1):
        raise SchemaValidationError("Manifest key 'repeats' should be integer >= 1")
    if "seeds" in manifest:
        if not isinstance(manifest["seeds"], list) or not all(isinstance(x, int) for x in manifest["seeds"]):
            raise SchemaValidationError("Manifest key 'seeds' should be int list")

    for idx, item in enumerate(payload["items"]):
        if not isinstance(item, dict):
            raise SchemaValidationError(f"Item[{idx}] should be object")
        for key in [
            "item_id",
            "dataset",
            "system",
            "status",
            "latency_ms",
            "question",
            "answer",
            "ground_truth",
        ]:
            if key not in item:
                raise SchemaValidationError(f"Item[{idx}] missing key '{key}'")
