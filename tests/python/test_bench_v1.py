"""ATDD tests for benchmark v1 platform (design patterns + contracts)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tdb_bench.cli import main
from tdb_bench.ci_gate import GatePolicy, evaluate_report_payload
from tdb_bench.adapters.letta import LettaAdapter
from tdb_bench.adapters.mem0 import Mem0Adapter
from tdb_bench.fairness import FairnessError, validate_fairness
from tdb_bench.models import RunResultV1
from tdb_bench.reporting import render_report_markdown
from tdb_bench.runner import BenchmarkRunner
from tdb_bench.schema import SchemaValidationError, validate_run_result_v1


def _write_config(tmp_path: Path, mode: str = "smoke") -> Path:
    cfg = {
        "version": 1,
        "profiles": {
            "smoke": {
                "seed": 7,
                "timeout_seconds": 5,
                "datasets": [
                    {
                        "name": "locomo",
                        "revision": "locomo-fixture-v1",
                        "path": str(Path(__file__).resolve().parents[2] / "python" / "tdb_bench" / "fixtures" / "locomo_smoke.jsonl"),
                        "max_items": 3,
                    },
                    {
                        "name": "longmemeval",
                        "revision": "longmemeval-fixture-v1",
                        "path": str(Path(__file__).resolve().parents[2] / "python" / "tdb_bench" / "fixtures" / "longmemeval_smoke.jsonl"),
                        "max_items": 3,
                    },
                ],
                "systems": ["tardigrade", "mem0_oss", "letta"],
                "evaluator": {
                    "mode": "deterministic",
                    "answerer_model": "deterministic-answerer-v1",
                    "judge_model": "deterministic-judge-v1",
                },
                "top_k": 3,
                "prompts": {
                    "answer": "Answer concisely.",
                    "judge": "Score factual overlap only.",
                },
            },
            "full": {
                "seed": 42,
                "timeout_seconds": 30,
                "datasets": [
                    {
                        "name": "locomo",
                        "revision": "locomo-fixture-v1",
                        "path": str(Path(__file__).resolve().parents[2] / "python" / "tdb_bench" / "fixtures" / "locomo_smoke.jsonl"),
                        "max_items": 3,
                    },
                    {
                        "name": "longmemeval",
                        "revision": "longmemeval-fixture-v1",
                        "path": str(Path(__file__).resolve().parents[2] / "python" / "tdb_bench" / "fixtures" / "longmemeval_smoke.jsonl"),
                        "max_items": 3,
                    },
                ],
                "systems": ["tardigrade", "mem0_oss", "letta"],
                "evaluator": {
                    "mode": "deterministic",
                    "answerer_model": "deterministic-answerer-v1",
                    "judge_model": "deterministic-judge-v1",
                },
                "top_k": 3,
                "prompts": {
                    "answer": "Answer concisely.",
                    "judge": "Score factual overlap only.",
                },
            },
        },
    }
    path = tmp_path / "bench_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


# ATDD 1: Adapter contracts should produce schema-valid output across systems.
def test_adapter_contract_schema_valid(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    out = tmp_path / "run.json"

    runner = BenchmarkRunner.from_config_file(cfg_path)
    result = runner.run(mode="smoke", output_path=out)
    validate_run_result_v1(result.to_dict())

    systems = {item["system"] for item in result.items}
    assert systems == {"tardigrade", "mem0_oss", "letta"}


# ATDD 2: Fairness validator must fail mismatched evaluator/prompts/top_k.
def test_fairness_guard_rejects_mismatch() -> None:
    system_cfg = {
        "tardigrade": {
            "top_k": 5,
            "answerer_model": "a",
            "judge_model": "j",
            "answer_prompt": "p1",
            "judge_prompt": "p2",
        },
        "mem0_oss": {
            "top_k": 4,
            "answerer_model": "a",
            "judge_model": "j",
            "answer_prompt": "p1",
            "judge_prompt": "p2",
        },
    }

    with pytest.raises(FairnessError):
        validate_fairness(system_cfg)


# ATDD 3: Deterministic evaluator and fixed seed should be reproducible.
def test_deterministic_seed_reproducible(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)

    runner_a = BenchmarkRunner.from_config_file(cfg_path)
    runner_b = BenchmarkRunner.from_config_file(cfg_path)

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"

    a = runner_a.run(mode="smoke", output_path=out_a)
    b = runner_b.run(mode="smoke", output_path=out_b)

    assert a.aggregates == b.aggregates


# ATDD 3b: Repeat mode should emit replicate metadata and uncertainty stats.
def test_repeat_mode_emits_uncertainty_metadata(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    out = tmp_path / "repeat.json"

    runner = BenchmarkRunner.from_config_file(cfg_path)
    result = runner.run(mode="smoke", output_path=out, repeat=3)

    assert result.manifest["repeats"] == 3
    assert len(result.manifest["seeds"]) == 3
    tardigrade = result.aggregates["systems"]["tardigrade"]
    assert tardigrade["run_count"] == 3
    assert "score_stddev" in tardigrade
    assert "score_ci95" in tardigrade


# ATDD 4: Missing external endpoints should be explicit skipped outcomes.
def test_external_unavailable_is_explicit_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEM0_BASE_URL", raising=False)
    monkeypatch.delenv("LETTA_BASE_URL", raising=False)

    cfg_path = _write_config(tmp_path)
    out = tmp_path / "run.json"

    runner = BenchmarkRunner.from_config_file(cfg_path)
    result = runner.run(mode="smoke", output_path=out)

    mem0_statuses = [i["status"] for i in result.items if i["system"] == "mem0_oss"]
    letta_statuses = [i["status"] for i in result.items if i["system"] == "letta"]

    assert set(mem0_statuses) == {"skipped"}
    assert set(letta_statuses) == {"skipped"}


# ATDD 4b: Network/API failure should be surfaced as failed (not silent drop).
def test_external_network_failure_is_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEM0_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("MEM0_API_KEY", "dev-key")
    monkeypatch.delenv("LETTA_BASE_URL", raising=False)
    monkeypatch.delenv("LETTA_API_KEY", raising=False)

    cfg_path = _write_config(tmp_path)
    out = tmp_path / "run.json"

    runner = BenchmarkRunner.from_config_file(cfg_path)
    result = runner.run(mode="smoke", output_path=out)

    mem0_statuses = [i["status"] for i in result.items if i["system"] == "mem0_oss"]
    assert set(mem0_statuses) == {"failed"}


# ATDD 4c: Mem0 adapter should use OSS REST contract (/memories + /search + /reset).
def test_mem0_adapter_real_api_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEM0_BASE_URL", "http://localhost:8888")
    monkeypatch.setenv("MEM0_API_KEY", "dev-key")
    calls: list[tuple[str, str]] = []

    def fake_request(self: Mem0Adapter, method: str, path: str, **_: object) -> object:  # noqa: ARG001
        calls.append((method, path))
        if path == "/search":
            return [{"memory": "Alice likes hiking on weekends"}]
        return {}

    monkeypatch.setattr(Mem0Adapter, "_request_json", fake_request, raising=False)
    adapter = Mem0Adapter()
    from tdb_bench.models import BenchmarkItem  # local import for test scope clarity

    b = BenchmarkItem(
        item_id="mem0-1",
        dataset="locomo",
        context="Alice likes hiking on weekends.",
        question="What does Alice like to do?",
        ground_truth="hiking on weekends",
    )
    adapter.reset()
    adapter.ingest([b])
    result = adapter.query(b, top_k=3)

    assert ("POST", "/reset") in calls
    assert ("POST", "/memories") in calls
    assert ("POST", "/search") in calls
    assert result.status == "ok"
    assert "hiking" in result.answer.lower()


# ATDD 4d: Letta adapter should use archival-memory endpoints with managed benchmark agent.
def test_letta_adapter_real_api_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LETTA_BASE_URL", "http://localhost:8283")
    monkeypatch.setenv("LETTA_API_KEY", "dev-key")
    calls: list[tuple[str, str, object]] = []

    def fake_request(self: LettaAdapter, method: str, path: str, **kwargs: object) -> object:
        calls.append((method, path, kwargs.get("params")))
        if method == "GET" and path == "/v1/agents/":
            return []
        if method == "POST" and path == "/v1/agents/":
            return {"id": "agent-12345678-1234-4234-8234-123456789abc"}
        if method == "GET" and path.endswith("/archival-memory/search"):
            params = kwargs.get("params", {})
            if isinstance(params, dict) and str(params.get("query", "")).lower() == "alice":
                return {"results": [{"text": "Alice likes hiking on weekends"}], "count": 1}
            return {"results": [], "count": 0}
        if method == "GET" and path.endswith("/archival-memory"):
            return []
        return {}

    monkeypatch.setattr(LettaAdapter, "_request_json", fake_request, raising=False)
    adapter = LettaAdapter()
    from tdb_bench.models import BenchmarkItem  # local import for test scope clarity

    b = BenchmarkItem(
        item_id="letta-1",
        dataset="locomo",
        context="Alice likes hiking on weekends.",
        question="What does Alice like to do?",
        ground_truth="hiking on weekends",
    )
    adapter.reset()
    adapter.ingest([b])
    result = adapter.query(b, top_k=3)

    method_paths = {(m, p) for (m, p, _) in calls}
    assert ("GET", "/v1/agents/") in method_paths
    assert ("POST", "/v1/agents/") in method_paths
    assert ("POST", "/v1/agents/agent-12345678-1234-4234-8234-123456789abc/archival-memory") in method_paths
    assert ("GET", "/v1/agents/agent-12345678-1234-4234-8234-123456789abc/archival-memory/search") in method_paths
    # Ensure fallback semantic query term extraction kicked in.
    assert any(
        m == "GET"
        and p.endswith("/archival-memory/search")
        and isinstance(params, dict)
        and str(params.get("query", "")).lower() == "alice"
        for (m, p, params) in calls
    )
    assert result.status == "ok"
    assert "hiking" in result.answer.lower()


# ATDD 4e: Letta ingest should chunk large contexts to avoid archival-memory payload rejection.
def test_letta_adapter_ingest_chunks_large_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LETTA_BASE_URL", "http://localhost:8283")
    monkeypatch.setenv("LETTA_API_KEY", "dev-key")
    monkeypatch.setenv("LETTA_INGEST_MAX_CHARS", "32")
    payloads: list[dict[str, str]] = []

    def fake_request(self: LettaAdapter, method: str, path: str, **kwargs: object) -> object:
        if method == "GET" and path == "/v1/agents/":
            return []
        if method == "POST" and path == "/v1/agents/":
            return {"id": "agent-12345678-1234-4234-8234-123456789abc"}
        if method == "POST" and path.endswith("/archival-memory"):
            payload = kwargs.get("payload")
            assert isinstance(payload, dict)
            assert "text" in payload
            assert len(str(payload["text"])) <= 32
            payloads.append({"text": str(payload["text"])})
            return {}
        return {}

    monkeypatch.setattr(LettaAdapter, "_request_json", fake_request, raising=False)
    adapter = LettaAdapter()
    from tdb_bench.models import BenchmarkItem  # local import for test scope clarity

    b = BenchmarkItem(
        item_id="letta-chunk-1",
        dataset="locomo",
        context="Alice likes hiking on weekends and also enjoys painting after dinner.",
        question="What does Alice like?",
        ground_truth="hiking and painting",
    )

    adapter.ingest([b])
    assert len(payloads) >= 2


# ATDD 5: Schema validator should reject malformed run documents.
def test_schema_validator_rejects_bad_payload() -> None:
    with pytest.raises(SchemaValidationError):
        validate_run_result_v1({"version": "wrong"})


# ATDD 6: CLI run/report/compare should emit stable artifacts.
def test_cli_run_report_compare(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    run_a = tmp_path / "run_a.json"
    run_b = tmp_path / "run_b.json"
    report_md = tmp_path / "report.md"
    report_json = tmp_path / "report.json"
    compare_md = tmp_path / "compare.md"

    assert main(["run", "--mode", "smoke", "--config", str(cfg_path), "--output", str(run_a)]) == 0
    assert main(["run", "--mode", "smoke", "--config", str(cfg_path), "--output", str(run_b)]) == 0

    assert main(["report", "--input", str(run_a), "--format", "md", "--output", str(report_md)]) == 0
    assert main(["report", "--input", str(run_a), "--format", "json", "--output", str(report_json)]) == 0
    assert main(
        [
            "compare",
            "--baseline",
            str(run_a),
            "--candidate",
            str(run_b),
            "--format",
            "md",
            "--output",
            str(compare_md),
        ]
    ) == 0

    assert run_a.exists()
    assert run_b.exists()
    report_md_text = report_md.read_text(encoding="utf-8")
    assert "# Benchmark Report" in report_md_text
    assert "- run_validity:" in report_md_text
    json_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert "aggregates" in json_report
    assert "Benchmark Comparison" in compare_md.read_text(encoding="utf-8")


# ATDD 6b: CLI repeat flag should run multiple replicates and persist manifest fields.
def test_cli_run_repeat_flag(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    run_path = tmp_path / "repeat_cli.json"
    assert (
        main(
            [
                "run",
                "--mode",
                "smoke",
                "--repeat",
                "2",
                "--config",
                str(cfg_path),
                "--output",
                str(run_path),
            ]
        )
        == 0
    )

    payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert payload["manifest"]["repeats"] == 2
    assert len(payload["manifest"]["seeds"]) == 2


# ATDD 7: RunResultV1 object should serialize and round-trip key fields.
def test_run_result_v1_round_trip(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    out = tmp_path / "run.json"

    runner = BenchmarkRunner.from_config_file(cfg_path)
    result = runner.run(mode="smoke", output_path=out)

    loaded = RunResultV1.from_dict(json.loads(out.read_text(encoding="utf-8")))
    assert loaded.version == 1
    assert loaded.manifest["mode"] == "smoke"
    assert loaded.status_summary


# ATDD 8: Report should mark runs invalid when a system has zero successful results.
def test_report_marks_invalid_when_system_has_no_ok_results() -> None:
    run = RunResultV1(
        version=1,
        manifest={"mode": "full", "git_sha": "abc123", "repeats": 1, "seeds": [42]},
        items=[],
        aggregates={
            "systems": {
                "tardigrade": {
                    "avg_score": 0.75,
                    "score_stddev": 0.0,
                    "score_ci95": 0.0,
                    "run_count": 1,
                    "ok": 10,
                    "skipped": 0,
                    "failed": 0,
                },
                "letta": {
                    "avg_score": 0.0,
                    "score_stddev": 0.0,
                    "score_ci95": 0.0,
                    "run_count": 1,
                    "ok": 0,
                    "skipped": 0,
                    "failed": 10,
                },
            }
        },
        comparisons={},
        status_summary={"ok": 10, "failed": 10},
    )

    report_md = render_report_markdown(run)
    assert "- run_validity: `invalid`" in report_md


# ATDD 9: CI gate should fail when run validity is invalid.
def test_ci_gate_fails_invalid_run_validity() -> None:
    payload = {
        "run_validity": {"state": "invalid", "reason": "no_successful_results_for=letta"},
        "status_summary": {"ok": 10, "failed": 10},
    }
    decision = evaluate_report_payload(payload, GatePolicy())
    assert decision.should_fail is True
    assert any("run_validity=invalid" in msg for msg in decision.failures)


# ATDD 10: CI gate should warn/fail on non-ok ratio thresholds.
def test_ci_gate_warn_and_fail_non_ok_ratio_thresholds() -> None:
    payload = {
        "run_validity": {"state": "comparable", "reason": "all_systems_ok"},
        "status_summary": {"ok": 80, "skipped": 20},
    }
    warn_only = evaluate_report_payload(
        payload,
        GatePolicy(
            warn_non_ok_ratio=0.1,
            fail_non_ok_ratio=0.5,
        ),
    )
    assert warn_only.should_fail is False
    assert warn_only.warnings

    fail = evaluate_report_payload(
        payload,
        GatePolicy(
            warn_non_ok_ratio=0.1,
            fail_non_ok_ratio=0.2,
        ),
    )
    assert fail.should_fail is True
    assert fail.failures
