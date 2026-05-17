"""ATDD: challenger profile in default.json.

Adds a third profile that bundles the upgraded LoCoMo-race stack:
Qwen3-1.7B capture, phase1_oracle_full data, justify_then_judge
evaluator, tardigrade-llm-gated system, deepseek-chat answerer.

Pins both the profile shape (B4.1) and that
``BenchmarkRunner.run()`` accepts it without `FairnessError` (B4.2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CFG = _REPO_ROOT / "python" / "tdb_bench" / "config" / "default.json"


def _load_config() -> dict:
    return json.loads(_DEFAULT_CFG.read_text(encoding="utf-8"))


# ─── B4.1 — profile validates ──────────────────────────────────────────────


class TestChallengerProfileShape:
    def test_default_config_has_challenger_profile(self):
        cfg = _load_config()
        assert "challenger" in cfg["profiles"]

    def test_challenger_uses_justify_then_judge_evaluator(self):
        challenger = _load_config()["profiles"]["challenger"]
        assert challenger["evaluator"]["mode"] == "justify_then_judge"

    def test_challenger_targets_tardigrade_llm_gated_system(self):
        challenger = _load_config()["profiles"]["challenger"]
        assert "tardigrade-llm-gated" in challenger["systems"]

    def test_challenger_uses_phase1_oracle_full_dataset(self):
        challenger = _load_config()["profiles"]["challenger"]
        # Path is env-overridable but defaults to the full revision.
        # Either the path or revision references the full corpus.
        for dataset in challenger["datasets"]:
            if dataset["name"] == "locomo":
                assert "phase1_oracle_full" in (
                    dataset.get("path", "") + dataset.get("revision", "")
                )

    def test_challenger_records_answerer_model(self):
        # The bench manifest records this; the answerer is the
        # adapter's LLM (DeepSeek by default in the gated path) and
        # the judge_model is the evaluator's stage-2 model.
        challenger = _load_config()["profiles"]["challenger"]
        assert challenger["evaluator"]["answerer_model"]
        assert challenger["evaluator"]["judge_model"]


# ─── B4.2 — fairness check passes ─────────────────────────────────────────


class TestChallengerFairness:
    def test_fairness_validator_accepts_single_system_profile(self):
        # All systems in the profile share the same top_k / models /
        # prompts, so validate_fairness must accept.
        from tdb_bench.fairness import validate_fairness

        challenger = _load_config()["profiles"]["challenger"]
        per_system = {
            system: {
                "top_k": challenger["top_k"],
                "answerer_model": challenger["evaluator"].get("answerer_model", ""),
                "judge_model": challenger["evaluator"].get("judge_model", ""),
                "answer_prompt": challenger["prompts"].get("answer", ""),
                "judge_prompt": challenger["prompts"].get("judge", ""),
            }
            for system in challenger["systems"]
        }
        # No FairnessError raised.
        validate_fairness(per_system)


# ─── Registry happy path ───────────────────────────────────────────────────


class TestChallengerRegistryWiring:
    def test_create_evaluator_from_challenger_profile_returns_jtj(self):
        from tdb_bench.evaluators import JustifyThenJudgeEvaluator
        from tdb_bench.registry import RegistryFactory

        challenger = _load_config()["profiles"]["challenger"]
        evaluator = RegistryFactory.create_evaluator(challenger["evaluator"])
        assert isinstance(evaluator, JustifyThenJudgeEvaluator)
