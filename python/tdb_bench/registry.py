"""Factory registry for adapters/datasets/evaluators."""

from __future__ import annotations

from tdb_bench.adapters import LettaAdapter, Mem0Adapter, TardigradeAdapter
from tdb_bench.adapters.retrieve_then_read import RetrieveThenReadAdapter
from tdb_bench.answerers import build_answerer_from_env
from tdb_bench.contracts import BenchmarkAdapter, DatasetAdapter, Evaluator
from tdb_bench.datasets import LoCoMoDatasetAdapter, LongMemEvalDatasetAdapter
from tdb_bench.evaluators import (
    DeepSeekProvider,
    DeterministicEvaluator,
    JustifyThenJudgeEvaluator,
    LLMGatedEvaluator,
    OpenAIProvider,
)
from tdb_bench.errors import ConfigError


_SYSTEM_TARDIGRADE = "tardigrade"
_SYSTEM_TARDIGRADE_LLM_GATED = "tardigrade-llm-gated"
_SYSTEM_MEM0_OSS = "mem0_oss"
_SYSTEM_LETTA = "letta"


class RegistryFactory:
    """Factory pattern to create pluggable benchmark components."""

    @staticmethod
    def create_adapter(system: str, timeout_seconds: int) -> BenchmarkAdapter:
        if system == _SYSTEM_TARDIGRADE:
            return TardigradeAdapter()
        if system == _SYSTEM_TARDIGRADE_LLM_GATED:
            generator, model_label = build_answerer_from_env()
            return RetrieveThenReadAdapter(
                inner=TardigradeAdapter(),
                generator=generator,
                answerer_model=model_label,
            )
        if system == _SYSTEM_MEM0_OSS:
            return Mem0Adapter(timeout_seconds=timeout_seconds)
        if system == _SYSTEM_LETTA:
            return LettaAdapter(timeout_seconds=timeout_seconds)
        raise ConfigError(f"Unknown system: {system}")

    @staticmethod
    def create_dataset(dataset_cfg: dict) -> DatasetAdapter:
        name = dataset_cfg["name"]
        revision = dataset_cfg["revision"]
        path = dataset_cfg["path"]

        if name == "locomo":
            return LoCoMoDatasetAdapter(revision=revision, path=path)
        if name == "longmemeval":
            return LongMemEvalDatasetAdapter(revision=revision, path=path)
        raise ConfigError(f"Unknown dataset: {name}")

    @staticmethod
    def create_evaluator(evaluator_cfg: dict) -> Evaluator:
        mode = evaluator_cfg.get("mode", "deterministic")
        judge_model = evaluator_cfg.get("judge_model", "gpt-4.1-mini")

        if mode == "deterministic":
            return DeterministicEvaluator()
        if mode in ("llm", "llm_gated"):
            providers = [
                DeepSeekProvider(),
                OpenAIProvider(model=judge_model),
            ]
            return LLMGatedEvaluator(providers=providers)
        if mode == "justify_then_judge":
            # Both stages use the same provider chain — DeepSeek first
            # (cheaper, already keyed), OpenAI as fallback. The justify
            # stage requests longer max_tokens than the judge stage; that
            # routing happens inside JustifyThenJudgeEvaluator.
            justify_providers = [
                DeepSeekProvider(),
                OpenAIProvider(model=judge_model),
            ]
            judge_providers = [
                DeepSeekProvider(),
                OpenAIProvider(model=judge_model),
            ]
            return JustifyThenJudgeEvaluator(
                justify_providers=justify_providers,
                judge_providers=judge_providers,
            )
        raise ConfigError(f"Unknown evaluator mode: {mode}")
