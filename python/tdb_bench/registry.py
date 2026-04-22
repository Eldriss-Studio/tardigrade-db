"""Factory registry for adapters/datasets/evaluators."""

from __future__ import annotations

from tdb_bench.adapters import LettaAdapter, Mem0Adapter, TardigradeAdapter
from tdb_bench.contracts import BenchmarkAdapter, DatasetAdapter, Evaluator
from tdb_bench.datasets import LoCoMoDatasetAdapter, LongMemEvalDatasetAdapter
from tdb_bench.evaluators import DeterministicEvaluator, LLMGatedEvaluator
from tdb_bench.errors import ConfigError


class RegistryFactory:
    """Factory pattern to create pluggable benchmark components."""

    @staticmethod
    def create_adapter(system: str, timeout_seconds: int) -> BenchmarkAdapter:
        if system == "tardigrade":
            return TardigradeAdapter()
        if system == "mem0_oss":
            return Mem0Adapter(timeout_seconds=timeout_seconds)
        if system == "letta":
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
        answerer_model = evaluator_cfg.get("answerer_model", "deterministic-answerer")
        judge_model = evaluator_cfg.get("judge_model", "deterministic-judge")

        if mode == "deterministic":
            return DeterministicEvaluator()
        if mode == "llm":
            return LLMGatedEvaluator(answerer_model=answerer_model, judge_model=judge_model)
        if mode == "llm_gated":
            return LLMGatedEvaluator(answerer_model=answerer_model, judge_model=judge_model)
        raise ConfigError(f"Unknown evaluator mode: {mode}")
