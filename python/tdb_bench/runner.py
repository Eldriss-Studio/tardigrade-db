"""Benchmark runner (Template Method orchestration)."""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from tdb_bench.errors import DatasetUnavailableError
from tdb_bench.fairness import validate_fairness
from tdb_bench.models import RunResultV1
from tdb_bench.registry import RegistryFactory
from tdb_bench.schema import validate_run_result_v1


@dataclass
class ProfileConfig:
    seed: int
    timeout_seconds: int
    datasets: list[dict]
    systems: list[str]
    evaluator: dict
    top_k: int
    prompts: dict


class BenchmarkRunner:
    """Template Method runner coordinating datasets, adapters and evaluators."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @classmethod
    def from_config_file(cls, path: Path) -> "BenchmarkRunner":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = _expand_env(payload)
        return cls(config=payload)

    def run(
        self,
        mode: str,
        output_path: Path,
        systems: list[str] | None = None,
        datasets: list[str] | None = None,
        repeat: int = 1,
        seeds: list[int] | None = None,
    ) -> RunResultV1:
        profile = self._profile(mode)
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        resolved_seeds = self._resolve_seeds(base_seed=profile.seed, repeat=repeat, seeds=seeds)

        selected_systems = systems or profile.systems
        selected_datasets = datasets or [d["name"] for d in profile.datasets]

        fairness_payload = {
            system: {
                "top_k": profile.top_k,
                "answerer_model": profile.evaluator.get("answerer_model", ""),
                "judge_model": profile.evaluator.get("judge_model", ""),
                "answer_prompt": profile.prompts.get("answer", ""),
                "judge_prompt": profile.prompts.get("judge", ""),
            }
            for system in selected_systems
        }
        validate_fairness(fairness_payload)

        items_out: list[dict[str, Any]] = []
        dataset_meta: dict[str, dict] = {}
        adapter_meta: dict[str, dict[str, str]] = {}

        for run_index, seed in enumerate(resolved_seeds):
            random.seed(seed)
            evaluator = RegistryFactory.create_evaluator(profile.evaluator)
            adapters = {
                name: RegistryFactory.create_adapter(name, timeout_seconds=profile.timeout_seconds)
                for name in selected_systems
            }
            if not adapter_meta:
                adapter_meta = {name: ad.metadata() for name, ad in adapters.items()}

            run_items = self._run_once(
                profile=profile,
                selected_systems=selected_systems,
                selected_datasets=selected_datasets,
                adapters=adapters,
                evaluator=evaluator,
                dataset_meta=dataset_meta,
            )
            for row in run_items:
                row["replicate"] = run_index
                row["seed"] = seed
            items_out.extend(run_items)

        result = RunResultV1(
            version=1,
            manifest=self._manifest(
                mode,
                profile,
                selected_systems,
                selected_datasets,
                dataset_meta,
                adapter_meta,
                repeats=len(resolved_seeds),
                seeds=resolved_seeds,
            ),
            items=items_out,
            aggregates=self._aggregates(items_out),
            comparisons=self._comparisons(items_out),
            status_summary=self._status_summary(items_out),
        )

        payload = result.to_dict()
        validate_run_result_v1(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        return result

    def _run_once(
        self,
        profile: ProfileConfig,
        selected_systems: list[str],
        selected_datasets: list[str],
        adapters: dict[str, Any],
        evaluator: Any,
        dataset_meta: dict[str, dict],
    ) -> list[dict[str, Any]]:
        items_out: list[dict[str, Any]] = []
        for dataset_cfg in profile.datasets:
            name = dataset_cfg["name"]
            if name not in selected_datasets:
                continue

            ds = RegistryFactory.create_dataset(dataset_cfg)
            dataset_meta[name] = ds.metadata()
            max_items = dataset_cfg.get("max_items")
            try:
                loaded_items = ds.load_items(max_items=max_items)
            except DatasetUnavailableError as exc:
                for system_name in selected_systems:
                    items_out.append(
                        {
                            "item_id": f"{name}-dataset-unavailable",
                            "dataset": name,
                            "system": system_name,
                            "status": "skipped",
                            "latency_ms": 0.0,
                            "question": "",
                            "answer": "",
                            "ground_truth": "",
                            "evidence": [],
                            "error": str(exc),
                            "score": 0.0,
                            "judgment": "skipped_dataset_unavailable",
                            "evaluator_mode": "none",
                        }
                    )
                continue

            adapter_ingest_failures: dict[str, str] = {}
            for system_name, adapter in adapters.items():
                try:
                    adapter.reset()
                    adapter.ingest(loaded_items)
                except Exception as exc:  # pragma: no cover - network/system variability
                    adapter_ingest_failures[system_name] = f"INGEST_FAILED: {exc}"

            for item in loaded_items:
                for system_name, adapter in adapters.items():
                    if system_name in adapter_ingest_failures:
                        q = None
                    else:
                        try:
                            q = adapter.query(item=item, top_k=profile.top_k)
                        except Exception as exc:  # pragma: no cover - network/system variability
                            q = None
                            adapter_ingest_failures[system_name] = f"QUERY_FAILED: {exc}"

                    if q is None:
                        row = {
                            "item_id": item.item_id,
                            "dataset": item.dataset,
                            "system": system_name,
                            "status": "failed",
                            "latency_ms": 0.0,
                            "question": item.question,
                            "answer": "",
                            "ground_truth": item.ground_truth,
                            "evidence": [],
                            "error": adapter_ingest_failures[system_name],
                            "score": 0.0,
                            "judgment": "failed_adapter_error",
                            "evaluator_mode": "none",
                        }
                        items_out.append(row)
                        continue

                    row = {
                        "item_id": item.item_id,
                        "dataset": item.dataset,
                        "system": system_name,
                        "status": q.status,
                        "latency_ms": round(float(q.latency_ms), 6),
                        "question": item.question,
                        "answer": q.answer,
                        "ground_truth": item.ground_truth,
                        "evidence": q.evidence,
                        "error": q.error,
                    }
                    if q.status == "ok":
                        scored = evaluator.score(item=item, answer=q.answer, evidence=q.evidence)
                        row["score"] = scored.score
                        row["judgment"] = scored.judgment
                        row["evaluator_mode"] = scored.evaluator_mode
                    else:
                        row["score"] = 0.0
                        row["judgment"] = q.status
                        row["evaluator_mode"] = "none"
                    items_out.append(row)
        return items_out

    def _profile(self, mode: str) -> ProfileConfig:
        raw = self.config["profiles"][mode]
        return ProfileConfig(
            seed=int(raw["seed"]),
            timeout_seconds=int(raw["timeout_seconds"]),
            datasets=list(raw["datasets"]),
            systems=list(raw["systems"]),
            evaluator=dict(raw["evaluator"]),
            top_k=int(raw["top_k"]),
            prompts=dict(raw["prompts"]),
        )

    def _manifest(
        self,
        mode: str,
        profile: ProfileConfig,
        systems: list[str],
        datasets: list[str],
        dataset_meta: dict[str, dict],
        adapter_meta: dict[str, dict],
        repeats: int,
        seeds: list[int],
    ) -> dict[str, Any]:
        return {
            "mode": mode,
            "timestamp_unix": int(time.time()),
            "git_sha": _git_sha(),
            "seed": profile.seed,
            "repeats": repeats,
            "seeds": seeds,
            "systems": systems,
            "datasets": datasets,
            "dataset_meta": dataset_meta,
            "adapter_meta": adapter_meta,
            "answerer_model": profile.evaluator.get("answerer_model", ""),
            "judge_model": profile.evaluator.get("judge_model", ""),
            "evaluator_mode": profile.evaluator.get("mode", "deterministic"),
            "top_k": profile.top_k,
            "prompts": profile.prompts,
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
            },
        }

    def _aggregates(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        per_system_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        per_system_run_scores: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        per_system_runs: dict[str, set[int]] = defaultdict(set)

        for row in items:
            system = row["system"]
            status = row["status"]
            run_index = int(row.get("replicate", 0))
            per_system_runs[system].add(run_index)
            per_system_status[system][status] += 1
            if status == "ok":
                score = float(row.get("score", 0.0))
                per_system_run_scores[system][run_index].append(score)

        systems_payload: dict[str, dict[str, Any]] = {}
        for system in sorted({r["system"] for r in items}):
            runs = sorted(per_system_runs.get(system, {0}))
            run_means: list[float] = []
            for idx in runs:
                run_scores = per_system_run_scores[system].get(idx, [])
                run_means.append(sum(run_scores) / len(run_scores) if run_scores else 0.0)
            avg_score = sum(run_means) / len(run_means) if run_means else 0.0
            stddev = _stddev_population(run_means)
            ci95 = 1.96 * stddev / math.sqrt(len(run_means)) if run_means else 0.0
            systems_payload[system] = {
                "avg_score": round(avg_score, 6),
                "score_stddev": round(stddev, 6),
                "score_ci95": round(ci95, 6),
                "run_count": len(run_means),
                "ok": int(per_system_status[system].get("ok", 0)),
                "skipped": int(per_system_status[system].get("skipped", 0)),
                "failed": int(per_system_status[system].get("failed", 0)),
            }

        return {"systems": systems_payload}

    def _status_summary(self, items: list[dict[str, Any]]) -> dict[str, int]:
        summary: dict[str, int] = defaultdict(int)
        for row in items:
            summary[row["status"]] += 1
        return dict(summary)

    def _comparisons(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        system_scores: dict[str, list[float]] = defaultdict(list)
        systems_present = sorted({row["system"] for row in items})
        for row in items:
            if row["status"] == "ok":
                system_scores[row["system"]].append(float(row.get("score", 0.0)))

        avg = {}
        for system in systems_present:
            scores = system_scores.get(system, [])
            avg[system] = sum(scores) / len(scores) if scores else 0.0

        baseline = avg.get("tardigrade", max(avg.values()))
        pairwise = {
            system: {
                "avg_score": round(score, 6),
                "delta_vs_tardigrade": round(score - baseline, 6),
            }
            for system, score in sorted(avg.items())
        }
        return {"pairwise": pairwise}

    @staticmethod
    def _resolve_seeds(base_seed: int, repeat: int, seeds: list[int] | None) -> list[int]:
        if seeds:
            if repeat != 1 and len(seeds) != repeat:
                raise ValueError("repeat must match number of explicit seeds")
            return [int(s) for s in seeds]
        return [base_seed + i for i in range(repeat)]


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    return value


def _stddev_population(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
