"""Benchmark runner (Template Method orchestration)."""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from tdb_bench.errors import DatasetUnavailableError
from tdb_bench.fairness import validate_fairness
from tdb_bench.models import RunResultV1
from tdb_bench.registry import RegistryFactory
from tdb_bench.schema import validate_run_result_v1


# Default worker count for the parallel runner codepath. Workers > 1
# dispatches each (item, system) tuple to a ThreadPoolExecutor; the
# engine's Arc<Mutex<>> + TardigradeAdapter._gpu_lock serialize the
# GPU section while LLM calls (~93% of per-item wall time) run in
# parallel. 1 preserves the existing sequential codepath.
DEFAULT_WORKERS = 1
WORKERS_ENV = "TDB_BENCH_WORKERS"


def _resolve_workers(explicit: int | None) -> int:
    """Pick explicit > env > default."""
    if explicit is not None and explicit >= 1:
        return explicit
    raw = os.getenv(WORKERS_ENV, "").strip()
    if raw:
        try:
            n = int(raw)
            if n >= 1:
                return n
        except ValueError:
            pass
    return DEFAULT_WORKERS


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
        workers: int | None = None,
    ) -> RunResultV1:
        profile = self._profile(mode)
        resolved_workers = _resolve_workers(workers)
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
                workers=resolved_workers,
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
        workers: int = DEFAULT_WORKERS,
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
                            "category": "unknown",
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

            _total = len(loaded_items)
            if workers > 1:
                ds_items = self._process_parallel(
                    loaded_items=loaded_items,
                    adapters=adapters,
                    adapter_ingest_failures=adapter_ingest_failures,
                    evaluator=evaluator,
                    top_k=profile.top_k,
                    workers=workers,
                    dataset_total=_total,
                )
            else:
                ds_items = self._process_sequential(
                    loaded_items=loaded_items,
                    adapters=adapters,
                    adapter_ingest_failures=adapter_ingest_failures,
                    evaluator=evaluator,
                    top_k=profile.top_k,
                    dataset_total=_total,
                )
            items_out.extend(ds_items)
        return items_out

    # ─── Phase orchestration ──────────────────────────────────────────

    def _process_sequential(
        self,
        *,
        loaded_items: list[Any],
        adapters: dict[str, Any],
        adapter_ingest_failures: dict[str, str],
        evaluator: Any,
        top_k: int,
        dataset_total: int,
    ) -> list[dict[str, Any]]:
        """Existing single-threaded path. workers=1 preserves prior behavior."""
        items_out: list[dict[str, Any]] = []
        _scored_count = 0
        _score_sum = 0.0
        for _idx, item in enumerate(loaded_items, 1):
            for system_name, adapter in adapters.items():
                row, scored_delta = self._process_one(
                    item=item,
                    system_name=system_name,
                    adapter=adapter,
                    evaluator=evaluator,
                    top_k=top_k,
                    failures=adapter_ingest_failures,
                )
                items_out.append(row)
                if scored_delta is not None:
                    _scored_count += 1
                    _score_sum += scored_delta
                    _avg = _score_sum / _scored_count
                    print(
                        f"[{_idx}/{dataset_total}] {item.dataset}/{item.item_id} "
                        f"score={scored_delta:.2f} avg={_avg:.4f} "
                        f"latency={row['latency_ms']:.0f}ms",
                        flush=True,
                    )
        return items_out

    def _process_parallel(
        self,
        *,
        loaded_items: list[Any],
        adapters: dict[str, Any],
        adapter_ingest_failures: dict[str, str],
        evaluator: Any,
        top_k: int,
        workers: int,
        dataset_total: int,
    ) -> list[dict[str, Any]]:
        """Multi-threaded path. Each (item, adapter) tuple is dispatched
        to a ThreadPoolExecutor. The adapter's internal GPU lock + the
        engine's Arc<Mutex<>> serialize Rust+torch calls; LLM calls
        run in parallel across threads.

        Output is sorted by (dataset, item_id, system) so results are
        order-deterministic regardless of completion order.
        """
        items_out: list[dict[str, Any]] = []
        progress_lock = threading.Lock()
        scored_count = 0
        score_sum = 0.0
        completed_idx = 0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for item in loaded_items:
                for system_name, adapter in adapters.items():
                    fut = ex.submit(
                        self._process_one,
                        item=item,
                        system_name=system_name,
                        adapter=adapter,
                        evaluator=evaluator,
                        top_k=top_k,
                        failures=adapter_ingest_failures,
                    )
                    futures[fut] = item

            for fut in as_completed(futures):
                row, scored_delta = fut.result()
                items_out.append(row)
                with progress_lock:
                    completed_idx += 1
                    if scored_delta is not None:
                        scored_count += 1
                        score_sum += scored_delta
                        _avg = score_sum / scored_count
                        print(
                            f"[{completed_idx}/{dataset_total}] "
                            f"{row['dataset']}/{row['item_id']} "
                            f"score={scored_delta:.2f} avg={_avg:.4f} "
                            f"latency={row['latency_ms']:.0f}ms",
                            flush=True,
                        )

        # Deterministic order so the run JSON is diff-friendly.
        items_out.sort(key=lambda r: (r["dataset"], r["item_id"], r["system"]))
        return items_out

    @staticmethod
    def _process_one(
        *,
        item: Any,
        system_name: str,
        adapter: Any,
        evaluator: Any,
        top_k: int,
        failures: dict[str, str],
    ) -> tuple[dict[str, Any], float | None]:
        """Run adapter.query + evaluator.score for one (item, system).

        Returns ``(row_dict, scored_delta_or_None)``. Safe to call
        concurrently — the adapter's internal locks serialize shared
        state; LLM calls are network-bound.
        """
        if system_name in failures:
            return (
                {
                    "item_id": item.item_id,
                    "dataset": item.dataset,
                    "category": getattr(item, "category", "unknown"),
                    "system": system_name,
                    "status": "failed",
                    "latency_ms": 0.0,
                    "question": item.question,
                    "answer": "",
                    "ground_truth": item.ground_truth,
                    "evidence": [],
                    "error": failures[system_name],
                    "score": 0.0,
                    "judgment": "failed_adapter_error",
                    "evaluator_mode": "none",
                },
                None,
            )

        try:
            q = adapter.query(item=item, top_k=top_k)
        except Exception as exc:  # pragma: no cover — network/system variability
            failures[system_name] = f"QUERY_FAILED: {exc}"
            return (
                {
                    "item_id": item.item_id,
                    "dataset": item.dataset,
                    "category": getattr(item, "category", "unknown"),
                    "system": system_name,
                    "status": "failed",
                    "latency_ms": 0.0,
                    "question": item.question,
                    "answer": "",
                    "ground_truth": item.ground_truth,
                    "evidence": [],
                    "error": failures[system_name],
                    "score": 0.0,
                    "judgment": "failed_adapter_error",
                    "evaluator_mode": "none",
                },
                None,
            )

        row = {
            "item_id": item.item_id,
            "dataset": item.dataset,
            "category": getattr(item, "category", "unknown"),
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
            return row, scored.score
        row["score"] = 0.0
        row["judgment"] = q.status
        row["evaluator_mode"] = "none"
        return row, None

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
                "by_category": self._per_category_breakdown(items, system),
            }

        return {"systems": systems_payload}

    @staticmethod
    def _per_category_breakdown(
        items: list[dict[str, Any]], system: str
    ) -> dict[str, dict[str, Any]]:
        """Group ``ok`` items for ``system`` by ``(dataset, category)``.

        Returns a dict keyed by ``"{dataset}/{category}"`` so a single
        system's payload can carry both LoCoMo's ``single_hop``…
        ``adversarial`` slices and LongMemEval's ``temporal-reasoning``,
        ``multi-session``, … slices side by side. The key includes the
        dataset prefix because LongMemEval's ``temporal-reasoning`` and
        LoCoMo's ``temporal`` measure different things and must not be
        merged. Phase 1B audit 2026-05-16 #89.
        """
        buckets: dict[str, list[float]] = defaultdict(list)
        all_for_system: dict[str, int] = defaultdict(int)
        for row in items:
            if row.get("system") != system:
                continue
            key = f"{row.get('dataset', 'unknown')}/{row.get('category', 'unknown')}"
            all_for_system[key] += 1
            if row.get("status") == "ok":
                buckets[key].append(float(row.get("score", 0.0)))
        out: dict[str, dict[str, Any]] = {}
        for key, count in sorted(all_for_system.items()):
            scores = buckets.get(key, [])
            avg = sum(scores) / len(scores) if scores else 0.0
            out[key] = {
                "n": count,
                "ok": len(scores),
                "avg_score": round(avg, 6),
            }
        return out

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
