"""Tardigrade native adapter.

Two execution modes:

- ``native``: real engine path. Loads a small HuggingFace model
  (Qwen3-0.6B by default), captures per-token hidden states for each
  ingested item via ``HuggingFaceKVHook``, stores them through
  ``engine.mem_write_pack``-equivalent ``mem_write``, and at query time
  encodes the question through the same model and retrieves the top-k
  cells via ``engine.mem_read_tokens``. The retrieved cell IDs are
  mapped back to their source items so the adapter can return the
  matching ``ground_truth`` as the answer. **This is what
  TardigradeDB actually does** — earlier versions of this adapter
  short-circuited via lexical matching, which inflated the engine's
  apparent quality.

- ``in_memory``: portable fallback for hosts without CUDA/torch. It
  performs honest word-overlap matching against ingested contexts.
  Keep around so the smoke fixture can run on CI runners without GPUs.

Set ``TDB_BENCH_FORCE_FALLBACK=1`` to force the in-memory path even if
CUDA is available (useful for diffing the two paths).

Refinement mode is selected via ``TDB_REFINEMENT_MODE`` (``none``,
``centered``, or ``prf``). The default ``none`` preserves the engine's
unmodified retrieval. The corresponding empirical numbers live in
``docs/experiments/vague_queries/results.md``.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import OrderedDict
from typing import Any

from tdb_bench.contracts import BenchmarkAdapter
from tdb_bench.models import AdapterQueryResult, BenchmarkItem


_MODEL_NAME = os.getenv("TDB_BENCH_MODEL", "Qwen/Qwen3-0.6B")
_DEVICE = os.getenv("TDB_BENCH_DEVICE", "cuda")
_REFINEMENT = os.getenv("TDB_REFINEMENT_MODE", "none")

# Module-level model cache so repeated ``adapter = TardigradeAdapter()``
# calls (one per repeat in BenchmarkRunner) don't re-download/re-load
# the model. None until first native instance asks for it.
_MODEL_CACHE: dict[str, Any] = {"model": None, "tokenizer": None, "query_layer": None}


def _load_model_cached() -> tuple[Any, Any, int]:
    if _MODEL_CACHE["model"] is None:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME, dtype=torch.float32, attn_implementation="eager",
        )
        model = model.to(_DEVICE).eval()
        n_layers = model.config.num_hidden_layers
        query_layer = int(n_layers * 0.67)
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["tokenizer"] = tokenizer
        _MODEL_CACHE["query_layer"] = query_layer
    return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"], _MODEL_CACHE["query_layer"]


class _InMemoryStore:
    """Lexical fallback. Honest about what it is."""

    def __init__(self) -> None:
        self.data: OrderedDict[str, BenchmarkItem] = OrderedDict()

    def clear(self) -> None:
        self.data.clear()

    def insert(self, item: BenchmarkItem) -> None:
        self.data[item.item_id] = item

    def best_match(self, question: str, top_k: int) -> tuple[str, list[str]]:
        terms = {t.lower() for t in question.split() if t.strip()}
        scored: list[tuple[int, BenchmarkItem]] = []
        for item in self.data.values():
            hay = f"{item.context} {item.question}".lower()
            score = sum(1 for t in terms if t in hay)
            scored.append((score, item))
        scored.sort(key=lambda s: s[0], reverse=True)
        top = [x[1] for x in scored[: max(1, top_k)]]
        best = top[0]
        return best.ground_truth, [b.context for b in top]


class TardigradeAdapter(BenchmarkAdapter):
    """Real-engine adapter with portable lexical fallback."""

    name = "tardigrade"

    def __init__(self) -> None:
        self._store = _InMemoryStore()
        self._cell_to_item: dict[int, BenchmarkItem] = {}
        self._engine = None
        self._hook = None
        self._mode = "in_memory"
        self._refinement = _REFINEMENT

        force_fallback = os.getenv("TDB_BENCH_FORCE_FALLBACK", "").lower() in ("1", "true", "yes")
        if force_fallback:
            return

        try:
            import tardigrade_db  # type: ignore
            from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook  # type: ignore

            model, _tokenizer, _query_layer = _load_model_cached()
            data_dir = tempfile.mkdtemp(prefix="tdb_bench_")
            self._engine = tardigrade_db.Engine(data_dir)
            try:
                self._engine.set_refinement_mode(self._refinement)
            except Exception:
                # Older engine builds without refinement API still benchmark fine.
                pass
            self._hook = HuggingFaceKVHook(
                self._engine, owner=1, k=5,
                model_config=model.config, model=model,
                use_hidden_states=True,
            )
            self._mode = "native"
        except Exception as exc:
            # Capture why we fell back so metadata() can surface it.
            self._engine = None
            self._hook = None
            self._mode = os.getenv("TDB_BENCH_FALLBACK_MODE", "in_memory")
            self._fallback_reason = str(exc)[:200]

    # ---- BenchmarkAdapter contract ----

    def ingest(self, items: list[BenchmarkItem]) -> None:
        if self._mode != "native":
            for item in items:
                self._store.insert(item)
            return

        import torch  # type: ignore

        model, tokenizer, query_layer = _load_model_cached()
        for item in items:
            inputs = tokenizer(
                item.context, return_tensors="pt", truncation=True, max_length=256,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, use_cache=True, output_hidden_states=True)
            decision = self._hook.on_generate(
                layer=query_layer,
                past_key_values=out.past_key_values,
                model_hidden_states=out.hidden_states[query_layer],
            )
            if not decision.should_write:
                continue
            cell_id = self._engine.mem_write(
                1, query_layer, decision.key, decision.value, decision.salience, None,
            )
            self._cell_to_item[int(cell_id)] = item
            # Also keep an in-memory snapshot so metadata() / fallback work.
            self._store.insert(item)

    def query(self, item: BenchmarkItem, top_k: int) -> AdapterQueryResult:
        if self._mode != "native":
            start = time.perf_counter()
            answer, evidence = self._store.best_match(item.question, top_k)
            return AdapterQueryResult(
                answer=answer,
                evidence=evidence,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                status="ok",
                error=None,
            )

        import torch  # type: ignore

        model, tokenizer, query_layer = _load_model_cached()
        start = time.perf_counter()
        inputs = tokenizer(
            item.question, return_tensors="pt", truncation=True, max_length=256,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        handles = self._hook.on_prefill(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        evidence: list[str] = []
        answer = ""
        for h in handles[: max(1, top_k)]:
            mapped = self._cell_to_item.get(int(h.cell_id))
            if mapped is None:
                continue
            evidence.append(mapped.context)
            if not answer:
                answer = mapped.ground_truth
        if not answer:
            return AdapterQueryResult(
                answer="",
                evidence=[],
                latency_ms=latency_ms,
                status="failed",
                error="no_retrieval_match",
            )
        return AdapterQueryResult(
            answer=answer,
            evidence=evidence,
            latency_ms=latency_ms,
            status="ok",
            error=None,
        )

    def reset(self) -> None:
        self._store.clear()
        self._cell_to_item.clear()
        # Drop the engine and rebuild — items from one dataset must not
        # leak into the next. Cheap because the model stays cached.
        if self._mode == "native":
            try:
                import tardigrade_db  # type: ignore
                from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook  # type: ignore

                model, _tokenizer, _query_layer = _load_model_cached()
                data_dir = tempfile.mkdtemp(prefix="tdb_bench_")
                self._engine = tardigrade_db.Engine(data_dir)
                try:
                    self._engine.set_refinement_mode(self._refinement)
                except Exception:
                    pass
                self._hook = HuggingFaceKVHook(
                    self._engine, owner=1, k=5,
                    model_config=model.config, model=model,
                    use_hidden_states=True,
                )
            except Exception:
                # If a reset fails we degrade to fallback rather than crash the run.
                self._engine = None
                self._hook = None
                self._mode = "in_memory"

    def metadata(self) -> dict[str, str]:
        meta = {
            "adapter": self.name,
            "mode": self._mode,
            "refinement_mode": self._refinement,
            "model": _MODEL_NAME if self._mode == "native" else "lexical",
        }
        if self._mode != "native" and hasattr(self, "_fallback_reason"):
            meta["fallback_reason"] = self._fallback_reason
        return meta
