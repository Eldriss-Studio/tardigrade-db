"""Per-model query-layer calibration for `KnowledgePackStore`.

Design patterns: **Factory Method** + **Strategy** (plus an external
**Repository** in :mod:`tardigrade_hooks.calibration_registry`).

::

                       ┌────────────────────────────┐
                       │  select_query_layer(...)   │  Factory Method
                       │  - check registry cache    │
                       │  - else run strategy       │
                       │  - else default ratio      │
                       └────────────┬───────────────┘
                                    │
                                    ▼
                       ┌────────────────────────────┐
                       │  CalibrationStrategy (ABC) │  Strategy
                       │     run(model, tok, corpus)│
                       └────────────┬───────────────┘
                                    │
                                    ▼
                       ┌────────────────────────────┐
                       │   LinearSweepStrategy      │  Concrete
                       │   - forward each fact once │
                       │   - forward each query     │
                       │   - sweep all layer indices│
                       │   - score top-1 / top-5    │
                       └────────────────────────────┘

# Why this exists

``KnowledgePackStore`` defaults the retrieval-key layer to
``int(num_layers * DEFAULT_CAPTURE_LAYER_RATIO)`` (0.67). On uniform-
softmax models (Qwen3, Llama-3, Mistral, Gemma-2) this works fine —
nearly any layer in the upper half encodes retrieval-discriminative
information. On **hybrid models** (Qwen3-Next, RecurrentGemma, Jamba,
Zamba, Falcon-Mamba, Granite-4, MiniMax, Hunyuan-T1, Nemotron-H, IBM
Bamba) most layers are linear / SSM / recurrent — and per
`Michalak & Abreu 2025 (arXiv:2510.19861)`_ and
`De et al. 2024 (Griffin paper) §6.2 <https://arxiv.org/abs/2402.19427>`_,
those layers carry no retrieval signal. Picking ``0.67 × num_layers``
on a hybrid model can land on a recurrent layer and produce ~0%
top-1 recall — the engine returns the same few "popular" packs for
every query.

The fix is empirical: run a small sweep, find the layer that actually
discriminates, and remember it. That's what this module does.

# Example

>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
>>> result = select_query_layer(model, tok)
>>> result.best_layer
17  # actual best layer varies per model

With persistence:

>>> from tardigrade_hooks import CalibrationRegistry
>>> reg = CalibrationRegistry()  # ~/.tardigrade/calibration.json
>>> result = select_query_layer(model, tok, registry=reg)
>>> # second call returns the cached result without re-running the sweep
>>> result = select_query_layer(model, tok, registry=reg)

.. _Michalak & Abreu 2025 (arXiv:2510.19861): https://arxiv.org/abs/2510.19861
"""

from __future__ import annotations

import datetime
import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import tardigrade_db

#: Logger for calibration progress. Calibration runs can take 30 s to a
#: few minutes, mostly silent. Enable visibility by configuring this
#: logger (e.g., ``logging.getLogger("tardigrade_hooks.calibrate").setLevel(
#: logging.INFO)``). Defaults to ``WARNING`` so the library is quiet by
#: default and only complains about real problems.
logger = logging.getLogger(__name__)

from ._calibration_corpus import DEFAULT_CORPUS
from ._hidden_states import (
    _compute_per_layer_hidden_states,
    layer_kind_labels,
)
from .constants import DEFAULT_CAPTURE_LAYER_RATIO
from .encoding import encode_per_token

_CALIBRATION_OWNER: int = 1  # Owner id used for in-tempdir calibration engines.
_DEFAULT_TOP_K: int = 5


@dataclass(frozen=True)
class LayerScore:
    """Retrieval recall for a single candidate layer.

    Attributes:
        layer: Index into ``output.hidden_states`` — 0 is embeddings,
            1..n_layers are per-layer outputs.
        kind: One of ``"embedding"``, ``"attention"``, ``"recurrent"``,
            or a raw config string (truncated) for unknown layer types.
        top1: Number of queries for which the expected pack was the
            engine's top-1 result.
        top5: Number of queries for which the expected pack was within
            the engine's top-5 results.
    """

    layer: int
    kind: str
    top1: int
    top5: int

    def as_dict(self) -> dict:
        return {"layer": self.layer, "kind": self.kind,
                "top1": self.top1, "top5": self.top5}

    @classmethod
    def from_dict(cls, d: dict) -> "LayerScore":
        return cls(layer=int(d["layer"]), kind=str(d["kind"]),
                   top1=int(d["top1"]), top5=int(d["top5"]))


@dataclass(frozen=True)
class CalibrationResult:
    """One full calibration run's findings for a specific model.

    Records the best ``query_layer`` for retrieval plus the per-layer
    scores that justified the choice and enough metadata to verify the
    result is still applicable (model id, library version, timestamp).
    """

    model_id: str
    tardigrade_db_version: str
    timestamp_iso: str
    n_layers: int
    hidden_size: int
    best_layer: int
    scores: tuple[LayerScore, ...]

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "tardigrade_db_version": self.tardigrade_db_version,
            "timestamp_iso": self.timestamp_iso,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "best_layer": self.best_layer,
            "scores": [s.as_dict() for s in self.scores],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationResult":
        return cls(
            model_id=str(d["model_id"]),
            tardigrade_db_version=str(d["tardigrade_db_version"]),
            timestamp_iso=str(d["timestamp_iso"]),
            n_layers=int(d["n_layers"]),
            hidden_size=int(d["hidden_size"]),
            best_layer=int(d["best_layer"]),
            scores=tuple(LayerScore.from_dict(s) for s in d["scores"]),
        )


class CalibrationStrategy(ABC):
    """Algorithm for finding the best retrieval-key layer.

    Subclasses implement different search strategies. The shipped
    :class:`LinearSweepStrategy` is exhaustive (tries every layer
    index). Future strategies may bisect, sweep only attention layers
    on hybrid models, or use a learned per-architecture prior.
    """

    @abstractmethod
    def run(
        self,
        model: Any,
        tokenizer: Any,
        corpus: list[tuple[str, str]],
        *,
        layer_filter: Callable[[int, str], bool] | None = None,
    ) -> CalibrationResult:
        """Run the sweep on this corpus; return the best layer + per-layer scores.

        Args:
            model: A HuggingFace causal-LM with ``output_hidden_states``
                support.
            tokenizer: The matching tokenizer (chat-template-capable).
            corpus: List of ``(fact_text, query_text)`` pairs.
            layer_filter: Optional predicate ``(layer_idx, kind) -> bool``
                that, when False, skips scoring that layer (still records
                it with score 0).
        """


class LinearSweepStrategy(CalibrationStrategy):
    """Exhaustive sweep: try every layer index, pick the highest top-1 (tiebreak top-5)."""

    def run(
        self,
        model: Any,
        tokenizer: Any,
        corpus: list[tuple[str, str]],
        *,
        layer_filter: Callable[[int, str], bool] | None = None,
    ) -> CalibrationResult:
        if not corpus:
            raise ValueError("CalibrationStrategy.run requires a non-empty corpus")

        # Lazy import to avoid forcing the adapter on consumers that
        # never call calibration.
        from .chat_template_adapter import select_chat_template_adapter

        adapter = select_chat_template_adapter(tokenizer)
        cfg = model.config
        n_layers = int(cfg.num_hidden_layers)
        hidden_size = int(cfg.hidden_size)

        model_id = _model_id_for(model)
        n_corpus = len(corpus)
        logger.info(
            "calibrating %s: forward-pass phase (%d facts + %d queries)",
            model_id, n_corpus, n_corpus,
        )

        # Phase 1: cache per-layer hidden states for every fact (with
        # chat-template wrap) and every query (without wrap — mirrors
        # the library's retrieval path).
        fact_hidden: list[list[np.ndarray]] = []
        fact_payloads: list[list] = []
        for i, (fact_text, _) in enumerate(corpus, 1):
            hs, payloads, _ = _compute_per_layer_hidden_states(
                model, tokenizer, fact_text, wrap_chat=True, adapter=adapter
            )
            fact_hidden.append(hs)
            fact_payloads.append(payloads)
            if i % 5 == 0 or i == n_corpus:
                logger.info("  facts forwarded: %d/%d", i, n_corpus)

        query_hidden: list[list[np.ndarray]] = []
        for i, (_, query_text) in enumerate(corpus, 1):
            hs, _, _ = _compute_per_layer_hidden_states(
                model, tokenizer, query_text, wrap_chat=False, adapter=adapter
            )
            query_hidden.append(hs)
            if i % 5 == 0 or i == n_corpus:
                logger.info("  queries forwarded: %d/%d", i, n_corpus)

        n_hidden_states = len(fact_hidden[0])
        kinds = layer_kind_labels(cfg, n_hidden_states)
        logger.info(
            "calibrating %s: layer sweep (%d candidate layers)",
            model_id, n_hidden_states,
        )

        # Phase 2: sweep every layer index, run engine round-trip per layer.
        scores: list[LayerScore] = []
        running_best: tuple[int, int] = (-1, -1)
        for li in range(n_hidden_states):
            kind = kinds[li]
            if layer_filter is not None and not layer_filter(li, kind):
                scores.append(LayerScore(layer=li, kind=kind, top1=0, top5=0))
                logger.debug("  layer %d (%s): skipped by filter", li, kind)
                continue
            top1, top5 = self._score_one_layer(
                li, fact_hidden, query_hidden, hidden_size
            )
            scores.append(LayerScore(layer=li, kind=kind, top1=top1, top5=top5))
            is_new_best = (top1, top5) > running_best
            running_best = max(running_best, (top1, top5))
            logger.info(
                "  layer %2d (%-9s) top-1 %2d/%d  top-5 %2d/%d%s",
                li, kind, top1, n_corpus, top5, n_corpus,
                "  ★ new best" if is_new_best else "",
            )

        # Phase 3: pick best — highest top-1, tiebreak by top-5.
        valid = [s for s in scores]
        best = max(valid, key=lambda s: (s.top1, s.top5))
        logger.info(
            "calibrating %s: best layer %d (%s) — top-1 %d/%d, top-5 %d/%d",
            model_id, best.layer, best.kind, best.top1, n_corpus,
            best.top5, n_corpus,
        )

        return CalibrationResult(
            model_id=_model_id_for(model),
            tardigrade_db_version=_current_library_version(),
            timestamp_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            n_layers=n_layers,
            hidden_size=hidden_size,
            best_layer=best.layer,
            scores=tuple(scores),
        )

    @staticmethod
    def _score_one_layer(
        layer_idx: int,
        fact_hidden: list[list[np.ndarray]],
        query_hidden: list[list[np.ndarray]],
        hidden_size: int,
    ) -> tuple[int, int]:
        """Store all facts in a fresh engine using this layer's hidden state
        as the retrieval key; query all queries; count top-1 / top-5 hits."""
        n = len(fact_hidden)
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = tardigrade_db.Engine(tmpdir)
            pack_ids: dict[int, int] = {}
            for i in range(n):
                h = fact_hidden[i][layer_idx]
                if h.shape[0] < 2:
                    return 0, 0  # degenerate — can't drop pos 0
                ret_key = encode_per_token(h[1:], hidden_size)
                # Empty layer_payloads — calibration tests retrieval only,
                # not injection. Pack carries retrieval_key + text only.
                pid = engine.mem_write_pack(
                    _CALIBRATION_OWNER, ret_key, [], 50.0,
                    text=f"calibration-fact-{i}",
                )
                pack_ids[i] = pid

            top1 = 0
            top5 = 0
            for i in range(n):
                h = query_hidden[i][layer_idx]
                if h.shape[0] < 2:
                    continue
                qkey = encode_per_token(h[1:], hidden_size)
                packs = engine.mem_read_pack(qkey, _DEFAULT_TOP_K, _CALIBRATION_OWNER)
                ids = [p["pack_id"] for p in packs]
                if ids[:1] == [pack_ids[i]]:
                    top1 += 1
                if pack_ids[i] in ids:
                    top5 += 1
            return top1, top5


def select_query_layer(
    model: Any,
    tokenizer: Any,
    *,
    registry: Any = None,
    corpus: list[tuple[str, str]] | None = None,
    strategy: CalibrationStrategy | None = None,
) -> CalibrationResult:
    """Factory Method: return the best retrieval-key layer for this model.

    Lookup order:

    1. If ``registry`` is provided and has a cached result for this model's
       id, return the cached result (no sweep).
    2. If ``tokenizer`` is ``None`` (hosted-API mode) — return a no-sweep
       result whose ``best_layer`` is the library default
       (``int(n_layers * DEFAULT_CAPTURE_LAYER_RATIO)``).
    3. Otherwise run ``strategy.run(model, tokenizer, corpus)`` (default
       :class:`LinearSweepStrategy`). If ``registry`` was provided, save
       the result before returning.

    Args:
        model: A HuggingFace causal-LM.
        tokenizer: The matching tokenizer, or ``None`` for hosted-API mode.
        registry: Optional :class:`CalibrationRegistry` for persistence.
        corpus: Optional list of ``(fact, query)`` pairs. Defaults to the
            bundled 20-fact synthetic corpus.
        strategy: Optional :class:`CalibrationStrategy` instance. Defaults
            to :class:`LinearSweepStrategy`.

    Returns:
        :class:`CalibrationResult` whose ``best_layer`` should be passed
        to ``KnowledgePackStore(..., query_layer=result.best_layer)``.
    """
    model_id = _model_id_for(model)

    if registry is not None:
        cached = registry.load(model_id)
        if cached is not None:
            return cached

    if tokenizer is None:
        n_layers = int(model.config.num_hidden_layers)
        return CalibrationResult(
            model_id=model_id,
            tardigrade_db_version=_current_library_version(),
            timestamp_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            n_layers=n_layers,
            hidden_size=int(model.config.hidden_size),
            best_layer=int(n_layers * DEFAULT_CAPTURE_LAYER_RATIO),
            scores=(),
        )

    if strategy is None:
        strategy = LinearSweepStrategy()
    if corpus is None:
        corpus = list(DEFAULT_CORPUS)

    result = strategy.run(model, tokenizer, corpus)

    if registry is not None:
        registry.save(result)
    return result


def _model_id_for(model: Any) -> str:
    """Best-effort identifier for a HuggingFace model.

    Prefers ``model.config.name_or_path`` (HF's canonical hub id).
    Falls back to the model class name.
    """
    cfg = getattr(model, "config", None)
    nop = getattr(cfg, "name_or_path", None)
    if nop:
        return str(nop)
    return f"unknown:{type(model).__name__}"


def _current_library_version() -> str:
    """Return tardigrade_db's currently-installed library version."""
    return getattr(tardigrade_db, "__version__", "unknown")
