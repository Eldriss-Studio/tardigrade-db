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

# Note: `tardigrade_db` (the compiled Rust extension) is intentionally
# *not* imported at module level. CI lint jobs intentionally don't
# build the native extension, so eager-importing it here would break
# `from tardigrade_hooks.calibrate import …` for every consumer that
# doesn't actually run calibration. The two call sites that genuinely
# need it (`_score_strategy` for `Engine`, `_current_library_version`
# for `__version__`) import it locally inside the function body.

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
from .constants import CALIBRATION_SALIENCE, DEFAULT_CAPTURE_LAYER_RATIO
from .encoding import encode_per_token

_CALIBRATION_OWNER: int = 1  # Owner id used for in-tempdir calibration engines.
_DEFAULT_TOP_K: int = 5


@dataclass(frozen=True)
class LayerScore:
    """Retrieval recall for a single ``(strategy, layer)`` candidate.

    Attributes:
        layer: Index into ``output.hidden_states`` or ``cache.layers`` —
            interpretation depends on ``strategy``. For
            ``"hidden_state"``, 0 is embeddings and 1..n are per-layer
            outputs of ``output.hidden_states``. For ``"k_vector"``, it
            is the softmax cache layer index in ``cache.layers``.
        kind: One of ``"embedding"``, ``"attention"``, ``"recurrent"``,
            or a raw config string (truncated) for unknown layer types.
        strategy: Retrieval-key strategy identifier — ``"hidden_state"``
            for :class:`tardigrade_hooks.HiddenStateKeyStrategy`, or
            ``"k_vector"`` for :class:`tardigrade_hooks.KVectorKeyStrategy`.
            Older calibration records lack this field; deserialisation
            defaults to ``"hidden_state"`` for backwards compatibility.
        top1: Number of queries for which the expected pack was the
            engine's top-1 result.
        top5: Number of queries for which the expected pack was within
            the engine's top-5 results.
    """

    layer: int
    kind: str
    top1: int
    top5: int
    strategy: str = "hidden_state"

    def as_dict(self) -> dict:
        return {"layer": self.layer, "kind": self.kind,
                "strategy": self.strategy,
                "top1": self.top1, "top5": self.top5}

    @classmethod
    def from_dict(cls, d: dict) -> "LayerScore":
        return cls(layer=int(d["layer"]), kind=str(d["kind"]),
                   top1=int(d["top1"]), top5=int(d["top5"]),
                   strategy=str(d.get("strategy", "hidden_state")))


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
    best_strategy: str = "hidden_state"

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "tardigrade_db_version": self.tardigrade_db_version,
            "timestamp_iso": self.timestamp_iso,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "best_layer": self.best_layer,
            "best_strategy": self.best_strategy,
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
            best_strategy=str(d.get("best_strategy", "hidden_state")),
            scores=tuple(LayerScore.from_dict(s) for s in d["scores"]),
        )

    def best_score(self) -> LayerScore:
        """Return the [`LayerScore`] for ``(best_strategy, best_layer)``.

        Looks up by the (strategy, layer) pair rather than positional
        index. ``scores`` is enumeration-ordered, NOT keyed by layer
        index — ``scores[best_layer]`` returns the right entry only when
        a single strategy enumerates one score per layer, which fails
        silently the moment multi-strategy enumeration kicks in.
        """
        for s in self.scores:
            if s.layer == self.best_layer and s.strategy == self.best_strategy:
                return s
        raise LookupError(
            f"no score with (layer={self.best_layer}, "
            f"strategy={self.best_strategy!r}) found in scores — "
            f"best_layer/best_strategy and scores got out of sync"
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

        # Lazy imports to avoid forcing the adapter / strategy modules
        # on consumers that never call calibration.
        from .chat_template_adapter import select_chat_template_adapter
        from ._hidden_states import _is_softmax_cache_layer
        from .retrieval_key_strategy import (
            HiddenStateKeyStrategy,
            KVectorKeyStrategy,
            RetrievalKeyStrategy,
        )

        from ._hidden_states import _text_config

        adapter = select_chat_template_adapter(tokenizer)
        # Drill into text_config for multimodal models (Gemma 3, Llama 3.2
        # Vision, …). Top-level config returned for text-only.
        cfg = _text_config(model.config)
        n_layers = int(cfg.num_hidden_layers)
        hidden_size = int(cfg.hidden_size)

        model_id = _model_id_for(model)
        n_corpus = len(corpus)
        logger.info(
            "calibrating %s: forward-pass phase (%d facts + %d queries)",
            model_id, n_corpus, n_corpus,
        )

        # Phase 1: cache per-layer hidden states + KV cache for every
        # fact (with chat-template wrap) and every query (without wrap
        # — mirrors the library's retrieval path).
        fact_hidden: list[list[np.ndarray]] = []
        fact_caches: list[Any] = []
        for i, (fact_text, _) in enumerate(corpus, 1):
            hs, _payloads, _seq_len, kv = _compute_per_layer_hidden_states(
                model, tokenizer, fact_text, wrap_chat=True, adapter=adapter,
                return_cache=True,
            )
            fact_hidden.append(hs)
            fact_caches.append(kv)
            if i % 5 == 0 or i == n_corpus:
                logger.info("  facts forwarded: %d/%d", i, n_corpus)

        query_hidden: list[list[np.ndarray]] = []
        query_caches: list[Any] = []
        for i, (_, query_text) in enumerate(corpus, 1):
            hs, _payloads, _seq_len, kv = _compute_per_layer_hidden_states(
                model, tokenizer, query_text, wrap_chat=False, adapter=adapter,
                return_cache=True,
            )
            query_hidden.append(hs)
            query_caches.append(kv)
            if i % 5 == 0 or i == n_corpus:
                logger.info("  queries forwarded: %d/%d", i, n_corpus)

        n_hidden_states = len(fact_hidden[0])
        kinds = layer_kind_labels(cfg, n_hidden_states)
        # Identify softmax-attention cache layer indices for K-vector
        # strategy candidates. cache.layers is indexed 0..n_layers-1
        # (no embedding entry — different convention from hidden_states).
        probe_cache = fact_caches[0]
        softmax_layer_indices = [
            i for i, layer in enumerate(probe_cache.layers)
            if _is_softmax_cache_layer(layer)
        ]

        # Build the candidate list: one HiddenStateKeyStrategy per
        # hidden_states index + one KVectorKeyStrategy per softmax
        # cache-layer index. The two strategies use different layer-
        # indexing conventions but `LayerScore.layer` records the
        # strategy-native index so the (strategy, layer) tuple is
        # unambiguous.
        candidates: list[tuple[str, int, str, RetrievalKeyStrategy]] = []
        for li in range(n_hidden_states):
            if layer_filter is not None and not layer_filter(li, kinds[li]):
                continue
            candidates.append(("hidden_state", li, kinds[li], HiddenStateKeyStrategy(li)))
        # For K-vector, map cache-layer index back to kinds[] which is
        # indexed over hidden_states (cache_layer_i corresponds to
        # hidden_states[i+1] — the output of that layer). Use the
        # hidden_states kind for clarity in the report.
        for li in softmax_layer_indices:
            kind_idx = li + 1  # hidden_states[li+1] is this layer's output
            kind = kinds[kind_idx] if kind_idx < len(kinds) else "attention"
            if layer_filter is not None and not layer_filter(li, kind):
                continue
            candidates.append(("k_vector", li, kind, KVectorKeyStrategy(li)))

        logger.info(
            "calibrating %s: sweep (%d candidates: %d hidden-state + %d k-vector)",
            model_id, len(candidates),
            sum(1 for c in candidates if c[0] == "hidden_state"),
            sum(1 for c in candidates if c[0] == "k_vector"),
        )

        # Phase 2: for each (strategy, layer) candidate, engine round-trip.
        scores: list[LayerScore] = []
        running_best: tuple[int, int] = (-1, -1)
        for strategy_name, li, kind, strategy in candidates:
            try:
                top1, top5 = self._score_strategy(
                    strategy, fact_hidden, fact_caches,
                    query_hidden, query_caches, hidden_size,
                )
            except Exception as exc:
                logger.warning(
                    "  %s@%d (%s) failed: %r; recording as 0/%d",
                    strategy_name, li, kind, exc, n_corpus,
                )
                top1, top5 = 0, 0
            scores.append(LayerScore(
                layer=li, kind=kind, strategy=strategy_name,
                top1=top1, top5=top5,
            ))
            is_new_best = (top1, top5) > running_best
            running_best = max(running_best, (top1, top5))
            logger.info(
                "  %-12s@%2d (%-9s) top-1 %2d/%d  top-5 %2d/%d%s",
                strategy_name, li, kind, top1, n_corpus, top5, n_corpus,
                "  ★ new best" if is_new_best else "",
            )

        # Phase 3: pick best — highest top-1, tiebreak first by top-5,
        # then by layer depth (prefer the deepest layer in any tied
        # set). The depth tiebreak matters because shallow layers
        # — especially the embedding under hidden_state strategy —
        # can ace a small synthetic corpus purely on surface-token
        # discrimination. Deeper layers encode semantic meaning that
        # survives paraphrasing.
        best = max(scores, key=lambda s: (s.top1, s.top5, s.layer))
        logger.info(
            "calibrating %s: best %s@%d (%s) — top-1 %d/%d, top-5 %d/%d",
            model_id, best.strategy, best.layer, best.kind,
            best.top1, n_corpus, best.top5, n_corpus,
        )

        return CalibrationResult(
            model_id=_model_id_for(model),
            tardigrade_db_version=_current_library_version(),
            timestamp_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            n_layers=n_layers,
            hidden_size=hidden_size,
            best_layer=best.layer,
            best_strategy=best.strategy,
            scores=tuple(scores),
        )

    @staticmethod
    def _score_strategy(
        strategy: Any,
        fact_hidden: list[list[np.ndarray]],
        fact_caches: list[Any],
        query_hidden: list[list[np.ndarray]],
        query_caches: list[Any],
        hidden_size: int,
    ) -> tuple[int, int]:
        """Engine round-trip with a single :class:`RetrievalKeyStrategy`.

        Stores all facts in a fresh engine using ``strategy.compute()``
        for the retrieval key; queries all queries via the same
        strategy; counts top-1 / top-5 hits.
        """
        import tardigrade_db  # local: see module-level comment

        n = len(fact_hidden)
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = tardigrade_db.Engine(tmpdir)
            pack_ids: dict[int, int] = {}
            for i in range(n):
                ret_key = strategy.compute(
                    fact_hidden[i], fact_caches[i], hidden_size
                )
                pid = engine.mem_write_pack(
                    _CALIBRATION_OWNER, ret_key, [], CALIBRATION_SALIENCE,
                    text=f"calibration-fact-{i}",
                )
                pack_ids[i] = pid

            top1 = 0
            top5 = 0
            for i in range(n):
                qkey = strategy.compute(
                    query_hidden[i], query_caches[i], hidden_size
                )
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
        # Multimodal-aware cfg resolution. ``_text_config`` is a no-op
        # for text-only models.
        from ._hidden_states import _text_config
        cfg = _text_config(model.config)
        n_layers = int(cfg.num_hidden_layers)
        return CalibrationResult(
            model_id=model_id,
            tardigrade_db_version=_current_library_version(),
            timestamp_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            n_layers=n_layers,
            hidden_size=int(cfg.hidden_size),
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
    try:
        import tardigrade_db  # local: see module-level comment
    except ImportError:
        return "unknown"
    return getattr(tardigrade_db, "__version__", "unknown")
