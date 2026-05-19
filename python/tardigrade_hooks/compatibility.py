"""Pre-flight model compatibility checker for tardigrade-db.

Consumers wiring a HuggingFace model into :class:`KnowledgePackStore` can
ask :func:`is_supported` whether the ``(model, tokenizer)`` pair will
work *before* deploy, instead of discovering an incompatibility at
runtime via ``AttributeError`` deep inside ``store()`` or as a silent
retrieval failure.

Design pattern: **Inspector** (free function, pure-config, no I/O).
Returns a **Value Object** — :class:`CompatibilityReport` — so consumers
can pattern-match on the verdict in their own bootstrap code.

The probe is config-only by design: no forward pass, no device hop, no
tokenizer mutation. That keeps it cheap enough to call before the model
is even moved to a GPU.

Example::

    from tardigrade_hooks import is_supported

    report = is_supported(model, tokenizer)
    if not report.is_supported:
        raise RuntimeError(f"unsupported: {report.blockers}")
    for note in report.notes:
        logger.warning(note)
    if report.recommended_strategy == "calibration_required":
        select_query_layer(model, tokenizer, registry=reg)
"""

from __future__ import annotations

from dataclasses import dataclass

from .chat_template_adapter import select_chat_template_adapter
from ._hidden_states import softmax_layer_count


@dataclass(frozen=True)
class CompatibilityReport:
    """Typed verdict from :func:`is_supported`.

    Frozen + hashable so consumers can stash reports in sets or use them
    as dict keys when caching pre-flight results across boots.

    Attributes:
        is_supported: True if the model can be used with
            :class:`KnowledgePackStore` without blockers.
        architecture: ``"uniform_softmax"`` (Qwen3, Llama-3, GPT-2, …),
            ``"hybrid"`` (RecurrentGemma, Jamba, Qwen3-Next, …), or
            ``"unknown"`` when the config doesn't expose
            ``num_hidden_layers``.
        n_hidden_layers: ``model.config.num_hidden_layers``, or 0 if
            missing.
        n_softmax_layers: count of attention-typed layers — equals
            ``n_hidden_layers`` for uniform-softmax, strictly less for
            hybrid models.
        recommended_strategy: one of ``"hidden_state"``,
            ``"k_vector"``, ``"calibration_required"``, or empty string
            when unsupported. ``"calibration_required"`` signals to the
            consumer that they should call ``select_query_layer()``
            before relying on retrieval.
        adapter_type: class name of the chat-template adapter
            :func:`select_chat_template_adapter` would pick for this
            tokenizer (empty string when unsupported).
        notes: soft warnings — things the consumer should know but
            that don't block storage.
        blockers: hard reasons why ``is_supported`` is False. Empty
            when the model is supported.
    """

    is_supported: bool
    architecture: str
    n_hidden_layers: int
    n_softmax_layers: int
    recommended_strategy: str
    adapter_type: str
    notes: tuple[str, ...]
    blockers: tuple[str, ...]


def _classify_architecture(n_hidden: int, n_softmax: int) -> str:
    if n_hidden == 0:
        return "unknown"
    if n_softmax < n_hidden:
        return "hybrid"
    return "uniform_softmax"


def is_supported(model, tokenizer) -> CompatibilityReport:
    """Pre-flight a ``(model, tokenizer)`` pair against
    :class:`KnowledgePackStore`'s requirements.

    The probe reads:
    - ``model.config.num_hidden_layers``
    - ``model.config.layers_block_type`` / ``layer_types`` (when present)
    - ``tokenizer.apply_chat_template`` (callable presence)
    - ``tokenizer.chat_template`` (None vs set)

    No forward pass. No tokenizer mutation. No device requirement.

    Returns a :class:`CompatibilityReport` populated with verdict,
    derived counts, recommended strategy, and human-readable notes /
    blockers. The verdict reflects whether **storage + retrieval** will
    work; whether the model produces good *generations* with the
    retrieved cache is a separate concern (model size, training).
    """
    notes: list[str] = []
    blockers: list[str] = []

    cfg = getattr(model, "config", None)
    n_hidden_layers: int = (
        getattr(cfg, "num_hidden_layers", 0) if cfg is not None else 0
    )

    if n_hidden_layers == 0:
        blockers.append(
            "model.config.num_hidden_layers is missing or zero — "
            "tardigrade-db cannot size the layer-payload loop without it"
        )
        return CompatibilityReport(
            is_supported=False,
            architecture="unknown",
            n_hidden_layers=0,
            n_softmax_layers=0,
            recommended_strategy="",
            adapter_type="",
            notes=tuple(notes),
            blockers=tuple(blockers),
        )

    n_softmax_layers = softmax_layer_count(cfg)
    architecture = _classify_architecture(n_hidden_layers, n_softmax_layers)

    if n_softmax_layers == 0:
        blockers.append(
            "model has no softmax attention layers — no observable "
            "retrieval signal (pure-recurrent / SSM / Mamba / RWKV "
            "architectures are not currently supported)"
        )

    # Tokenizer must expose a callable apply_chat_template — the
    # ChatTemplateAdapter factory probes it during ``store()``. A bare
    # object missing this method makes the whole storage path unusable.
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        blockers.append(
            "tokenizer is missing a callable apply_chat_template "
            "method — KnowledgePackStore requires HuggingFace-style "
            "chat-template tokenizers"
        )
    elif getattr(tokenizer, "chat_template", None) is None:
        notes.append(
            "tokenizer.chat_template is None — facts will be stored "
            "without a chat-template wrap; recall may degrade vs a "
            "tokenizer that ships a real chat_template"
        )

    if blockers:
        return CompatibilityReport(
            is_supported=False,
            architecture=architecture,
            n_hidden_layers=n_hidden_layers,
            n_softmax_layers=n_softmax_layers,
            recommended_strategy="",
            adapter_type="",
            notes=tuple(notes),
            blockers=tuple(blockers),
        )

    # Chat-template adapter selection delegates to the same Factory the
    # KnowledgePackStore constructor uses — keeps the pre-flight verdict
    # consistent with what storage will actually pick at runtime.
    adapter_type = type(select_chat_template_adapter(tokenizer)).__name__

    if architecture == "hybrid":
        recommended_strategy = "calibration_required"
        notes.append(
            "hybrid attention model — run calibration via "
            "select_query_layer(model, tokenizer, registry=reg) "
            "to discover the best (strategy, layer) pair; "
            "K-vector strategy is the likely winner"
        )
    else:
        recommended_strategy = "hidden_state"

    return CompatibilityReport(
        is_supported=True,
        architecture=architecture,
        n_hidden_layers=n_hidden_layers,
        n_softmax_layers=n_softmax_layers,
        recommended_strategy=recommended_strategy,
        adapter_type=adapter_type,
        notes=tuple(notes),
        blockers=tuple(blockers),
    )
