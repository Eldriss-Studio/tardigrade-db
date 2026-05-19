"""TardigradeDB inference hooks — Python-side ABC and reference implementations."""

from .calibrate import (
    CalibrationResult,
    CalibrationStrategy,
    LayerScore,
    LinearSweepStrategy,
    select_query_layer,
)
from .calibration_registry import CalibrationRegistry
from .chat_template_adapter import (
    ChatTemplateAdapter,
    LegacySystemAdapter,
    UserMessageAdapter,
    select_chat_template_adapter,
)
from .hook import MemoryCellHandle, TardigradeHook, WriteDecision
from .retrieval_key_strategy import (
    HiddenStateKeyStrategy,
    KVectorKeyStrategy,
    RetrievalKeyStrategy,
)

__all__ = [
    "CalibrationRegistry",
    "CalibrationResult",
    "CalibrationStrategy",
    "ChatTemplateAdapter",
    "HiddenStateKeyStrategy",
    "KVectorKeyStrategy",
    "LayerScore",
    "LegacySystemAdapter",
    "LinearSweepStrategy",
    "MemoryCellHandle",
    "RetrievalKeyStrategy",
    "TardigradeClient",
    "TardigradeHook",
    "UserMessageAdapter",
    "WriteDecision",
    "select_chat_template_adapter",
    "select_query_layer",
]


def __getattr__(name: str):
    """Lazy re-export of :class:`TardigradeClient`.

    Eager-importing ``client`` here pulls in ``tardigrade_db._native``
    (the compiled extension), which CI lint jobs intentionally don't
    build. PEP 562 ``__getattr__`` lets ``from tardigrade_hooks
    import TardigradeClient`` keep working at runtime while leaving
    sibling imports like ``from tardigrade_hooks.constants import X``
    free of the native dependency.
    """
    if name == "TardigradeClient":
        from .client import TardigradeClient

        return TardigradeClient
    raise AttributeError(f"module 'tardigrade_hooks' has no attribute {name!r}")
