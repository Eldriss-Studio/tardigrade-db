"""TardigradeDB inference hooks — Python-side ABC and reference implementations."""

from .hook import MemoryCellHandle, TardigradeHook, WriteDecision

__all__ = [
    "MemoryCellHandle",
    "TardigradeClient",
    "TardigradeHook",
    "WriteDecision",
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
