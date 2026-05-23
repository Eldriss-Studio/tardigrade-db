"""Anti-Corruption Layer (Evans, DDD) for vLLM's internal contracts.

vLLM's internal APIs change between minor releases — the v0.18→v0.19
transition alone moved ``KVCacheBlocks`` from a bare list to a
dataclass, removed ``ForwardContext.kv_caches`` in favour of a
``register_kv_caches`` lifecycle hook, and reshaped the ``Request``
object's attribute surface. Without a bridge, every one of those
changes is a scattered patch across the connector.

This module is the single point of contact for those shifts. The
connector imports only from here; the bridge speaks vLLM's dialect on
one side and TardigradeDB's domain types on the other. The next vLLM
bump touches this file alone.

Bridges currently in place:

- [`KVCacheBlocksAdapter`] — vLLM 0.19+'s ``KVCacheBlocks`` dataclass
  (and legacy list / tuple-of-lists shapes) to ``int`` / ``list[int]``.
- [`RequestAdapter`] — vLLM's ``Request`` duck-type to plain
  ``request_id`` / ``prompt_token_ids`` accessors with sensible
  fallbacks.

Out of scope (lives elsewhere because there's no value in moving it):

- ``RequestSlotResolver`` (in ``slot_resolver.py``) is already a
  Strategy over ``attn_metadata`` shapes; promoting it into the bridge
  would just rename it without changing the contract surface.
- ``register_kv_caches`` is a method vLLM calls *on us* (a contract we
  fulfill), not a method we call into vLLM — it belongs on the
  connector class, not on the bridge.
"""

from typing import Optional


class KVCacheBlocksAdapter:
    """Bridge vLLM's allocation result to plain ``int`` / ``list[int]``.

    Adapter (GoF Structural): vLLM 0.19+ returns a ``KVCacheBlocks``
    dataclass with a ``get_block_ids()`` method that yields
    ``tuple[list[int], ...]`` (outer tuple = per kv-cache group). Older
    vLLM passed the inner ``tuple[list[int], ...]`` directly, or even a
    bare ``list[int]``. The ``hasattr`` duck-type gate accepts all three
    so callers don't need to know which vLLM is installed.

    All methods are static — the adapter is stateless and the call
    sites are read-only translations.
    """

    @staticmethod
    def first_block_id(blocks) -> Optional[int]:
        """Return group 0's first block ID, or ``None`` if the
        allocation is empty (e.g. precomputed empty ``KVCacheBlocks``
        for a request that didn't need an allocation).
        """
        if hasattr(blocks, "get_block_ids"):
            groups = blocks.get_block_ids()
            if not groups or not groups[0]:
                return None
            return int(groups[0][0])
        # Legacy: tuple of lists or bare list of block-id ints.
        first_group = blocks[0] if isinstance(blocks, tuple) else blocks
        if not first_group:
            return None
        return int(first_group[0] if isinstance(first_group, list) else first_group)

    @staticmethod
    def block_ids(blocks) -> list[int]:
        """Return group 0's block IDs as a ``list[int]``, or ``[]`` on
        empty allocation. Used when downstream code needs to iterate
        (e.g. building per-block load requests for the worker side).
        """
        if hasattr(blocks, "get_block_ids"):
            groups = blocks.get_block_ids()
            if not groups or not groups[0]:
                return []
            return [int(b) for b in groups[0]]
        # Legacy: tuple of lists or bare list.
        first_group = blocks[0] if isinstance(blocks, tuple) else blocks
        if not first_group:
            return []
        if isinstance(first_group, list):
            return [int(b) for b in first_group]
        return [int(first_group)]


class RequestAdapter:
    """Bridge vLLM's ``Request`` duck-type to plain accessors.

    vLLM doesn't expose a stable ``Request`` protocol; the connector
    receives whatever object the scheduler builds for the current
    request. Older vLLM lacked ``request_id`` entirely; we fell back to
    ``id(request)`` for a stable per-process key. ``prompt_token_ids``
    can be absent on certain request types (encoder-only, embedding).

    Centralizing these ``getattr`` patterns here means the next time
    vLLM renames an attribute, we change it in one place.
    """

    @staticmethod
    def request_id(request):
        """Return ``request.request_id`` if present, else ``id(request)``.
        The fallback gives a stable in-process key so per-request state
        tracking still works against older vLLM."""
        return getattr(request, "request_id", id(request))

    @staticmethod
    def prompt_token_ids(request) -> Optional[list[int]]:
        """Return ``request.prompt_token_ids`` or ``None`` if absent.
        Callers gate on ``None`` rather than catching ``AttributeError``."""
        return getattr(request, "prompt_token_ids", None)
