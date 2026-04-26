"""Per-request slot/block extraction from vLLM's attention metadata.

Strategy pattern: ``RequestSlotResolver`` encapsulates vLLM-version-specific
knowledge of how ``slot_mapping`` interleaves multiple in-flight requests
within a single forward step. The connector consumes the resulting
``BatchSlice`` parameter objects without needing to understand the layout.

vLLM 0.19 attention-metadata layout (validated against FlashAttentionMetadata):

  slot_mapping     : Tensor[num_actual_tokens]   absolute slot indices
  query_start_loc  : Tensor[num_reqs + 1]        cumulative token counts
  block_table      : Tensor[num_reqs, max_blocks_per_req]  (not used here)
  seq_lens         : Tensor[num_reqs]                       (not used here)

The resolver emits one ``BatchSlice`` per in-flight request in the step,
identified by its ``batch_index``. Stable cross-step request IDs come from
``update_state_after_alloc`` correlation, which is the connector's job.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchSlice:
    """One in-flight request's contribution to a single forward step.

    Parameter Object: bundles everything the connector needs to slice the
    layer's KV tensor for this specific request, without leaking the
    underlying ``attn_metadata`` shape.
    """
    batch_index: int
    """Position within the current step's batch (0..num_reqs-1).

    NOT a stable request_id — the same logical request gets a different
    ``batch_index`` on each forward step.
    """

    block_indices: tuple[int, ...]
    """Sorted, deduplicated block IDs touched by this request this step."""

    slot_count: int
    """Number of valid (non-padded) slots written this step."""

    first_slot: int
    """First absolute slot index (useful for padding-vs-data detection)."""


class RequestSlotResolver:
    """Strategy: extract per-request slices from vLLM ``attn_metadata``.

    Pure function, no state. Safe to share across threads/requests.
    """

    def resolve(self, attn_metadata, block_size: int) -> list[BatchSlice]:
        """Return one ``BatchSlice`` per request active in this step.

        Args:
            attn_metadata: vLLM attention metadata (e.g., FlashAttentionMetadata).
              Must expose ``slot_mapping`` (1D tensor) and ``query_start_loc``
              (1D tensor of length num_reqs + 1).
            block_size: vLLM block size (typically 16). Used to convert
              absolute slot indices to block indices.

        Returns:
            One ``BatchSlice`` per active request, ordered by ``batch_index``.
            Empty list if ``attn_metadata`` is missing required fields.
        """
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        if slot_mapping is None or query_start_loc is None:
            return []

        # Convert to plain Python — these tensors are small (per-step batch metadata)
        starts = query_start_loc.tolist() if hasattr(query_start_loc, "tolist") else list(query_start_loc)
        slots = slot_mapping.tolist() if hasattr(slot_mapping, "tolist") else list(slot_mapping)

        slices: list[BatchSlice] = []
        for i in range(len(starts) - 1):
            start, end = starts[i], starts[i + 1]
            if end <= start:
                continue
            req_slots = slots[start:end]
            req_blocks = sorted({slot // block_size for slot in req_slots})
            slices.append(BatchSlice(
                batch_index=i,
                block_indices=tuple(req_blocks),
                slot_count=end - start,
                first_slot=req_slots[0],
            ))
        return slices
