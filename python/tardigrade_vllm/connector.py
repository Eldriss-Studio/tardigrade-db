"""TardigradeDB KV Connector for vLLM — persistent memory injection.

Adapter pattern: bridges TardigradeDB's flat KV arrays to vLLM's
paged attention block format via the KV Connector v1 API.

On save: captures KV blocks from vLLM generation → flattens → stores in engine.
On load: queries engine for matching memory → reshapes to blocks → injects.

Requires vLLM >= 0.9.0. Install with: pip install vllm tardigrade-db

Configuration via engine_args:
    --kv-connector tardigrade_vllm.connector.TardigradeConnector
    --kv-connector-config '{"db_path": "/data/agent-memory"}'
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add tardigrade_hooks to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

import tardigrade_db
from tardigrade_vllm.format import blocks_to_flat, flat_to_blocks

logger = logging.getLogger("tardigrade_vllm")


if HAS_VLLM:

    class TardigradeConnector(KVConnectorBase_V1):
        """vLLM KV Connector that persists KV cache in TardigradeDB.

        Scheduler side: queries engine for matching memories.
        Worker side: loads/saves KV blocks to/from engine.

        Config keys (passed via --kv-connector-config JSON):
            db_path: str — engine storage directory
            owner: int — owner ID for memory isolation (default: 1)
        """

        def __init__(
            self,
            vllm_config: "VllmConfig",
            role: "KVConnectorRole",
        ):
            super().__init__(vllm_config, role)

            config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
            db_path = config.get("db_path", os.environ.get("TARDIGRADE_DB_PATH", "./tardigrade-memory"))
            self.owner = config.get("owner", int(os.environ.get("TARDIGRADE_OWNER", "1")))

            self.engine = tardigrade_db.Engine(db_path)
            logger.info(f"TardigradeDB engine opened: {db_path} ({self.engine.pack_count()} packs)")

            model_config = vllm_config.model_config
            self.num_layers = model_config.hf_config.num_hidden_layers
            self.num_kv_heads = getattr(
                model_config.hf_config, "num_key_value_heads",
                model_config.hf_config.num_attention_heads,
            )
            self.head_dim = getattr(
                model_config.hf_config, "head_dim",
                model_config.hf_config.hidden_size // model_config.hf_config.num_attention_heads,
            )
            self.kv_dim = self.num_kv_heads * self.head_dim
            self.block_size = vllm_config.cache_config.block_size

            # Per-request state for save accumulation
            self._save_buffers = {}  # request_id → {layer_idx: (k_blocks, v_blocks)}
            # Per-request state for load
            self._load_packs = {}  # request_id → pack_read_result

        # -- Scheduler-side methods ------------------------------------------------

        def get_num_new_matched_tokens(
            self,
            request,
            num_computed_tokens: int,
        ) -> tuple[Optional[int], bool]:
            """Check if TardigradeDB has stored KV for this request's context.

            For now, returns 0 — semantic matching requires computing hidden
            states which the scheduler doesn't have. Future: use the request's
            token IDs to compute a lightweight retrieval key.

            TODO: Implement semantic matching via request prefix hash or
            lightweight token embedding.
            """
            # Placeholder: no external tokens available yet.
            # Full implementation would compute a retrieval key from
            # request.prompt_token_ids and query self.engine.mem_read_pack.
            return 0, False

        def update_state_after_alloc(
            self,
            request,
            blocks,
            num_external_tokens: int,
        ) -> None:
            """Update state after block allocation. Currently no-op."""
            pass

        def build_connector_meta(self, scheduler_output) -> Optional["KVConnectorMetadata"]:
            """Build metadata for this scheduler step. Currently no-op."""
            return None

        # -- Worker-side methods ---------------------------------------------------

        def start_load_kv(self, forward_context, **kwargs) -> None:
            """Load stored KV cache from TardigradeDB into vLLM's paged buffer.

            Called before the forward pass. Currently no-op until
            get_num_new_matched_tokens implements semantic matching.
            """
            pass

        def wait_for_layer_load(self, layer_name: str) -> None:
            """Block until layer KV is loaded. Currently no-op."""
            pass

        def save_kv_layer(
            self,
            layer_name: str,
            kv_layer,
            attn_metadata,
            **kwargs,
        ) -> None:
            """Save a layer's KV cache from vLLM to TardigradeDB.

            Accumulates layers per request. When all layers for a request
            are collected, writes a complete pack to the engine.

            Note: This is called per-layer during the attention forward pass.
            The actual pack write happens in wait_for_save() when all layers
            are accumulated.
            """
            # Extract layer index from layer_name (e.g., "layers.0.self_attn")
            try:
                layer_idx = int(layer_name.split(".")[1])
            except (IndexError, ValueError):
                return

            # kv_layer is a tuple of (k_cache, v_cache) tensors
            # Each is (num_tokens, num_kv_heads, head_dim) or block-shaped
            if kv_layer is None:
                return

            # Accumulate — keyed by a request identifier from attn_metadata
            # For simplicity, use a single buffer (batch=1 assumption for now)
            buf = self._save_buffers.setdefault("current", {})
            buf[layer_idx] = kv_layer

        def wait_for_save(self) -> None:
            """Flush accumulated KV layers to TardigradeDB as a pack.

            Called after the forward pass completes. Writes all accumulated
            layers as a single atomic pack.
            """
            buf = self._save_buffers.pop("current", None)
            if not buf or len(buf) < self.num_layers:
                return

            # Build pack layers from accumulated blocks
            from tardigrade_hooks.encoding import encode_per_token

            layer_payloads = []
            for layer_idx in sorted(buf.keys()):
                kv = buf[layer_idx]
                if hasattr(kv, '__len__') and len(kv) == 2:
                    k_cache, v_cache = kv
                    # Convert to numpy
                    if hasattr(k_cache, 'detach'):
                        k_np = k_cache.detach().cpu().numpy().astype(np.float32)
                        v_np = v_cache.detach().cpu().numpy().astype(np.float32)
                    else:
                        k_np = np.asarray(k_cache, dtype=np.float32)
                        v_np = np.asarray(v_cache, dtype=np.float32)

                    # Flatten to TardigradeDB format: [K_flat | V_flat]
                    flat = np.concatenate([k_np.ravel(), v_np.ravel()])
                    layer_payloads.append((layer_idx, flat))

            if not layer_payloads:
                return

            # Generate a simple retrieval key (mean of first layer K values)
            first_layer_data = layer_payloads[0][1]
            half = len(first_layer_data) // 2
            k_mean = first_layer_data[:half].reshape(-1, self.kv_dim).mean(axis=0)
            # Use mean-pooled K as a simple retrieval key
            retrieval_key = k_mean

            try:
                pack_id = self.engine.mem_write_pack(
                    self.owner, retrieval_key, layer_payloads, 80.0
                )
                logger.debug(f"Saved KV pack {pack_id} ({len(layer_payloads)} layers)")
            except Exception as e:
                logger.warning(f"Failed to save KV pack: {e}")

        # -- Optional lifecycle methods --------------------------------------------

        def request_finished(
            self,
            request,
            block_ids,
        ) -> None:
            """Clean up per-request state."""
            pass

        def shutdown(self) -> None:
            """Clean shutdown — engine handles its own persistence."""
            logger.info("TardigradeDB connector shutting down")

else:
    # Stub for environments without vLLM
    class TardigradeConnector:
        """Stub — vLLM not installed. Install with: pip install vllm"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TardigradeConnector requires vLLM >= 0.9.0. "
                "Install with: pip install vllm"
            )
