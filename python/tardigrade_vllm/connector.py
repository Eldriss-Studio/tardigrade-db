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

    class _TardigradeConnectorMetadata(KVConnectorMetadata):
        """Per-step metadata passed scheduler → worker.

        Currently empty: in single-process mode the worker reads load state
        directly from the connector instance's _load_packs / _load_meta dicts.
        Reserved for future distributed serving where scheduler and worker
        live in separate processes.
        """
        pass

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

            self.hidden_size = model_config.hf_config.hidden_size

            # Load embedding table for lightweight retrieval key computation.
            # The scheduler doesn't have hidden states — only token IDs.
            # We use the embedding table to convert tokens → vectors cheaply.
            self._embed_weights = None  # lazy-loaded from model weights

            # Per-request state
            self._save_buffers = {}  # request_id → {layer_idx: (k_blocks, v_blocks)}
            self._load_packs = {}  # request_id → {pack_data, seq_len}
            self._load_meta = {}  # request_id → {block_ids, num_tokens}

            # Minimum retrieval score to consider a match
            self._match_threshold = float(config.get("match_threshold", 150.0))

        def _get_embed_weights(self):
            """Lazy-load the model's token embedding weights for retrieval keys."""
            if self._embed_weights is not None:
                return self._embed_weights

            try:
                import torch
                # Try to load just the embedding weights from the model
                from transformers import AutoModel
                model_name = self.vllm_config.model_config.model
                model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
                self._embed_weights = model.get_input_embeddings().weight.detach().cpu().numpy()
                del model
                logger.info(f"Loaded embedding table: {self._embed_weights.shape}")
            except Exception as e:
                logger.warning(f"Could not load embedding weights: {e}")
                self._embed_weights = np.array([])  # empty = disable matching
            return self._embed_weights

        def _compute_retrieval_key(self, token_ids):
            """Compute a retrieval key from token IDs using the embedding table.

            Mean-pools token embeddings into a single vector. Lightweight
            alternative to full hidden-state computation — runs on CPU,
            no GPU needed, no model forward pass.
            """
            embed = self._get_embed_weights()
            if embed.size == 0 or len(token_ids) == 0:
                return None

            # Look up embeddings for each token
            valid_ids = [tid for tid in token_ids if 0 <= tid < embed.shape[0]]
            if not valid_ids:
                return None

            token_embeds = embed[valid_ids]  # (n_tokens, hidden_size)
            # Mean-pool to single vector
            return token_embeds.mean(axis=0).astype(np.float32)

        # -- Scheduler-side methods ------------------------------------------------

        def get_num_new_matched_tokens(
            self,
            request,
            num_computed_tokens: int,
        ) -> tuple[Optional[int], bool]:
            """Check if TardigradeDB has stored KV matching this request.

            Computes a lightweight retrieval key from the request's token IDs
            using the model's embedding table (no GPU, no forward pass). Queries
            the engine for matching packs. If a match is found above threshold,
            reports the number of loadable tokens.
            """
            if self.engine.pack_count() == 0:
                return 0, False

            # Get request ID for per-request state tracking
            req_id = getattr(request, "request_id", id(request))

            # Already matched this request
            if req_id in self._load_packs:
                return self._load_packs[req_id]["seq_len"], True

            prompt_ids = getattr(request, "prompt_token_ids", None)
            if prompt_ids is None or len(prompt_ids) == 0:
                return 0, False

            # Compute retrieval key from token embeddings
            query_key = self._compute_retrieval_key(prompt_ids)
            if query_key is None:
                return 0, False

            # Query engine with trace boost
            try:
                packs = self.engine.mem_read_pack_with_trace_boost(
                    query_key, 1, self.owner, 0.3
                )
            except Exception as e:
                logger.debug(f"Retrieval failed: {e}")
                return 0, False

            if not packs or packs[0]["score"] < self._match_threshold:
                return 0, False

            pack = packs[0]
            # Determine seq_len from first layer's data
            if pack["layers"]:
                layer_data = pack["layers"][0]["data"]
                half = len(layer_data) // 2
                seq_len = half // self.kv_dim
            else:
                return 0, False

            # Stash for load phase
            self._load_packs[req_id] = {
                "pack": pack,
                "seq_len": seq_len,
            }

            logger.debug(
                f"Matched pack {pack['pack_id']} (score={pack['score']:.1f}, "
                f"seq_len={seq_len}) for request {req_id}"
            )
            return seq_len, True  # async load supported

        def update_state_after_alloc(
            self,
            request,
            blocks,
            num_external_tokens: int,
        ) -> None:
            """Record allocated block IDs for the load phase."""
            req_id = getattr(request, "request_id", id(request))
            if req_id in self._load_packs and num_external_tokens > 0:
                self._load_meta[req_id] = {
                    "block_ids": blocks,
                    "num_tokens": num_external_tokens,
                }

        def build_connector_meta(self, scheduler_output) -> "KVConnectorMetadata":
            """Build metadata for worker-side load.

            vLLM 0.19+ requires a non-None KVConnectorMetadata instance per
            scheduler step (asserted in gpu_model_runner). Even when there is
            nothing scheduler-side to communicate, return an empty instance.

            In a full distributed implementation, this would package the
            matched pack IDs and block mappings for the worker to load during
            forward. For single-process testing the worker reads directly from
            self._load_packs / self._load_meta.
            """
            return _TardigradeConnectorMetadata()

        # -- Worker-side methods ---------------------------------------------------

        def start_load_kv(self, forward_context, **kwargs) -> None:
            """Load stored KV cache from TardigradeDB into vLLM's paged buffer.

            Adapter step: converts TardigradeDB flat [K|V] arrays to vLLM's
            paged block format, then copies into pre-allocated GPU block slots.

            For each matched request:
              1. Retrieve the stashed pack data (from scheduler-side matching)
              2. Per layer: flat_to_blocks → torch tensor → GPU copy into slots
              3. Clean up per-request state

            Synchronous per-block copy for correctness. Batch GPU transfer
            (scatter via advanced indexing) is a future optimization.
            """
            import torch

            for req_id, load_info in list(self._load_packs.items()):
                meta = self._load_meta.get(req_id)
                if meta is None:
                    continue

                pack = load_info["pack"]
                seq_len = load_info["seq_len"]
                block_ids = meta["block_ids"]

                for layer_entry in pack["layers"]:
                    layer_idx = layer_entry["layer_idx"]
                    layer_data = np.array(layer_entry["data"], dtype=np.float32)

                    k_blocks, v_blocks = flat_to_blocks(
                        layer_data, self.num_kv_heads, self.head_dim,
                        self.block_size,
                    )

                    # forward_context.kv_caches[layer] = (k_cache, v_cache)
                    # each shaped (num_total_blocks, block_size, num_kv_heads, head_dim)
                    kv_cache = forward_context.kv_caches[layer_idx]
                    k_cache, v_cache = kv_cache[0], kv_cache[1]

                    k_tensor = torch.from_numpy(k_blocks)
                    v_tensor = torch.from_numpy(v_blocks)

                    # Copy each block into its allocated GPU slot
                    num_blocks = k_tensor.shape[0]
                    for i in range(min(num_blocks, len(block_ids))):
                        k_cache[block_ids[i]].copy_(k_tensor[i])
                        v_cache[block_ids[i]].copy_(v_tensor[i])

                logger.debug(
                    f"Loaded pack {pack.get('pack_id', '?')} "
                    f"({len(pack.get('layers', []))} layers, "
                    f"{len(block_ids)} blocks) for request {req_id}"
                )

                # Clean up this request's load state
                del self._load_packs[req_id]
                self._load_meta.pop(req_id, None)

        def wait_for_layer_load(self, layer_name: str) -> None:
            """Block until layer KV is loaded. Synchronous for now."""
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
            # vLLM layer_name examples across versions:
            #   "layers.0.self_attn"               (vLLM 0.9.x)
            #   "model.layers.0.self_attn.attn"    (vLLM 0.19+)
            # Find the integer immediately following the "layers" segment.
            layer_idx = None
            parts = layer_name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass
            if layer_idx is None:
                logger.debug(
                    f"save_kv_layer: could not parse layer index from {layer_name!r}"
                )
                return

            if kv_layer is None:
                return

            # Single-batch buffer (vLLM connector is per-step, not per-request)
            buf = self._save_buffers.setdefault("current", {})
            buf[layer_idx] = kv_layer

        def wait_for_save(self) -> None:
            """Flush accumulated KV layers to TardigradeDB as a pack.

            Called after the forward pass completes. Writes all accumulated
            layers as a single atomic pack.
            """
            buf = self._save_buffers.pop("current", None)
            if not buf or len(buf) < self.num_layers:
                logger.debug(
                    f"wait_for_save: skipping write — buf_size="
                    f"{len(buf) if buf else 0}, expected={self.num_layers}"
                )
                return

            # In vLLM 0.19, kv_layer is the FULL GPU KV cache for that layer:
            #   torch.Tensor of shape [2, num_blocks, block_size, num_kv_heads, head_dim]
            #   where dim 0 is K (idx 0) vs V (idx 1).
            #
            # Saving the full cache would be ~12 GiB across 28 layers — we only
            # want this request's blocks. Without per-request block_ids in
            # attn_metadata yet, save block 0 as a proof-of-round-trip.
            #
            # TODO: thread per-request slot_mapping/block_ids from attn_metadata
            # so we save exactly the tokens this request produced.
            BLOCK_SLICE = 1  # number of leading blocks to capture per layer

            layer_payloads = []
            for layer_idx in sorted(buf.keys()):
                kv = buf[layer_idx]

                if not hasattr(kv, "shape"):
                    # Older vLLM format: tuple (k_cache, v_cache) — backward compat
                    if hasattr(kv, "__len__") and len(kv) == 2:
                        k_cache, v_cache = kv
                        k_np = (k_cache.detach().cpu().numpy().astype(np.float32)
                                if hasattr(k_cache, "detach")
                                else np.asarray(k_cache, dtype=np.float32))
                        v_np = (v_cache.detach().cpu().numpy().astype(np.float32)
                                if hasattr(v_cache, "detach")
                                else np.asarray(v_cache, dtype=np.float32))
                    else:
                        continue
                else:
                    # vLLM 0.19+ format: single Tensor [2, blocks, bs, h, d]
                    if kv.dim() != 5 or kv.shape[0] != 2:
                        logger.debug(
                            f"wait_for_save: unexpected kv_layer shape "
                            f"for layer {layer_idx}: {tuple(kv.shape)}"
                        )
                        continue
                    # Cast to float32 in torch first — numpy can't handle bfloat16
                    import torch as _torch
                    sliced = (
                        kv[:, :BLOCK_SLICE]
                        .detach()
                        .to(_torch.float32)
                        .cpu()
                        .numpy()
                    )
                    k_np = sliced[0]  # (BLOCK_SLICE, bs, h, d)
                    v_np = sliced[1]

                # Flatten to TardigradeDB format: [K_flat | V_flat]
                flat = np.concatenate([k_np.ravel(), v_np.ravel()])
                layer_payloads.append((layer_idx, flat))

            if not layer_payloads:
                logger.debug("wait_for_save: no layer payloads built")
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
                logger.debug(
                    f"wait_for_save: wrote pack {pack_id} "
                    f"({len(layer_payloads)} layers)"
                )
            except Exception as e:
                logger.warning(f"wait_for_save: write failed: {e!r}")

        # -- Optional lifecycle methods --------------------------------------------

        def request_finished(
            self,
            request,
            block_ids,
        ) -> tuple[bool, Optional[dict]]:
            """Clean up per-request state.

            vLLM 0.19+ requires this to return (delay_free_blocks, kv_xfer_params):
              - delay_free_blocks=True if the connector is asynchronously
                saving/sending and blocks must NOT be freed yet.
              - kv_xfer_params: optional dict surfaced in request outputs.

            We save synchronously in wait_for_save() during the same forward
            pass, so blocks can be freed immediately. Defensive cleanup of
            any leftover per-request state.
            """
            req_id = getattr(request, "request_id", id(request))
            self._load_packs.pop(req_id, None)
            self._load_meta.pop(req_id, None)
            self._save_buffers.pop(req_id, None)
            return False, None

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
