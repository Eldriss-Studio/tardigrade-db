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
from tardigrade_vllm.slot_resolver import BatchSlice, RequestSlotResolver

logger = logging.getLogger("tardigrade_vllm")


def _parse_layer_index(layer_name: str):
    """Extract the integer following the ``layers`` segment in a layer name.

    Handles both ``"layers.0.self_attn"`` (vLLM 0.9.x) and
    ``"model.layers.0.self_attn.attn"`` (vLLM 0.19+). Returns ``None`` if
    no integer is found in the expected position.
    """
    parts = layer_name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return None


if HAS_VLLM:

    def _looks_like_kv_cache_config(obj) -> bool:
        """Heuristic: distinguish KVCacheConfig from KVConnectorRole at runtime.

        vLLM's KVConnectorRole is a small enum; KVCacheConfig is a heavier
        object with attributes like ``num_blocks`` or ``block_size``. Used to
        disambiguate the second positional arg between old- and new-style
        connector __init__ calls.
        """
        if obj is None:
            return False
        # Enums (Role) won't have these attributes
        return any(hasattr(obj, attr) for attr in ("num_blocks", "block_size", "kv_cache_groups"))

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
            kv_cache_config=None,
            role: "KVConnectorRole" = None,
        ):
            # vLLM 0.19+ calls __init__ with three positional args:
            #   (vllm_config, kv_cache_config, role)
            # vLLM 0.9.x called it with two:
            #   (vllm_config, role)
            # Detect old-style call where the second arg is actually the role.
            if role is None and kv_cache_config is not None and not _looks_like_kv_cache_config(kv_cache_config):
                role = kv_cache_config
                kv_cache_config = None

            # Forward to base; older bases that take only (config, role) still work
            # because kv_cache_config defaults to None on this side too.
            try:
                super().__init__(vllm_config, kv_cache_config, role)
            except TypeError:
                # Older vLLM base that only accepts (vllm_config, role)
                super().__init__(vllm_config, role)

            self.kv_cache_config = kv_cache_config
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

            # Per-request state. save_buffers is keyed by step-local
            # batch_index. Across forward steps, save_kv_layer overwrites the
            # same (batch_index, layer_idx) entry so we always hold the
            # latest snapshot.
            self._save_buffers: dict = {}  # batch_index → {layer_idx: (kv_layer, BatchSlice)}

            # Per-request pack tracking for coalescing (Step 2). Maps a stable
            # request fingerprint (block_indices[0]) to the pack_id we last
            # wrote for it. On each step we delete the old pack before writing
            # the new one — net effect: one pack per request, contents = the
            # latest snapshot. Worker- and scheduler-side connectors are
            # separate instances, so we can't use request_finished here.
            self._pack_id_by_fingerprint: dict[int, int] = {}

            self._load_packs = {}  # request_id → {pack_data, seq_len}
            self._load_meta = {}  # request_id → {block_ids, num_tokens}

            # Strategy for extracting per-request slices from attn_metadata
            self._slot_resolver = RequestSlotResolver()

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
            """Accumulate one layer's KV for each in-flight request this step.

            Called per layer per forward pass. We use the slot resolver to
            split the layer's KV tensor into per-request slices via
            ``slot_mapping`` / ``query_start_loc``. Each request's data is
            buffered separately, keyed by step-local ``batch_index``.

            Step 2 will key by stable ``request_id`` once
            ``update_state_after_alloc`` correlation is wired up.
            """
            layer_idx = _parse_layer_index(layer_name)
            if layer_idx is None:
                logger.debug(f"save_kv_layer: could not parse layer index from {layer_name!r}")
                return
            if kv_layer is None:
                return

            slices = self._slot_resolver.resolve(attn_metadata, self.block_size)
            if not slices:
                # No per-request slot info — fall back to single-buffer behaviour
                # so we don't silently drop the step on unknown attn_metadata shapes.
                slices = [BatchSlice(batch_index=0, block_indices=(), slot_count=0, first_slot=0)]

            for sl in slices:
                buf = self._save_buffers.setdefault(sl.batch_index, {})
                buf[layer_idx] = (kv_layer, sl)

        def wait_for_save(self) -> None:
            """Commit one pack per in-flight request, deduplicated across steps.

            Coalescing strategy (Step 2): commit on every forward pass, but
            delete the previous pack for the same request before writing.
            End state: one pack per request, contents = the latest snapshot.

            Required because vLLM instantiates separate scheduler-side and
            worker-side connectors — request_finished fires on the scheduler
            but our buffer lives on the worker, so we can't defer commit to
            the lifecycle hook.

            Request fingerprint = first block index of the BatchSlice. Stable
            for the lifetime of the request because block 0 (or whichever the
            scheduler allocated first) does not get reassigned mid-request.
            """
            if not self._save_buffers:
                return

            for batch_idx in sorted(self._save_buffers.keys()):
                layer_buf = self._save_buffers[batch_idx]
                if len(layer_buf) < self.num_layers:
                    continue

                # Fingerprint from any layer's BatchSlice (block_indices is
                # the same across layers within a step for a given request).
                _, sample_slice = next(iter(layer_buf.values()))
                fingerprint = (
                    sample_slice.block_indices[0]
                    if sample_slice.block_indices
                    else (-1 - batch_idx)  # synthetic for unknown-block fallback
                )

                # Drop the previous pack for this request, if any.
                prior = self._pack_id_by_fingerprint.get(fingerprint)
                if prior is not None:
                    try:
                        self.engine.delete_pack(prior)
                    except Exception as e:
                        logger.debug(f"wait_for_save: stale delete failed: {e!r}")

                pack_id = self._write_pack_for_batch(layer_buf)
                if pack_id is not None:
                    self._pack_id_by_fingerprint[fingerprint] = pack_id

            self._save_buffers.clear()

        def _write_pack_for_batch(self, layer_buf: dict):
            """Build and write a pack from one request's accumulated layers.

            ``layer_buf`` is ``{layer_idx: (kv_layer_tensor, BatchSlice)}``.
            Slices each layer to ``BatchSlice.block_indices`` and trims to
            ``slot_count`` valid tokens before flattening.

            Returns the new pack_id on success, None on failure / no payload.
            """
            import torch as _torch

            layer_payloads: list[tuple[int, np.ndarray]] = []
            for layer_idx in sorted(layer_buf.keys()):
                kv, sl = layer_buf[layer_idx]
                k_np, v_np = self._extract_kv_slice(kv, sl, _torch)
                if k_np is None:
                    continue
                # Flatten to TardigradeDB format: [K_flat | V_flat]
                flat = np.concatenate([k_np.ravel(), v_np.ravel()])
                layer_payloads.append((layer_idx, flat))

            if not layer_payloads:
                logger.debug("wait_for_save: no layer payloads built")
                return None

            # Retrieval key: mean of layer-0 K projection over the request's tokens.
            # Step 6 will replace this with the engine's empirically-validated
            # hidden-states / per-token K strategy.
            first_layer_data = layer_payloads[0][1]
            half = len(first_layer_data) // 2
            k_first_layer = first_layer_data[:half].reshape(-1, self.kv_dim)
            retrieval_key = k_first_layer.mean(axis=0).astype(np.float32)

            try:
                pack_id = self.engine.mem_write_pack(
                    self.owner, retrieval_key, layer_payloads, 80.0
                )
                logger.debug(
                    f"wait_for_save: wrote pack {pack_id} ({len(layer_payloads)} layers)"
                )
                return pack_id
            except Exception as e:
                logger.warning(f"wait_for_save: write failed: {e!r}")
                return None

        def _extract_kv_slice(self, kv, sl: BatchSlice, _torch):
            """Return (k_np, v_np) for one request's blocks within one layer.

            Handles both vLLM tensor formats:
              - vLLM 0.19+: single Tensor [2, num_blocks, bs, kv_heads, head_dim]
              - older: (k_cache, v_cache) tuple
            And bfloat16 → float32 cast since numpy can't represent bf16.

            Returns (None, None) if the layer's data isn't usable.
            """
            # Backward-compat: tuple format (vLLM <= 0.9)
            if not hasattr(kv, "shape"):
                if not (hasattr(kv, "__len__") and len(kv) == 2):
                    return None, None
                k_cache, v_cache = kv
                k_np = self._tensor_to_float32_numpy(k_cache, _torch)
                v_np = self._tensor_to_float32_numpy(v_cache, _torch)
                return k_np, v_np

            # vLLM 0.19+ format: single Tensor [2, blocks, bs, kv_heads, head_dim]
            if kv.dim() != 5 or kv.shape[0] != 2:
                logger.debug(f"_extract_kv_slice: unexpected kv shape {tuple(kv.shape)}")
                return None, None

            if not sl.block_indices:
                # No resolver info — fall back to first block as a soft default
                indices = [0]
            else:
                indices = list(sl.block_indices)

            sliced = (
                kv[:, indices]              # [2, n_blocks, bs, h, d]
                .detach()
                .to(_torch.float32)
                .cpu()
                .numpy()
            )
            # Reshape to (tokens, kv_dim) and trim to the actual valid token count
            n_blocks = sliced.shape[1]
            slots_per_block = sliced.shape[2]
            total_slots = n_blocks * slots_per_block
            valid = sl.slot_count if sl.slot_count > 0 else total_slots
            valid = min(valid, total_slots)

            k_np = sliced[0].reshape(total_slots, self.kv_dim)[:valid]
            v_np = sliced[1].reshape(total_slots, self.kv_dim)[:valid]
            return k_np, v_np

        @staticmethod
        def _tensor_to_float32_numpy(t, _torch):
            if hasattr(t, "detach"):
                return t.detach().to(_torch.float32).cpu().numpy()
            return np.asarray(t, dtype=np.float32)

        # -- Optional lifecycle methods --------------------------------------------

        def request_finished(
            self,
            request,
            block_ids,
        ) -> tuple[bool, Optional[dict]]:
            """Scheduler-side lifecycle hook. Cleanup only.

            vLLM 0.19+ contract: returns (delay_free_blocks, kv_xfer_params).
            We save synchronously inside the worker's wait_for_save, so blocks
            can be freed immediately (False).

            Cannot commit packs here: scheduler and worker are separate
            connector instances and the save buffer lives on the worker.
            Pack coalescing happens via dedup-by-fingerprint in wait_for_save.
            """
            req_id = getattr(request, "request_id", id(request))
            self._load_packs.pop(req_id, None)
            self._load_meta.pop(req_id, None)
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
