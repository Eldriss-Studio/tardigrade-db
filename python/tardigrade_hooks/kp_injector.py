# Knowledge Packs-style KV cache storage and injection.
#
# Stores the complete past_key_values from a chat-template-formatted
# fact as a single TardigradeDB memory. Recovers it as a DynamicCache
# and injects it directly into model.generate().
#
# Based on "Knowledge Packs: Zero-Token Knowledge Delivery via KV Cache
# Injection" (arXiv 2604.03270). Key requirements:
#   1. Wrap facts in chat template BEFORE computing KV
#   2. Clone DynamicCache before injection
#   3. Let HuggingFace auto-handle position_ids
#   4. Attention mask = ones(kv_len + query_len)

import numpy as np
import torch

from .calibrate import _model_id_for
from .chat_template_adapter import (
    ChatTemplateAdapter,
    select_chat_template_adapter,
)
from .constants import (
    DEFAULT_CAPTURE_LAYER_RATIO,
    EDGE_CONTRADICTS,
    EDGE_FOLLOWS,
    EDGE_SUPPORTS,
)
from .encoding import encode_per_token
from .multi_composer import NaiveConcatComposer
from transformers import DynamicCache

import tardigrade_db


def _move_cache_to_device(cache, device):
    """Return a new ``DynamicCache`` with all layers moved to ``device``.

    Composers in :mod:`tardigrade_hooks.multi_composer` build caches from raw
    numpy data and have no model handle to learn the target device from —
    they return CPU tensors by design. Callers that hold the model pass the
    composed cache through this helper so it can be injected into a GPU
    model. When the cache is already on the target device, the ``.to()``
    calls are no-ops.
    """
    if not cache.layers:
        return cache
    moved = DynamicCache()
    for li in range(len(cache.layers)):
        layer = cache.layers[li]
        moved.update(layer.keys.to(device), layer.values.to(device), li)
    return moved


class KnowledgePackStore:
    """Stores and retrieves complete KV caches through TardigradeDB.

    Each memory is stored as N cells (one per layer), all sharing the
    same cell_id prefix. The retrieval key is a hidden-state summary
    for Top5Avg matching. The value is the full K+V payload per layer.
    """

    def __init__(
        self,
        engine,
        model,
        tokenizer,
        owner=1,
        query_layer=None,
        adapter: ChatTemplateAdapter | None = None,
        calibration_registry=None,
    ):
        self.engine = engine
        self.model = model
        self.tokenizer = tokenizer
        self.owner = owner
        # Factory default: probe the tokenizer's chat template to pick the
        # right adapter. Existing Qwen3-family callers get LegacySystemAdapter
        # (backwards-compat); strict-template callers (Qwen3.5, Llama-3, etc.)
        # get UserMessageAdapter automatically. See chat_template_adapter.py.
        self.adapter = adapter or select_chat_template_adapter(tokenizer)

        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.hidden_size = cfg.hidden_size

        # Layer-selection order:
        #   1. Explicit `query_layer` arg wins (existing escape hatch).
        #   2. Else: consult the calibration registry if provided and a
        #      cached entry exists for this model_id. This is how hybrid
        #      models (Qwen3-Next, RecurrentGemma, Jamba, Zamba, …) get
        #      the right attention layer auto-picked without the caller
        #      having to know the architecture's layer layout.
        #   3. Else: fall back to the static DEFAULT_CAPTURE_LAYER_RATIO
        #      heuristic — works fine on uniform-softmax models, may
        #      land on a recurrent layer for hybrid models (silent
        #      ~0% recall — calibrate to fix).
        if query_layer is not None:
            self.query_layer = query_layer
        elif calibration_registry is not None:
            cached = calibration_registry.load(_model_id_for(model))
            self.query_layer = (
                cached.best_layer if cached is not None
                else int(self.n_layers * DEFAULT_CAPTURE_LAYER_RATIO)
            )
        else:
            self.query_layer = int(self.n_layers * DEFAULT_CAPTURE_LAYER_RATIO)

    def store(self, fact_text, salience=80.0, auto_link=True, auto_link_threshold=None):
        """Store a fact's KV cache across all layers.

        Wraps the fact in the model's chat template before computing KV.
        Stores hidden states as retrieval key (for Top5Avg matching)
        and full K+V payload as injection value (per layer).

        If auto_link is True, searches existing memories before writing
        and creates trace links to similar ones (Zettelkasten pattern).

        Returns the pack_id assigned by the engine.
        """
        # Tensors that touch the model must live on the model's device.
        # `.to(device)` is a no-op when the device already matches, so this
        # keeps the historical CPU path correct while also supporting CUDA.
        device = self.model.device

        # Chat template wrapping — the adapter decides the message shape
        # so this works on any tokenizer's chat template.
        messages = self.adapter.store_messages(fact_text)
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        input_ids = self.tokenizer.encode(formatted, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            out = self.model(input_ids, use_cache=True, output_hidden_states=True)

        kv = out.past_key_values

        # Retrieval key: hidden states at query_layer (per-token, skip pos 0)
        hidden = out.hidden_states[self.query_layer][0]  # (seq, hidden_size)
        h_tokens = hidden[1:].float().cpu().numpy().astype(np.float32)  # skip pos 0
        retrieval_key = encode_per_token(h_tokens, self.hidden_size)

        # Build layer payloads for pack API
        layer_payloads = []
        for li in range(self.n_layers):
            k = kv.layers[li].keys[0]   # (heads, seq, head_dim)
            v = kv.layers[li].values[0]
            k_np = k.permute(1, 0, 2).reshape(seq_len, self.kv_dim).detach().float().cpu().numpy().astype(np.float32)
            v_np = v.permute(1, 0, 2).reshape(seq_len, self.kv_dim).detach().float().cpu().numpy().astype(np.float32)
            payload = np.concatenate([k_np.ravel(), v_np.ravel()])
            layer_payloads.append((li, payload))

        if auto_link:
            result = self.engine.mem_write_pack_with_auto_link(
                self.owner, retrieval_key, layer_payloads, salience,
                auto_link_threshold=auto_link_threshold, text=fact_text,
            )
            return result["pack_id"]

        # Plain write without auto-link discovery.
        return self.engine.mem_write_pack(
            self.owner, retrieval_key, layer_payloads, salience, text=fact_text
        )

    def forget(self, pack_id):
        """Delete a memory permanently. Irreversible."""
        self.engine.delete_pack(pack_id)

    def _build_query_ids_from_pack(self, pack, query_text, device):
        """Helper extracted in the refactor pass of the Adapter rollout.

        The three retrieval methods (`retrieve_and_inject`,
        `retrieve_with_trace`, `retrieve_and_inject_multi`) all do the
        same dance: recover the stored fact text from the pack, ask the
        adapter for `(fact_messages, full_messages)`, encode both, and
        slice `full_ids[fact_len:]` to obtain the `query_ids` tensor that
        gets forwarded through the model with `past_key_values = cache`.

        Centralised here so changes to the encoding strategy (e.g. a
        future adapter that produces a different message-shape pair)
        only touch one site.
        """
        stored_fact_text = (
            pack.get("text")
            or self.engine.pack_text(pack["pack_id"])
            or ""
        )
        fact_messages, full_messages = self.adapter.retrieve_messages(
            stored_fact_text, query_text
        )
        fact_fmt = self.tokenizer.apply_chat_template(
            fact_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        full_fmt = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_ids = self.tokenizer.encode(full_fmt, return_tensors="pt").to(device)
        fact_len = len(self.tokenizer.encode(fact_fmt))
        return full_ids[:, fact_len:]

    def retrieve_and_inject(self, query_text):
        """Retrieve the best matching memory and build a DynamicCache.

        Returns (cache, query_ids, attention_mask) ready for model.generate(),
        or (None, query_ids, None) if no memory found.
        """
        # Tensors that touch the model must live on the model's device.
        # `.to(device)` is a no-op when the device already matches.
        device = self.model.device

        # Build the query portion of the chat template
        # We need: [system: fact][user: query][assistant: ...]
        # The fact was stored with system template. The query continues from there.
        # For retrieval, compute hidden states of the query text.
        query_input = self.tokenizer.encode(query_text, return_tensors="pt").to(device)
        with torch.no_grad():
            query_out = self.model(query_input, output_hidden_states=True)

        hidden = query_out.hidden_states[self.query_layer][0]
        h_tokens = hidden[1:].float().cpu().numpy().astype(np.float32)
        query_key = encode_per_token(h_tokens, self.hidden_size)

        # Retrieve via Rust pack API (returns complete pack with all layers)
        packs = self.engine.mem_read_pack(query_key, 1, self.owner)
        if not packs:
            return None, query_input, None

        pack = packs[0]
        layers = pack["layers"]
        if len(layers) < self.n_layers:
            return None, query_input, None

        # Reconstruct DynamicCache from pack layers
        sample_data = np.array(layers[0]["data"], dtype=np.float32)
        half = len(sample_data) // 2
        seq_len = half // self.kv_dim

        cache = DynamicCache()
        # Reconstructed K/V must share dtype with the model. Storage
        # round-trips through float32 numpy (Q4 quantisation requires it),
        # but a BF16 / FP16 model running scaled_dot_product_attention
        # will reject an FP32 cache with "query, key, value must have
        # the same dtype". Cast back to the model's dtype here.
        model_dtype = next(self.model.parameters()).dtype
        for layer_info in sorted(layers, key=lambda l: l["layer_idx"]):
            val = np.array(layer_info["data"], dtype=np.float32)
            half = len(val) // 2
            kt = torch.tensor(val[:half]).reshape(1, seq_len, self.num_kv_heads, self.head_dim)
            kt = kt.permute(0, 2, 1, 3).to(device=device, dtype=model_dtype)
            vt = torch.tensor(val[half:]).reshape(1, seq_len, self.num_kv_heads, self.head_dim)
            vt = vt.permute(0, 2, 1, 3).to(device=device, dtype=model_dtype)
            cache.update(kt, vt, layer_info["layer_idx"])

        # Build query_ids via the adapter-driven helper. The cache (loaded
        # from the engine) already encodes the stored fact's KV state; the
        # helper computes the post-fact suffix of the full prompt encoding
        # so model.generate(query_ids, past_key_values=cache, ...) sees a
        # consistent sequence.
        query_ids = self._build_query_ids_from_pack(pack, query_text, device)

        kv_len = cache.get_seq_length()
        q_len = query_ids.shape[1]
        attention_mask = torch.ones(1, kv_len + q_len, dtype=torch.long, device=device)

        return cache, query_ids, attention_mask

    @staticmethod
    def _normalize_gen_kwargs(gen_kwargs):
        """Suppress sampling defaults when greedy decoding is requested.

        Models like Qwen3 ship `temperature`/`top_p`/`top_k` defaults in their
        generation_config for sampling mode. Calling `model.generate()` with
        `do_sample=False` then triggers transformers' "generation flags not
        valid and may be ignored" warning — informational but noisy.

        When the caller explicitly opts into greedy decoding, we explicitly
        null those fields (without mutating the model's config) so the call
        is clean. We never override values the caller passed themselves —
        `setdefault` only fills in unset keys.
        """
        if gen_kwargs.get("do_sample") is False:
            for key in ("temperature", "top_p", "top_k"):
                gen_kwargs.setdefault(key, None)
        return gen_kwargs

    def generate(self, query_text, **gen_kwargs):
        """Full pipeline: retrieve memory, inject KV, generate response.

        Returns (generated_text, prompt_tokens, had_memory).
        """
        gen_kwargs = self._normalize_gen_kwargs(gen_kwargs)
        cache, query_ids, attn_mask = self.retrieve_and_inject(query_text)
        q_len = query_ids.shape[1]

        if cache is None:
            # No memory found — generate without injection
            with torch.no_grad():
                out = self.model.generate(query_ids, **gen_kwargs)
            text = self.tokenizer.decode(out[0][q_len:], skip_special_tokens=True).strip()
            return text, q_len, False

        # Clone cache to avoid in-place mutation
        clone = DynamicCache()
        for li in range(len(cache.layers)):
            layer = cache.layers[li]
            clone.update(layer.keys.clone(), layer.values.clone(), li)

        with torch.no_grad():
            out = self.model.generate(
                query_ids,
                past_key_values=clone,
                attention_mask=attn_mask,
                **gen_kwargs,
            )

        text = self.tokenizer.decode(out[0][q_len:], skip_special_tokens=True).strip()
        return text, q_len, True

    # -- Trace-linked storage and retrieval ------------------------------------

    EDGE_FOLLOWS: int = EDGE_FOLLOWS
    EDGE_CONTRADICTS: int = EDGE_CONTRADICTS
    EDGE_SUPPORTS: int = EDGE_SUPPORTS

    def store_and_link(self, fact_text, related_pack_id, salience=80.0):
        """Store a fact and link it to an existing memory (Follows edge).

        Returns the new pack_id.
        """
        pack_id = self.store(fact_text, salience=salience, auto_link=False)
        self.engine.add_pack_link(pack_id, related_pack_id)
        return pack_id

    def store_supporting(self, fact_text, related_pack_id, salience=80.0):
        """Store a fact that supports an existing memory (Supports edge).

        Use when the new fact reinforces or elaborates on an existing one.

        Returns the new pack_id.
        """
        pack_id = self.store(fact_text, salience=salience, auto_link=False)
        self.engine.add_pack_edge(pack_id, related_pack_id, self.EDGE_SUPPORTS)
        return pack_id

    def store_contradicting(self, fact_text, related_pack_id, salience=80.0):
        """Store a fact that contradicts an existing memory (Contradicts edge).

        Use when the new fact invalidates or corrects an existing one.

        Returns the new pack_id.
        """
        pack_id = self.store(fact_text, salience=salience, auto_link=False)
        self.engine.add_pack_edge(pack_id, related_pack_id, self.EDGE_CONTRADICTS)
        return pack_id

    def store_linked(self, facts, salience=80.0):
        """Store related facts and link them for multi-hop retrieval.

        Creates bidirectional links between all packs so that retrieving
        any one of them discovers the rest via trace traversal.

        Returns list of pack_ids.
        """
        pack_ids = []
        for fact in facts:
            pack_id = self.store(fact, salience)
            pack_ids.append(pack_id)

        # Link all packs to each other via Rust engine
        for i, pid in enumerate(pack_ids):
            for other_pid in pack_ids[i + 1:]:
                self.engine.add_pack_link(pid, other_pid)

        return pack_ids

    def retrieve_with_trace(self, query_text, k=1, composer=None, boost_factor=0.3):
        """Retrieve memories with trace-boosted scoring, then follow links.

        Trace-Boosted Retrieval: memories with trace connections get a
        score boost proportional to their link count. This promotes
        connected memories (discovery hubs) over isolated ones that
        score slightly higher on content similarity alone.

        Returns (cache, query_ids, attention_mask) or (None, query_ids, None).
        """
        if composer is None:
            composer = NaiveConcatComposer()

        # Tensors that touch the model must live on the model's device.
        device = self.model.device

        query_input = self.tokenizer.encode(query_text, return_tensors="pt").to(device)
        with torch.no_grad():
            query_out = self.model(query_input, output_hidden_states=True)

        hidden = query_out.hidden_states[self.query_layer][0]
        h_tokens = hidden[1:].float().cpu().numpy().astype(np.float32)
        query_key = encode_per_token(h_tokens, self.hidden_size)

        # Trace-Boosted Retrieval with link traversal: single Rust call
        # handles expanded retrieval, score boosting, re-ranking, and
        # following trace links to discover related packs.
        packs = self.engine.mem_read_pack_with_trace_boost_and_follow(
            query_key, k, self.owner, boost_factor
        )
        if not packs:
            return None, query_input, None

        cache = composer.compose(
            packs, self.num_kv_heads, self.head_dim, self.kv_dim, self.n_layers
        )
        cache = _move_cache_to_device(cache, device)

        # Use the highest-scored pack's stored text for adapter-side
        # boundary computation. Multiple packs are composed in the cache;
        # the prompt template only needs ONE fact-text instance to compute
        # the byte prefix length.
        query_ids = self._build_query_ids_from_pack(packs[0], query_text, device)

        kv_len = cache.get_seq_length()
        q_len = query_ids.shape[1]
        attention_mask = torch.ones(1, kv_len + q_len, dtype=torch.long, device=device)

        return cache, query_ids, attention_mask

    def generate_with_trace(self, query_text, k=1, composer=None, boost_factor=0.3, **gen_kwargs):
        """Full pipeline: retrieve + trace hop + compose + inject + generate.

        Returns (generated_text, prompt_tokens, had_memory).
        """
        gen_kwargs = self._normalize_gen_kwargs(gen_kwargs)
        cache, query_ids, attn_mask = self.retrieve_with_trace(
            query_text, k=k, composer=composer, boost_factor=boost_factor
        )
        q_len = query_ids.shape[1]

        if cache is None:
            with torch.no_grad():
                out = self.model.generate(query_ids, **gen_kwargs)
            text = self.tokenizer.decode(out[0][q_len:], skip_special_tokens=True).strip()
            return text, q_len, False

        clone = DynamicCache()
        for li in range(len(cache.layers)):
            layer = cache.layers[li]
            clone.update(layer.keys.clone(), layer.values.clone(), li)

        with torch.no_grad():
            out = self.model.generate(
                query_ids,
                past_key_values=clone,
                attention_mask=attn_mask,
                **gen_kwargs,
            )

        text = self.tokenizer.decode(out[0][q_len:], skip_special_tokens=True).strip()
        return text, q_len, True

    # -- Multi-memory injection ------------------------------------------------

    def retrieve_and_inject_multi(self, query_text, k=3, composer=None):
        """Retrieve k memories and compose into a single DynamicCache.

        Returns (cache, query_ids, attention_mask) ready for model.generate(),
        or (None, query_ids, None) if no memories found.
        """
        if composer is None:
            composer = NaiveConcatComposer()

        # Tensors that touch the model must live on the model's device.
        device = self.model.device

        query_input = self.tokenizer.encode(query_text, return_tensors="pt").to(device)
        with torch.no_grad():
            query_out = self.model(query_input, output_hidden_states=True)

        hidden = query_out.hidden_states[self.query_layer][0]
        h_tokens = hidden[1:].float().cpu().numpy().astype(np.float32)
        query_key = encode_per_token(h_tokens, self.hidden_size)

        packs = self.engine.mem_read_pack(query_key, k, self.owner)
        if not packs:
            return None, query_input, None

        cache = composer.compose(
            packs, self.num_kv_heads, self.head_dim, self.kv_dim, self.n_layers
        )
        cache = _move_cache_to_device(cache, device)

        # Multi-pack composition: use the top-ranked pack's stored text for
        # adapter-side boundary computation. The composed cache spans all
        # k packs; the prompt template only needs ONE fact-text instance.
        query_ids = self._build_query_ids_from_pack(packs[0], query_text, device)

        kv_len = cache.get_seq_length()
        q_len = query_ids.shape[1]
        attention_mask = torch.ones(1, kv_len + q_len, dtype=torch.long, device=device)

        return cache, query_ids, attention_mask

    def generate_multi(self, query_text, k=3, composer=None, **gen_kwargs):
        """Full pipeline: retrieve k memories, compose, inject, generate.

        Returns (generated_text, prompt_tokens, had_memory).
        """
        gen_kwargs = self._normalize_gen_kwargs(gen_kwargs)
        cache, query_ids, attn_mask = self.retrieve_and_inject_multi(
            query_text, k=k, composer=composer
        )
        q_len = query_ids.shape[1]

        if cache is None:
            with torch.no_grad():
                out = self.model.generate(query_ids, **gen_kwargs)
            text = self.tokenizer.decode(out[0][q_len:], skip_special_tokens=True).strip()
            return text, q_len, False

        clone = DynamicCache()
        for li in range(len(cache.layers)):
            layer = cache.layers[li]
            clone.update(layer.keys.clone(), layer.values.clone(), li)

        with torch.no_grad():
            out = self.model.generate(
                query_ids,
                past_key_values=clone,
                attention_mask=attn_mask,
                **gen_kwargs,
            )

        text = self.tokenizer.decode(out[0][q_len:], skip_special_tokens=True).strip()
        return text, q_len, True
