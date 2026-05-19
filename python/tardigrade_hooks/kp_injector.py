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
    DEFAULT_STORE_SALIENCE,
    EDGE_CONTRADICTS,
    EDGE_FOLLOWS,
    EDGE_SUPPORTS,
)
from .encoding import encode_per_token
from ._hidden_states import _softmax_layer_payloads, softmax_layer_count
from .multi_composer import NaiveConcatComposer
from transformers import DynamicCache

# Note: `import tardigrade_db` is intentionally NOT at module level. CI
# lint jobs (e.g. bench-smoke-gate) import `tardigrade_hooks.constants`
# without first running `maturin develop`, so eager-importing the
# native extension here breaks unrelated lint paths. The native
# extension is reached via the `engine` instance the caller passes in;
# no module-level reference is needed.


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
        retrieval_key_strategy=None,
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
        # ``n_softmax_layers`` — the architecture-aware count used by both
        # the storage write filter (skip recurrent layers) and the read-side
        # pack-integrity guard (a pack is complete when its softmax layers
        # are present, not when it has *every* layer index — hybrid models
        # never have every index). On uniform-softmax models this equals
        # ``n_layers``, so the guard's strength is unchanged.
        self.n_softmax_layers = softmax_layer_count(cfg)
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.hidden_size = cfg.hidden_size

        # Strategy + layer selection order:
        #   1. Explicit `retrieval_key_strategy` wins outright.
        #   2. Else: explicit `query_layer` → HiddenStateKeyStrategy(query_layer).
        #   3. Else: consult `calibration_registry` if provided. Cached
        #      result drives BOTH the strategy choice (best_strategy:
        #      "hidden_state" → HiddenStateKeyStrategy, "k_vector" →
        #      KVectorKeyStrategy) AND the layer choice (best_layer).
        #      Unknown strategy name → fall back to HiddenStateKeyStrategy.
        #   4. Else: HiddenStateKeyStrategy at the static-ratio layer.
        #      This is the pre-Phase-2 default; works on uniform-softmax
        #      models, fails silently on hybrid (calibrate to fix).
        from .retrieval_key_strategy import (
            HiddenStateKeyStrategy,
            KVectorKeyStrategy,
        )
        default_layer = int(self.n_layers * DEFAULT_CAPTURE_LAYER_RATIO)
        if retrieval_key_strategy is not None:
            self.retrieval_key_strategy = retrieval_key_strategy
        elif query_layer is not None:
            self.retrieval_key_strategy = HiddenStateKeyStrategy(query_layer)
        elif calibration_registry is not None:
            cached = calibration_registry.load(_model_id_for(model))
            if cached is None:
                self.retrieval_key_strategy = HiddenStateKeyStrategy(default_layer)
            elif cached.best_strategy == "k_vector":
                self.retrieval_key_strategy = KVectorKeyStrategy(cached.best_layer)
            else:
                # "hidden_state" or any unknown name → HiddenStateKeyStrategy
                # at the cached layer. Graceful degradation for forward-
                # compat with strategies we don't yet recognise.
                self.retrieval_key_strategy = HiddenStateKeyStrategy(cached.best_layer)
        else:
            self.retrieval_key_strategy = HiddenStateKeyStrategy(default_layer)

        # `self.query_layer` is preserved for backwards compat with code
        # that reads it directly (a number of older call sites). It
        # mirrors the strategy's layer index regardless of whether the
        # strategy is hidden-state or k-vector.
        self.query_layer = self.retrieval_key_strategy.layer_index if hasattr(
            self.retrieval_key_strategy, "layer_index"
        ) else getattr(
            self.retrieval_key_strategy, "query_layer",
            getattr(self.retrieval_key_strategy, "softmax_layer_idx", default_layer),
        )

    def store(self, fact_text, salience=DEFAULT_STORE_SALIENCE, auto_link=True, auto_link_threshold=None):
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

        # Pre-instantiate cache so the K vectors are accessible whether
        # the model returns `output.past_key_values` (uniform-softmax
        # Qwen3/Llama-3/Mistral path) or only mutates the passed-in
        # cache in place (RecurrentGemma's CausalLMOutput path which
        # omits the .past_key_values attribute entirely).
        cache_in = DynamicCache(config=self.model.config)
        with torch.no_grad():
            out = self.model(
                input_ids,
                past_key_values=cache_in,
                use_cache=True,
                output_hidden_states=True,
            )
        returned = getattr(out, "past_key_values", None)
        if (
            returned is not None
            and hasattr(returned, "get_seq_length")
            and returned.get_seq_length() > 0
        ):
            kv = returned
        else:
            kv = cache_in

        # Retrieval key via the configured strategy. Hidden-state strategy
        # reads `out.hidden_states[query_layer]`; K-vector strategy reads
        # K from `kv.layers[softmax_layer_idx]`. Same engine, different
        # encoding — picked by architecture or by calibration.
        hidden_per_layer = [
            h[0].float().cpu().numpy().astype(np.float32)
            for h in out.hidden_states
        ]
        retrieval_key = self.retrieval_key_strategy.compute(
            hidden_per_layer, kv, self.hidden_size
        )

        # Build layer payloads via the softmax-only filter. On uniform-
        # softmax models this iterates every layer; on hybrid models
        # (RecurrentGemma, Jamba, Qwen3-Next, …) the recurrent layers are
        # skipped — they have no ``.keys`` / ``.values`` to store.
        layer_payloads = self._build_layer_payloads(kv, seq_len)

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

    def _build_layer_payloads(self, kv, seq_len):
        """Build the per-layer K/V payload list for ``mem_write_pack``.

        Delegates to :func:`_softmax_layer_payloads`, which filters out
        layers without ``.keys`` (recurrent / linear-attention layers on
        hybrid models — see Michalak & Abreu 2025 + the v11 spike for why
        only softmax layers carry observable retrieval signal). Exposed
        as an instance method so future strategies can override it
        without rewriting ``store()``.
        """
        return _softmax_layer_payloads(kv, seq_len, self.kv_dim)

    def forget(self, pack_id):
        """Delete a memory permanently. Irreversible."""
        self.engine.delete_pack(pack_id)

    def _compute_query_key(self, query_text):
        """Strategy-aware query-key extraction.

        Returns ``(query_key, query_input_ids)``. The forward pass uses
        a pre-instantiated DynamicCache so hybrid models (where
        ``output.past_key_values`` is absent) still expose their K
        tensors via the in-place-mutated cache. The configured strategy
        decides whether to read from hidden states or from K vectors.
        """
        device = self.model.device
        query_input = self.tokenizer.encode(query_text, return_tensors="pt").to(device)
        cache_in = DynamicCache(config=self.model.config)
        with torch.no_grad():
            query_out = self.model(
                query_input,
                past_key_values=cache_in,
                use_cache=True,
                output_hidden_states=True,
            )
        returned = getattr(query_out, "past_key_values", None)
        if (
            returned is not None
            and hasattr(returned, "get_seq_length")
            and returned.get_seq_length() > 0
        ):
            qkv = returned
        else:
            qkv = cache_in
        hidden_per_layer = [
            h[0].float().cpu().numpy().astype(np.float32)
            for h in query_out.hidden_states
        ]
        query_key = self.retrieval_key_strategy.compute(
            hidden_per_layer, qkv, self.hidden_size
        )
        return query_key, query_input

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

        # Compute the query's retrieval key via the configured strategy.
        # (HiddenStateKeyStrategy reads hidden_states[query_layer];
        # KVectorKeyStrategy reads K from cache.layers[softmax_layer_idx].)
        query_key, query_input = self._compute_query_key(query_text)

        # Retrieve via Rust pack API (returns complete pack with all layers)
        packs = self.engine.mem_read_pack(query_key, 1, self.owner)
        if not packs:
            return None, query_input, None

        pack = packs[0]
        layers = pack["layers"]
        # Pack-integrity guard: every softmax layer must be present. Hybrid
        # models persist fewer layers than ``n_layers`` (recurrent layers
        # carry no K/V); the architecture-aware count is what matters.
        if len(layers) < self.n_softmax_layers:
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

    def store_and_link(self, fact_text, related_pack_id, salience=DEFAULT_STORE_SALIENCE):
        """Store a fact and link it to an existing memory (Follows edge).

        Returns the new pack_id.
        """
        pack_id = self.store(fact_text, salience=salience, auto_link=False)
        self.engine.add_pack_link(pack_id, related_pack_id)
        return pack_id

    def store_supporting(self, fact_text, related_pack_id, salience=DEFAULT_STORE_SALIENCE):
        """Store a fact that supports an existing memory (Supports edge).

        Use when the new fact reinforces or elaborates on an existing one.

        Returns the new pack_id.
        """
        pack_id = self.store(fact_text, salience=salience, auto_link=False)
        self.engine.add_pack_edge(pack_id, related_pack_id, self.EDGE_SUPPORTS)
        return pack_id

    def store_contradicting(self, fact_text, related_pack_id, salience=DEFAULT_STORE_SALIENCE):
        """Store a fact that contradicts an existing memory (Contradicts edge).

        Use when the new fact invalidates or corrects an existing one.

        Returns the new pack_id.
        """
        pack_id = self.store(fact_text, salience=salience, auto_link=False)
        self.engine.add_pack_edge(pack_id, related_pack_id, self.EDGE_CONTRADICTS)
        return pack_id

    def store_linked(self, facts, salience=DEFAULT_STORE_SALIENCE):
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

        # Strategy-aware query-key extraction (see _compute_query_key).
        query_key, query_input = self._compute_query_key(query_text)

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

        # Strategy-aware query-key extraction (see _compute_query_key).
        query_key, query_input = self._compute_query_key(query_text)

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
