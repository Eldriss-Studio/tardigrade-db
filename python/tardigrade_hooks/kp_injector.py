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

from .encoding import encode_per_token
from .multi_composer import NaiveConcatComposer
from transformers import DynamicCache

import tardigrade_db


class KnowledgePackStore:
    """Stores and retrieves complete KV caches through TardigradeDB.

    Each memory is stored as N cells (one per layer), all sharing the
    same cell_id prefix. The retrieval key is a hidden-state summary
    for Top5Avg matching. The value is the full K+V payload per layer.
    """

    def __init__(self, engine, model, tokenizer, owner=1, query_layer=None):
        self.engine = engine
        self.model = model
        self.tokenizer = tokenizer
        self.owner = owner

        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.hidden_size = cfg.hidden_size

        if query_layer is None:
            self.query_layer = int(self.n_layers * 0.67)
        else:
            self.query_layer = query_layer

        # Pack ID -> original fact text (for sequential recomputation)
        self._text_registry = {}

    def store(self, fact_text, salience=80.0, auto_link=True, auto_link_threshold=None):
        """Store a fact's KV cache across all layers.

        Wraps the fact in the model's chat template before computing KV.
        Stores hidden states as retrieval key (for Top5Avg matching)
        and full K+V payload as injection value (per layer).

        If auto_link is True, searches existing memories before writing
        and creates trace links to similar ones (Zettelkasten pattern).

        Returns the pack_id assigned by the engine.
        """
        # Chat template wrapping
        messages = [{"role": "system", "content": fact_text}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(formatted, return_tensors="pt")
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            out = self.model(input_ids, use_cache=True, output_hidden_states=True)

        kv = out.past_key_values

        # Retrieval key: hidden states at query_layer (per-token, skip pos 0)
        hidden = out.hidden_states[self.query_layer][0]  # (seq, hidden_size)
        h_tokens = hidden[1:].numpy().astype(np.float32)  # skip pos 0
        retrieval_key = encode_per_token(h_tokens, self.hidden_size)

        # Build layer payloads for pack API
        layer_payloads = []
        for li in range(self.n_layers):
            k = kv.layers[li].keys[0]   # (heads, seq, head_dim)
            v = kv.layers[li].values[0]
            k_np = k.permute(1, 0, 2).reshape(seq_len, self.kv_dim).detach().cpu().numpy().astype(np.float32)
            v_np = v.permute(1, 0, 2).reshape(seq_len, self.kv_dim).detach().cpu().numpy().astype(np.float32)
            payload = np.concatenate([k_np.ravel(), v_np.ravel()])
            layer_payloads.append((li, payload))

        # Auto-link: search existing memories BEFORE writing (otherwise the
        # new pack outscores everything when querying with its own key)
        auto_link_matches = []
        if auto_link and self.engine.pack_count() > 0:
            threshold = auto_link_threshold if auto_link_threshold is not None else 250.0
            existing = self.engine.mem_read_pack(retrieval_key, 1, self.owner)
            auto_link_matches = [
                p["pack_id"] for p in existing if p["score"] >= threshold
            ]

        # Single atomic write via Rust pack API (one fsync for all layers)
        pack_id = self.engine.mem_write_pack(
            self.owner, retrieval_key, layer_payloads, salience
        )

        # Register fact text for sequential recomputation
        self._text_registry[pack_id] = fact_text

        # Create trace links to similar existing packs (via Rust engine)
        for match_id in auto_link_matches:
            self.engine.add_pack_link(pack_id, match_id)

        return pack_id

    def retrieve_and_inject(self, query_text):
        """Retrieve the best matching memory and build a DynamicCache.

        Returns (cache, query_ids, attention_mask) ready for model.generate(),
        or (None, query_ids, None) if no memory found.
        """
        # Build the query portion of the chat template
        # We need: [system: fact][user: query][assistant: ...]
        # The fact was stored with system template. The query continues from there.
        # For retrieval, compute hidden states of the query text.
        query_input = self.tokenizer.encode(query_text, return_tensors="pt")
        with torch.no_grad():
            query_out = self.model(query_input, output_hidden_states=True)

        hidden = query_out.hidden_states[self.query_layer][0]
        h_tokens = hidden[1:].numpy().astype(np.float32)
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
        for layer_info in sorted(layers, key=lambda l: l["layer_idx"]):
            val = np.array(layer_info["data"], dtype=np.float32)
            half = len(val) // 2
            kt = torch.tensor(val[:half]).reshape(1, seq_len, self.num_kv_heads, self.head_dim)
            kt = kt.permute(0, 2, 1, 3)
            vt = torch.tensor(val[half:]).reshape(1, seq_len, self.num_kv_heads, self.head_dim)
            vt = vt.permute(0, 2, 1, 3)
            cache.update(kt, vt, layer_info["layer_idx"])

        # Build query_ids as continuation of the stored fact
        # The stored fact used system template. The query should use user template.
        fact_messages = [{"role": "system", "content": "placeholder"}]
        fact_fmt = self.tokenizer.apply_chat_template(
            fact_messages, tokenize=False, add_generation_prompt=False
        )

        messages = [
            {"role": "system", "content": "placeholder"},
            {"role": "user", "content": query_text},
        ]
        full_fmt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # The query portion starts after the system template
        full_ids = self.tokenizer.encode(full_fmt, return_tensors="pt")
        fact_len = len(self.tokenizer.encode(fact_fmt))
        query_ids = full_ids[:, fact_len:]

        kv_len = cache.get_seq_length()
        q_len = query_ids.shape[1]
        attention_mask = torch.ones(1, kv_len + q_len, dtype=torch.long)

        return cache, query_ids, attention_mask

    def generate(self, query_text, **gen_kwargs):
        """Full pipeline: retrieve memory, inject KV, generate response.

        Returns (generated_text, prompt_tokens, had_memory).
        """
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

    def store_and_link(self, fact_text, related_pack_id, salience=80.0):
        """Store a fact and link it to an existing memory.

        Use when the agent learns a new detail about something it already
        remembers. The engine records the link; the agent decides what to link.

        Returns the new pack_id.
        """
        pack_id = self.store(fact_text, salience=salience, auto_link=False)
        self.engine.add_pack_link(pack_id, related_pack_id)
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

        query_input = self.tokenizer.encode(query_text, return_tensors="pt")
        with torch.no_grad():
            query_out = self.model(query_input, output_hidden_states=True)

        hidden = query_out.hidden_states[self.query_layer][0]
        h_tokens = hidden[1:].numpy().astype(np.float32)
        query_key = encode_per_token(h_tokens, self.hidden_size)

        # Retrieve more candidates than k for trace-boost re-ranking
        k_expanded = max(k * 5, 10)
        packs = self.engine.mem_read_pack(query_key, k_expanded, self.owner)
        if not packs:
            return None, query_input, None

        # Trace-Boosted Retrieval: re-rank by connection density (via Rust)
        for pack in packs:
            link_count = len(self.engine.pack_links(pack["pack_id"]))
            pack["score"] *= (1.0 + link_count * boost_factor)

        packs.sort(key=lambda p: p["score"], reverse=True)
        packs = packs[:k]

        # Follow trace links to discover related packs (via Rust)
        retrieved_ids = {p["pack_id"] for p in packs}
        linked_ids = set()
        for p in packs:
            linked_ids.update(self.engine.pack_links(p["pack_id"]))

        for pid in linked_ids - retrieved_ids:
            try:
                packs.append(self.engine.load_pack_by_id(pid))
            except Exception:
                pass  # Pack may have been deleted

        cache = composer.compose(
            packs, self.num_kv_heads, self.head_dim, self.kv_dim, self.n_layers
        )

        fact_messages = [{"role": "system", "content": "placeholder"}]
        fact_fmt = self.tokenizer.apply_chat_template(
            fact_messages, tokenize=False, add_generation_prompt=False
        )
        messages = [
            {"role": "system", "content": "placeholder"},
            {"role": "user", "content": query_text},
        ]
        full_fmt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_ids = self.tokenizer.encode(full_fmt, return_tensors="pt")
        fact_len = len(self.tokenizer.encode(fact_fmt))
        query_ids = full_ids[:, fact_len:]

        kv_len = cache.get_seq_length()
        q_len = query_ids.shape[1]
        attention_mask = torch.ones(1, kv_len + q_len, dtype=torch.long)

        return cache, query_ids, attention_mask

    def generate_with_trace(self, query_text, k=1, composer=None, boost_factor=0.3, **gen_kwargs):
        """Full pipeline: retrieve + trace hop + compose + inject + generate.

        Returns (generated_text, prompt_tokens, had_memory).
        """
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

        query_input = self.tokenizer.encode(query_text, return_tensors="pt")
        with torch.no_grad():
            query_out = self.model(query_input, output_hidden_states=True)

        hidden = query_out.hidden_states[self.query_layer][0]
        h_tokens = hidden[1:].numpy().astype(np.float32)
        query_key = encode_per_token(h_tokens, self.hidden_size)

        packs = self.engine.mem_read_pack(query_key, k, self.owner)
        if not packs:
            return None, query_input, None

        cache = composer.compose(
            packs, self.num_kv_heads, self.head_dim, self.kv_dim, self.n_layers
        )

        # Query IDs: user template continuation after system
        fact_messages = [{"role": "system", "content": "placeholder"}]
        fact_fmt = self.tokenizer.apply_chat_template(
            fact_messages, tokenize=False, add_generation_prompt=False
        )
        messages = [
            {"role": "system", "content": "placeholder"},
            {"role": "user", "content": query_text},
        ]
        full_fmt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_ids = self.tokenizer.encode(full_fmt, return_tensors="pt")
        fact_len = len(self.tokenizer.encode(fact_fmt))
        query_ids = full_ids[:, fact_len:]

        kv_len = cache.get_seq_length()
        q_len = query_ids.shape[1]
        attention_mask = torch.ones(1, kv_len + q_len, dtype=torch.long)

        return cache, query_ids, attention_mask

    def generate_multi(self, query_text, k=3, composer=None, **gen_kwargs):
        """Full pipeline: retrieve k memories, compose, inject, generate.

        Returns (generated_text, prompt_tokens, had_memory).
        """
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
