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

    def store(self, fact_text, salience=80.0):
        """Store a fact's KV cache across all layers.

        Wraps the fact in the model's chat template before computing KV.
        Stores hidden states as retrieval key (for Top5Avg matching)
        and full K+V payload as injection value (per layer).

        Returns the number of cells written (= n_layers).
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
        # Encoded with sentinel header for PerTokenRetriever
        hidden = out.hidden_states[self.query_layer][0]  # (seq, hidden_size)
        h_tokens = hidden[1:].numpy().astype(np.float32)  # skip pos 0
        n_tok = len(h_tokens)
        header = np.array([-1.0e9, float(n_tok), float(self.hidden_size)], dtype=np.float32)
        retrieval_key = np.concatenate([header, h_tokens.ravel()])

        # Build layer payloads for pack API
        layer_payloads = []
        for li in range(self.n_layers):
            k = kv.layers[li].keys[0]   # (heads, seq, head_dim)
            v = kv.layers[li].values[0]
            k_np = k.permute(1, 0, 2).reshape(seq_len, self.kv_dim).detach().cpu().numpy().astype(np.float32)
            v_np = v.permute(1, 0, 2).reshape(seq_len, self.kv_dim).detach().cpu().numpy().astype(np.float32)
            payload = np.concatenate([k_np.ravel(), v_np.ravel()])
            layer_payloads.append((li, payload))

        # Single atomic write via Rust pack API (one fsync for all layers)
        self.engine.mem_write_pack(
            self.owner, retrieval_key, layer_payloads, salience
        )

        return self.n_layers

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
        n_tok = len(h_tokens)
        header = np.array([-1.0e9, float(n_tok), float(self.hidden_size)], dtype=np.float32)
        query_key = np.concatenate([header, h_tokens.ravel()])

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
