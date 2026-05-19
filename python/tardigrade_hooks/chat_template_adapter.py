"""Chat-template adapters for `KnowledgePackStore`.

Design patterns: **Adapter** + **Factory Method**.

This module insulates `KnowledgePackStore` and `multi_composer` from the
quirks of individual HuggingFace tokenizer chat templates. Different model
families impose different constraints — Qwen3 was unusually lenient and
accepted `[{"role": "system", "content": ...}]` message lists; Qwen3.5,
Llama-3, Gemma-2, Mistral-Instruct, Phi-3.5, and virtually every modern
instruction-tuned template require a `user` role to be present.

Rather than special-case each family in the storage/retrieval flow, this
module exposes a small Adapter ABC with two concrete implementations and a
Factory that picks the right one by probing the tokenizer at construction
time.

::

                          ┌───────────────────────┐
                          │ ChatTemplateAdapter   │ (ABC)
                          │   store_messages()    │
                          │   retrieve_messages() │
                          └─────────┬─────────────┘
                                    │
                  ┌─────────────────┴──────────────────┐
                  ▼                                    ▼
        ┌───────────────────────┐         ┌───────────────────────┐
        │ UserMessageAdapter    │         │ LegacySystemAdapter   │
        │ (strict templates:    │         │ (lenient templates:   │
        │  Qwen3.5, Llama-3,    │         │  Qwen3, GPT-2 with    │
        │  Gemma-2, Mistral,    │         │  the custom test      │
        │  Phi-3.5)             │         │  harness template)    │
        └───────────────────────┘         └───────────────────────┘

The Factory `select_chat_template_adapter` probes the tokenizer with a
system-only `apply_chat_template` call:

- If accepted → ``LegacySystemAdapter`` (backwards-compat with existing
  packs stored under the Qwen3 wrap).
- If rejected → ``UserMessageAdapter`` (strict-template default).

The probe is read-only with respect to the tokenizer; it does not mutate
vocabulary, special tokens, or any other tokenizer state.

# Example

>>> from transformers import AutoTokenizer
>>> tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
>>> adapter = select_chat_template_adapter(tok)
>>> isinstance(adapter, LegacySystemAdapter)
True

When constructing a [`KnowledgePackStore`][KnowledgePackStore], pass an
explicit adapter to override the Factory default, or omit the kwarg to let
the Factory pick:

>>> kps = KnowledgePackStore(engine, model, tok, adapter=UserMessageAdapter())

# Known constraints

- **Store and retrieve must use the same adapter for a given pack.** The
  pack's KV state on disk was captured under one specific wrap; retrieving
  with a different adapter computes a `fact_len` boundary that no longer
  aligns with the cached state, producing incoherent inputs. This is
  enforced de facto when a consumer uses one `KnowledgePackStore` instance
  for the session (the Factory is called once in `__init__`). Mixing
  adapters across sessions on the same `tardigrade_data/` requires the
  consumer to either rebuild memory or explicitly pin to the original
  adapter.

- **Empty assistant content in `UserMessageAdapter`'s retrieve form** (the
  synthetic `assistant: ""` turn between the stored-fact user turn and
  the query user turn) works on ChatML-family templates (Qwen3, Qwen3.5,
  Llama-3, TinyLlama). If a future template rejects empty content, the
  adapter would need a single-space placeholder; the rest of the design
  stays unchanged.

- **Qwen3.5's hybrid linear/standard attention architecture** is a separate
  compatibility blocker for tardigrade-db (orthogonal to this module).
  `kv.layers[i]` for a `LinearAttentionLayer` has no `.keys` attribute,
  so `kp_injector.store` can't capture KV from those layers. Using this
  adapter with Qwen3.5 fixes the chat-template error but the KV-capture
  step then fails with `AttributeError`. Supporting Qwen3.5 fully is a
  separate library concern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ChatTemplateAdapter(ABC):
    """Adapter for tokenizer chat-template idiosyncrasies.

    Concrete implementations wrap a model family's expectations behind a
    stable two-method contract so the rest of tardigrade-db can stay
    template-agnostic.
    """

    @abstractmethod
    def store_messages(self, fact_text: str) -> list[dict]:
        """Messages-list to pass to `tokenizer.apply_chat_template()` when
        storing a fact's KV state.

        The returned list must be acceptable by the target tokenizer's
        chat template (must not raise `jinja2.TemplateError`).
        """

    @abstractmethod
    def retrieve_messages(
        self,
        stored_fact_text: str,
        query_text: str,
    ) -> tuple[list[dict], list[dict]]:
        """Messages-lists for the retrieval-side template-length trick.

        Returns ``(fact_messages, full_messages)``. The caller:

        1. Applies the tokenizer's chat template to both lists.
        2. Encodes both to token-id tensors.
        3. Computes ``fact_len = len(encode(fact_fmt))``.
        4. Slices ``full_ids[:, fact_len:]`` to obtain ``query_ids`` (the
           tokens to forward through the model with ``past_key_values =
           cache``).

        The contract: the encoded ``fact_messages`` must be a strict prefix
        of the encoded ``full_messages`` (under the target tokenizer's
        chat template). Otherwise the slice produces garbage.

        ``stored_fact_text`` is supplied so adapters that need exact
        prefix parity (e.g., the ``UserMessageAdapter``) can reconstruct
        the fact's wrapping precisely. Adapters that don't need it (e.g.,
        ``LegacySystemAdapter``, which uses a literal placeholder for
        byte-for-byte compatibility with pre-adapter stored packs) may
        ignore it.
        """


class UserMessageAdapter(ChatTemplateAdapter):
    """Wraps facts as user-role messages.

    Compatible with any modern instruction-tuned chat template — including
    Qwen3.5, Llama-3, Gemma-2, Mistral-Instruct, and Phi-3.5 — that
    requires at least one `user` message in `apply_chat_template`.

    On retrieve, the canonical multi-turn shape ``[user, assistant, user]``
    is used. The empty assistant turn carries no semantic content; its
    only role is to separate the two user turns for templates that
    disallow consecutive same-role messages.
    """

    def store_messages(self, fact_text: str) -> list[dict]:
        return [{"role": "user", "content": fact_text}]

    def retrieve_messages(
        self,
        stored_fact_text: str,
        query_text: str,
    ) -> tuple[list[dict], list[dict]]:
        fact_msgs = [{"role": "user", "content": stored_fact_text}]
        full_msgs = [
            {"role": "user", "content": stored_fact_text},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": query_text},
        ]
        return fact_msgs, full_msgs


class LegacySystemAdapter(ChatTemplateAdapter):
    """Wraps facts as system-role messages — the original `KnowledgePackStore`
    wrap from before the Adapter pattern was introduced.

    Preserved for backwards compatibility with stored packs already on
    disk under this wrap (a switch to `UserMessageAdapter` would key
    them differently because the cache encodes different KV state under
    different wraps). Also used by the GPT-2 + custom-chat-template test harness
    where the lenient template explicitly supports system-only message
    lists.

    On retrieve, this adapter uses a literal ``"placeholder"`` for the
    system content in `fact_messages`, matching the pre-adapter code byte
    for byte. The stored fact text is therefore unused by this adapter's
    `retrieve_messages` — the cache (loaded from the engine) carries the
    real fact's KV state; the template-length computation only needs the
    structural shape, not the content.
    """

    def store_messages(self, fact_text: str) -> list[dict]:
        return [{"role": "system", "content": fact_text}]

    def retrieve_messages(
        self,
        stored_fact_text: str,  # noqa: ARG002 — see docstring
        query_text: str,
    ) -> tuple[list[dict], list[dict]]:
        fact_msgs = [{"role": "system", "content": "placeholder"}]
        full_msgs = [
            {"role": "system", "content": "placeholder"},
            {"role": "user", "content": query_text},
        ]
        return fact_msgs, full_msgs


def select_chat_template_adapter(tokenizer) -> ChatTemplateAdapter:
    """Factory Method: pick the right `ChatTemplateAdapter` for a tokenizer.

    Probes the tokenizer by calling `apply_chat_template` with a
    system-only message list:

    - If accepted (no exception) → returns `LegacySystemAdapter`, preserving
      backwards compatibility with packs already stored under the Qwen3
      system-wrap.
    - If rejected (any exception) → returns `UserMessageAdapter`, the
      strict-template default that works on Qwen3.5, Llama-3, Gemma-2,
      Mistral-Instruct, Phi-3.5, and other modern templates.
    - If `tokenizer` is `None` (consumer running in a non-local-model
      mode, e.g. a hosted-API path) → returns `UserMessageAdapter` as a
      safe default; the caller likely won't exercise the adapter, and if
      they do, the strict form is the safer choice.

    The probe is read-only: it does not mutate vocabulary, special
    tokens, or any other tokenizer state.

    # Example

    >>> from transformers import AutoTokenizer
    >>> tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    >>> adapter = select_chat_template_adapter(tok)
    >>> isinstance(adapter, UserMessageAdapter)
    True
    """
    if tokenizer is None:
        return UserMessageAdapter()
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "probe"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return LegacySystemAdapter()
    except Exception:
        return UserMessageAdapter()
