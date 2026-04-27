"""vLLM client with governed memory prefix injection.

Prepends a deterministic memory prefix (assembled by MemoryPrefixBuilder)
to every prompt before sending it to vLLM. Because the prefix text is
token-identical across requests for the same owner, vLLM's automatic
prefix-cache serves the stored KV at zero prefill cost on repeat requests.

Two deployment modes:
  - Offline (vllm.LLM): call prepare_prompt() then llm.generate()
  - Online (OpenAI-compatible API): call prepare_messages() then post to /v1/chat/completions

Usage:
    from tardigrade_vllm.prefix_client import VLLMMemoryClient

    client = VLLMMemoryClient(engine, owner=1)
    prompt = client.prepare_prompt("What is the vault code?")
    # → "Memory context:\\n- Agent Snibblex reported...\\n\\nWhat is the vault code?"

    output = llm.generate([prompt])
"""

import tardigrade_db
from tardigrade_hooks.prefix_builder import MemoryPrefixBuilder, PrefixResult
from tardigrade_hooks.prefix_format import BulletListFormat


class VLLMMemoryClient:
    """Prepends governed memory prefix to prompts for vLLM serving.

    The prefix is rebuilt on each call to pick up governance changes
    (new memories, tier promotions, decayed evictions). The version
    hash lets callers detect when the prefix has changed so upstream
    caches can be invalidated.
    """

    def __init__(
        self,
        engine,
        owner,
        format=None,
        include_validated=True,
        token_budget=None,
        tokenizer=None,
        separator="\n\n",
    ):
        self._builder = MemoryPrefixBuilder(
            engine,
            owner,
            format=format or BulletListFormat(),
            include_validated=include_validated,
            token_budget=token_budget,
            tokenizer=tokenizer,
        )
        self._separator = separator
        self._cached_result: PrefixResult | None = None

    def build_prefix(self) -> PrefixResult:
        self._cached_result = self._builder.build()
        return self._cached_result

    def prepare_prompt(self, user_prompt: str) -> str:
        result = self.build_prefix()
        if not result.text:
            return user_prompt
        return result.text + self._separator + user_prompt

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Prepend memory prefix as a system message for chat-style APIs.

        If messages already start with a system message, the memory
        prefix is prepended to its content. Otherwise a new system
        message is inserted at position 0.
        """
        result = self.build_prefix()
        if not result.text:
            return messages

        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                **messages[0],
                "content": result.text + self._separator + messages[0]["content"],
            }
        else:
            messages.insert(0, {"role": "system", "content": result.text})
        return messages

    @property
    def version(self) -> int:
        if self._cached_result is None:
            self.build_prefix()
        return self._cached_result.version

    @property
    def prefix_pack_ids(self) -> list[int]:
        if self._cached_result is None:
            self.build_prefix()
        return self._cached_result.pack_ids

    def has_changed(self, previous_version: int) -> bool:
        return self._builder.has_changed(previous_version)
