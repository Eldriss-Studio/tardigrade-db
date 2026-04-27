# Path 2: Memory Prefix Adapter

**Date:** April 27, 2026  
**Status:** Complete — end-to-end verified on GPU with vLLM 0.19.1 + Qwen3-0.6B (4/4 tests passing)

## The Problem

Path 1 proved that KV injection transfers knowledge (9/10 synthetic facts, 100% recall ratio). But the HuggingFace path (`model.generate(past_key_values=...)`) serves one request at a time in a single Python process — it can't power a production API.

vLLM handles production serving (batching, paging, multi-user GPU sharing), but its KV Connector v1 API only supports **prefix-cache** — token-identical prefix matching. It cannot inject cross-prompt KV.

## The Solution

`MemoryPrefixBuilder` composes a deterministic text prefix per owner from their governed memories. Because the same owner's prefix produces identical tokens across requests, vLLM's stock prefix-cache serves the stored KV automatically — zero prefill cost on repeat requests, no fork required.

TardigradeDB decides **what** goes in the prefix (governance: tier filtering, importance ranking, token budgets). vLLM handles **serving** it efficiently (KV caching, batching, scheduling).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TardigradeDB Engine                   │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Storage  │  │  Governance  │  │  list_packs(owner) │  │
│  │ (KV+text)│  │  (AKL tiers) │  │  → tier+importance │  │
│  └──────────┘  └──────────────┘  └─────────┬─────────┘  │
│                                            │             │
│                                   ┌────────▼────────┐   │
│                                   │ PrefixBuilder   │   │
│                                   │ filter → sort → │   │
│                                   │ budget → format │   │
│                                   └────────┬────────┘   │
└────────────────────────────────────────────┼────────────┘
                                             │
                            ┌────────────────▼────────────────┐
                            │         PrefixResult            │
                            │  text: "Memory context:\n-..."  │
                            │  version: 0x3a7f...  (hash)     │
                            │  pack_ids: [3, 7, 12]           │
                            │  token_estimate: 145            │
                            └────────────────┬────────────────┘
                                             │
                     ┌───────────────────────┼───────────────────────┐
                     │                       │                       │
              ┌──────▼──────┐   ┌────────────▼──────────┐  ┌────────▼────────┐
              │  HF Path    │   │  vLLM Prefix Path     │  │  MCP / API      │
              │  (Path 1)   │   │  (Path 2 — next step) │  │  (text output)  │
              │  inject KV  │   │  prepend to prompt     │  │  return to LLM  │
              └─────────────┘   └───────────────────────┘  └─────────────────┘
```

## Implementation

### Rust: `engine.list_packs(owner)`

New engine API that enumerates all packs for an owner with governance metadata.

**`crates/tdb-engine/src/pack_directory.rs`** — added `pack_ids()` iterator over all pack IDs.

**`crates/tdb-engine/src/engine.rs`** — added `list_packs(owner_filter: Option<OwnerId>)`:
- Iterates `pack_directory.pack_ids()`
- For each pack: reads retrieval cell → gets owner from pool → applies owner filter
- Looks up governance for tier and importance (defaults to Draft/0.0 if absent)
- Returns `Vec<(PackId, OwnerId, Tier, f32)>` sorted by importance descending

**`crates/tdb-python/src/lib.rs`** — Python binding returns list of dicts:
```python
engine.list_packs(owner=1)
# → [{"pack_id": 3, "owner": 1, "tier": 2, "importance": 87.5, "text": "The capital of..."}, ...]
```

### Python: `MemoryPrefixBuilder`

**`python/tardigrade_hooks/prefix_builder.py`**

```python
builder = MemoryPrefixBuilder(
    engine,
    owner=1,
    format=BulletListFormat(),    # or TierAnnotatedFormat()
    include_validated=True,        # include Validated tier (not just Core)
    token_budget=500,              # max prefix tokens (None = unlimited)
    tokenizer=tokenizer,           # for accurate counting (None = len//4 estimate)
)

result = builder.build()
# result.text     → "Memory context:\n- fact1\n- fact2\n..."
# result.version  → 0x3a7f... (SHA-256 content hash, changes when memories change)
# result.pack_ids → [3, 7, 12]
# result.token_estimate → 145

builder.has_changed(old_version)  # True if prefix content differs
```

**Tier filtering:**
- Core (salience ≥ 80 → importance 85+ → tier threshold 85): always included
- Validated (salience ~60 → importance ~65 → tier threshold 65): included when `include_validated=True`
- Draft (salience < 55 → importance < 65): never included

**Token budget:** Drops lowest-importance memories first until prefix fits within budget. Header ("Memory context:\n") cost counted.

**Version:** SHA-256 of `(pack_id, text)` pairs for all selected memories, truncated to 64-bit int. Deterministic — no external state needed.

### Format strategies

**`python/tardigrade_hooks/prefix_format.py`**

| Strategy | Output | Use case |
|----------|--------|----------|
| `BulletListFormat` | `Memory context:\n- fact1\n- fact2` | Default — clean, model-agnostic |
| `TierAnnotatedFormat` | `Memory context:\n- [Core] fact1\n- [Validated] fact2` | Debugging, when the model should weigh Core higher |

Both escape newlines in fact text (replace `\n` with space) to maintain one-fact-per-line structure.

## Test Coverage

### Rust (4 acceptance tests)

| Test | What it validates |
|------|------------------|
| `test_list_packs_returns_all_packs` | Returns correct count for all packs |
| `test_list_packs_filters_by_owner` | Owner filter isolates per-user packs |
| `test_list_packs_sorted_by_importance_descending` | Higher-importance packs sort first |
| `test_list_packs_empty_engine` | Empty engine returns empty vec |

### Python (11 ATDD tests)

| Test | What it validates |
|------|------------------|
| `test_empty_engine_returns_empty_prefix` | No packs → empty text, empty pack_ids |
| `test_core_memories_included` | Core-tier packs appear in prefix |
| `test_draft_memories_excluded` | Draft-tier packs filtered out |
| `test_validated_memories_included_when_enabled` | Validated tier is optional |
| `test_prefix_is_deterministic` | Same memories → identical text and version |
| `test_prefix_ordered_by_importance` | Highest importance first |
| `test_token_budget_truncates` | Budget limits prefix length |
| `test_version_increments_on_change` | New pack → version changes |
| `test_version_stable_when_unchanged` | No change → same version |
| `test_format_strategy_swappable` | Different format → different text |
| `test_newlines_in_fact_text_escaped` | Newlines in facts don't break format |

## vLLM Client Integration (April 27, 2026)

`VLLMMemoryClient` (`python/tardigrade_vllm/prefix_client.py`) wraps `MemoryPrefixBuilder` for vLLM serving.

```python
from tardigrade_vllm.prefix_client import VLLMMemoryClient

client = VLLMMemoryClient(engine, owner=1, token_budget=500)

# Offline (vllm.LLM)
prompt = client.prepare_prompt("What is the vault code?")
# → "Memory context:\n- Agent Snibblex reported the vault code is 9-Quornth-44\n\nWhat is the vault code?"
output = llm.generate([prompt])

# Online (OpenAI chat API)
messages = client.prepare_messages([{"role": "user", "content": "What is the vault code?"}])
# → [{"role": "system", "content": "Memory context:\n- Agent Snibblex..."}, {"role": "user", ...}]

# Staleness detection
old_version = client.version
# ... time passes, governance changes ...
if client.has_changed(old_version):
    # prefix KV is stale — vLLM will recompute on next request
    pass
```

**Design decision:** The prefix is prepended at the application layer, not inside the connector. The connector operates after tokenization (it receives `prompt_token_ids`), so the prefix must be added before the prompt reaches vLLM. This keeps the connector untouched and the integration clean — vLLM's automatic prefix-cache handles KV reuse for the repeated prefix tokens.

**`prepare_messages` behavior:** If the message list already starts with a system message, the memory prefix is prepended to its content (preserving the existing system prompt). Otherwise a new system message is inserted at position 0.

### Additional test coverage (13 ATDD tests)

| Test | What it validates |
|------|------------------|
| `test_empty_engine_returns_bare_prompt` | No memories → user prompt returned unchanged |
| `test_prefix_prepended_to_prompt` | Memory prefix appears before user prompt |
| `test_custom_separator` | Configurable separator between prefix and prompt |
| `test_prepare_messages_inserts_system` | System message added when none exists |
| `test_prepare_messages_merges_system` | Prefix merged into existing system message |
| `test_empty_prefix_passthrough_messages` | No memories → messages unchanged |
| `test_version_changes_on_new_memory` | `has_changed()` detects new memories |
| `test_format_strategy_propagates` | Format strategy flows through to output |
| `test_owner_isolation` | Owner 1 sees only owner 1 memories |
| `test_token_budget_limits_prefix` | Budget produces shorter prefix |
| `test_draft_excluded_from_prefix` | Draft-tier memories filtered out |
| `test_prefix_pack_ids` | `prefix_pack_ids` reflects included packs |
| `test_prepare_messages_no_mutation` | Input message list not modified |
