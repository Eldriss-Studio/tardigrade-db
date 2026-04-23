# Plan: Dual-Store Architecture + GGUF Support

## Context

Tonight's experiments fundamentally changed our understanding of TardigradeDB:

1. **Mean-pooled hidden states** work for **retrieval** (80-92% recall in two-agent test) but are **catastrophically broken for injection** (P → 0, model outputs "?" at 89%). Root cause: hidden states live in `d_model` space, KV cache lives in `head_dim` space — reshaping one as the other is a category error.

2. **Full per-token KV injection** works dramatically well (26x-829x improvement, matching/exceeding Text RAG). Even Q4-quantized KV retains 89% of quality.

3. **The GGUFModelResolver** (finding Ollama/LM Studio model files by name) is still needed and unchanged.

4. **The multi-view embedding strategy is obsolete.** It was designed to compensate for single-layer extraction, but the real problem isn't too few views — it's that mean-pooled vectors can't be injected at all.

## What Changed

| Component | Old Plan | New Plan |
|-----------|----------|----------|
| **Storage** | Mean-pooled hidden states only | **Dual-store: mean-pooled index + full per-token KV** |
| **EmbeddingViews** | 4 view strategies (mean, max, first, last) | **Removed — multi-view doesn't solve the real problem** |
| **Injection** | Reshape mean-pool → `(1, nH, 1, hD)` via kv_injector | **Inject full KV as `past_key_values`** |
| **kv_injector.py** | Keep as-is | **Rewrite — current reshape approach is broken** |
| **GGUFModelResolver** | Find GGUF files by name | **Unchanged — still needed** |
| **LlamaCppHook** | Extract embeddings, multi-view | **Retrieval-only hook (no injection — llama-cpp can't extract KV)** |
| **HuggingFaceHook** | Mean-pool → inject | **Full KV capture → store both index and KV** |

## The Revised Architecture

```
WRITE PATH (HuggingFace models — full access):
  Text → Model forward pass
       → Extract mean-pooled hidden state per layer → SEARCH INDEX (what TardigradeDB stores today)
       → Extract full per-token KV cache → PAYLOAD (new: stored alongside index)
       → Q4 quantize both
       → engine.mem_write(index=mean_pool, payload=full_kv)

WRITE PATH (GGUF models — embedding only):
  Text → llama-cpp-python → Final-layer embedding → SEARCH INDEX only
       → No KV extraction possible
       → Retrieval works, injection not available

READ PATH:
  Query → Extract hidden state → Similarity search against indices
       → If full KV available: inject as past_key_values (zero re-encoding)
       → If only index available: return text/metadata (application does RAG)
```

## Design Patterns

| Pattern | Component | Purpose |
|---------|-----------|---------|
| **Adapter** | `LlamaCppHook` | Translates llama-cpp-python embedding API → TardigradeHook (retrieval only) |
| **Adapter** | `HuggingFaceHook` (revised) | Full KV capture + mean-pooled indexing |
| **Factory** | `GGUFModelResolver` | Resolves `"llama3.2:3b"` → `/path/to/blob` |
| **Strategy** | `InjectionStrategy` | Pluggable injection: full KV, text fallback, or none |
| Template Method | `TardigradeHook` (existing) | Hook lifecycle |

## Scope Decision: What to Build Now vs Later

### Now (this plan)

1. **GGUFModelResolver** — Find GGUF files from Ollama/LM Studio by name. This is self-contained, immediately useful, and unchanged from the original plan.

2. **LlamaCppHook** — Retrieval-only hook for GGUF models. Uses `llama-cpp-python` final-layer embeddings for semantic search. No injection (not possible without full KV). This closes the "Ollama/LM Studio users can't use TardigradeDB" gap.

3. **Fix kv_injector.py** — The current implementation injects mean-pooled hidden states reshaped as KV, which is broken. Either:
   - Remove it (defer injection to the dual-store work)
   - Or fix it to only accept actual KV cache tensors (not hidden states)

### Later (separate plan, requires Rust engine changes)

4. **Dual-store engine** — `mem_write` needs to accept both a search index vector AND a full KV payload. This is a Rust-level change to `tdb-engine`, `tdb-storage`, and the PyO3 bindings. Significant scope.

5. **Revised HuggingFaceHook** — Captures both mean-pooled index and full per-token KV. Writes both to the dual-store engine.

6. **KV injection path** — New injection logic that uses actual `past_key_values` from the stored KV payload, not reshaped hidden states.

7. **RoPE validation** — Test whether full KV injection works with modern models that use Rotary Position Encoding (Llama 3, Qwen, Gemma) vs GPT-2's absolute encoding.

## Implementation Sequence (ATDD-first)

### Step 1: Acceptance Tests for GGUFModelResolver

**File:** `tests/python/test_gguf_resolver.py`

| Test | What it proves |
|------|---------------|
| AT-1 | `resolve("llama3.2:3b")` → correct blob path from Ollama manifest |
| AT-2 | `resolve("llama3.2")` without tag → tries `latest` first |
| AT-3 | LM Studio model resolves to `.gguf` file path |
| AT-4 | Unknown model raises `GGUFNotFoundError` with helpful message |
| AT-5 | `list_models()` returns available models from both sources |

### Step 2: Implement GGUFModelResolver

**File:** `python/tardigrade_hooks/gguf_resolver.py`

Ollama resolution:
1. Parse `"llama3.2:3b"` → `(name="llama3.2", tag="3b")`
2. Read `~/.ollama/models/manifests/registry.ollama.ai/library/{name}/{tag}` as JSON
3. Find layer with `mediaType == "application/vnd.ollama.image.model"`
4. Return `~/.ollama/models/blobs/{digest}` (replacing `:` with `-` in digest)

LM Studio: recursive glob for `*.gguf` in `~/.lmstudio/models/`, fuzzy match.

### Step 3: Acceptance Tests for LlamaCppHook

**File:** `tests/python/test_llama_cpp_hook.py`

| Test | What it proves |
|------|---------------|
| AT-6 | `isinstance(LlamaCppHook(...), TardigradeHook)` is True |
| AT-7 | `on_generate` with high-norm embeddings → `should_write=True` |
| AT-8 | `on_generate` with near-zero embeddings → `should_write=False` |
| AT-9 | `WriteDecision.key` has shape `(d_model,)` and dtype `float32` |
| AT-10 | After writing cells, `on_prefill` retrieves `MemoryCellHandle` list |
| AT-11 | `store_memory(text)` writes 1 cell per memory (single-layer) |
| AT-12 | `query_memory(text)` retrieves correct memory from stored set |
| AT-13 | Integration: real GGUF extraction produces float32 array of expected dimension |

### Step 4: Implement LlamaCppHook

**File:** `python/tardigrade_hooks/llama_cpp_hook.py`

Retrieval-only hook:
- Constructor: `model: str | Path` — string triggers `GGUFModelResolver`
- `text_to_embedding(text) → np.ndarray (d_model,)` — final-layer embedding
- `on_generate(layer, hidden_states) → WriteDecision`
- `on_prefill(layer, query_states) → list[MemoryCellHandle]`
- `store_memory(text) → int` — convenience: extract → mem_write
- `query_memory(text, k) → list[MemoryCellHandle]` — convenience: extract → mem_read

No injection. This hook enables semantic memory retrieval for GGUF users. The application layer decides what to do with retrieved memories (e.g., prepend as text context).

### Step 5: Fix kv_injector.py

Add a guard that rejects mean-pooled hidden states:
- Check that input dimensions match `(1, num_heads, seq_len, head_dim)` for actual KV
- Raise `ValueError` if someone passes a reshaped hidden state
- Document that injection requires full per-token KV (future dual-store feature)

### Step 6: Update Exports and Examples

- `python/tardigrade_hooks/__init__.py` — export `LlamaCppHook`, `GGUFModelResolver`
- Refactor `examples/llama_memory_test.py` to use `LlamaCppHook`

### Step 7: Refactor Pass + Gap Review

- Shared salience logic between `hf_hook.py` and `llama_cpp_hook.py`
- Error handling: corrupt manifests, missing blobs, OOM
- Edge cases: empty text, very long text exceeding n_ctx, 1D vs 2D embeddings

## Verification

### Unit tests
```bash
cd ~/Dev/tardigrade-db && source .venv/bin/activate
pytest tests/python/test_gguf_resolver.py tests/python/test_llama_cpp_hook.py -v
```

### Integration (requires Ollama models)
```bash
pytest tests/python/test_llama_cpp_hook.py -v -k "AT_13"
```

### End-to-end
```bash
python examples/llama_memory_test.py clear
python examples/llama_memory_test.py store "memory one" "memory two"
python examples/llama_memory_test.py query "related question"
```

### All existing tests
```bash
cargo test --workspace --exclude tdb-python
pytest tests/python/ -v
```

## What This Plan Does NOT Cover (Deferred)

These require Rust engine changes and are a separate, larger effort:

1. **Dual-store `mem_write`** — Engine-level support for index vector + KV payload
2. **Full KV capture in HuggingFaceHook** — Storing actual `past_key_values` alongside mean-pooled index
3. **KV injection from stored payloads** — New injection path using real KV, not reshaped hidden states
4. **RoPE compatibility testing** — Validating cross-context KV injection with modern positional encoding
5. **Storage size optimization** — Q4-quantized full KV is ~40-200x larger than mean-pooled. Need segment management for larger payloads.
