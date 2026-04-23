# Plan: LlamaCppHook — GGUF Model Support for TardigradeDB

## Context

TardigradeDB's hook system only supports HuggingFace transformers (`HuggingFaceHook`). People running local models via **Ollama** and **LM Studio** use GGUF files, which are invisible to the current system. In our two-agent experiment, the GPT-2 test (12 layers via HF) achieved 80% recall, while the Llama 3.2 test (1 layer via raw llama-cpp-python) dropped to 60%. The gap is both an integration gap (no hook for GGUF) and a quality gap (single-layer vs multi-layer extraction).

**Goal:** Add a `LlamaCppHook` that lets users point at an Ollama/LM Studio model by name (e.g., `"llama3.2:3b"`) and get working memory storage and retrieval — following ATDD and established design patterns.

**Constraint:** `llama-cpp-python` only exposes **final-layer embeddings**, not per-layer hidden states. We compensate with a multi-view embedding strategy.

## Architecture Decisions

- **TardigradeHook ABC** — No changes. `layer` is just an int; single-layer hooks use `layer=0..N` for views.
- **WriteDecision / MemoryCellHandle** — No changes. Multi-view produces multiple WriteDecisions, not new fields.
- **Engine (Rust)** — No changes. `mem_write`/`mem_read` are layer-agnostic.
- **kv_injector.py** — Not used in GGUF path (no PyTorch DynamicCache injection for llama-cpp models).

## Design Patterns

| Pattern | Component | Purpose |
|---------|-----------|---------|
| **Adapter** | `LlamaCppHook` | Translates llama-cpp-python embedding API → TardigradeHook |
| **Strategy** | `EmbeddingView` | Pluggable embedding projections (mean, max, first, last) |
| **Factory** | `GGUFModelResolver` | Resolves `"llama3.2:3b"` → `/path/to/blob` |
| Template Method | `TardigradeHook` (existing) | Hook lifecycle |

## New Files

```
python/tardigrade_hooks/
    embedding_views.py      # NEW — Strategy: view projections
    gguf_resolver.py        # NEW — Factory: model name → GGUF path
    llama_cpp_hook.py       # NEW — Adapter: LlamaCppHook

tests/python/
    test_embedding_views.py # NEW — AT-6 through AT-10
    test_gguf_resolver.py   # NEW — AT-1 through AT-5
    test_llama_cpp_hook.py  # NEW — AT-11 through AT-18
```

**Modified files:**
- `python/tardigrade_hooks/__init__.py` — export `LlamaCppHook`, `GGUFModelResolver`
- `examples/llama_memory_test.py` — refactor to use `LlamaCppHook` instead of raw code

---

## Implementation Sequence (ATDD-first)

### Step 1: Acceptance Tests for EmbeddingViews

**File:** `tests/python/test_embedding_views.py`

Pure numpy tests, no model or engine needed.

| Test | What it proves |
|------|---------------|
| AT-6 | `MeanPoolView` produces `(d_model,)` from `(n_tokens, d_model)` |
| AT-7 | `MaxPoolView` picks per-dimension maximum |
| AT-8 | `FirstTokenView` returns row 0, `LastTokenView` returns row -1 |
| AT-9 | Custom callable conforming to `EmbeddingView` protocol works |
| AT-10 | 4 default views produce pairwise-distinct vectors |

### Step 2: Implement EmbeddingViews

**File:** `python/tardigrade_hooks/embedding_views.py`

```python
@dataclass
class EmbeddingView:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    
    def __call__(self, token_embeddings: np.ndarray) -> np.ndarray:
        return self.fn(token_embeddings).astype(np.float32)

MEAN_POOL  = EmbeddingView("mean",  lambda t: t.mean(axis=0))
MAX_POOL   = EmbeddingView("max",   lambda t: t.max(axis=0))
FIRST_TOKEN = EmbeddingView("first", lambda t: t[0])
LAST_TOKEN  = EmbeddingView("last",  lambda t: t[-1])

DEFAULT_VIEWS = [MEAN_POOL, MAX_POOL, FIRST_TOKEN, LAST_TOKEN]
```

Make AT-6 through AT-10 pass.

### Step 3: Acceptance Tests for GGUFModelResolver

**File:** `tests/python/test_gguf_resolver.py`

Uses a temp directory with synthetic Ollama manifests for unit tests. Real-path tests marked `@pytest.mark.skipif` for CI.

| Test | What it proves |
|------|---------------|
| AT-1 | `resolve("llama3.2:3b")` → correct blob path from Ollama manifest |
| AT-2 | `resolve("llama3.2")` without tag → tries `latest` first |
| AT-3 | LM Studio model resolves to `.gguf` file path |
| AT-4 | Unknown model raises `GGUFNotFoundError` with helpful message |
| AT-5 | `list_models()` returns available models from both sources |

### Step 4: Implement GGUFModelResolver

**File:** `python/tardigrade_hooks/gguf_resolver.py`

Ollama resolution algorithm:
1. Parse `"llama3.2:3b"` → `(name="llama3.2", tag="3b")`
2. Read `~/.ollama/models/manifests/registry.ollama.ai/library/{name}/{tag}` as JSON
3. Find layer with `mediaType == "application/vnd.ollama.image.model"`
4. Return `~/.ollama/models/blobs/{digest}` (replacing `:` with `-` in digest)

LM Studio: recursive glob for `*.gguf` in `~/.lmstudio/models/`, fuzzy match on name.

Make AT-1 through AT-5 pass.

### Step 5: Acceptance Tests for LlamaCppHook

**File:** `tests/python/test_llama_cpp_hook.py`

Uses the existing `engine` fixture pattern from `test_hook.py`. Mock llama-cpp-python for unit tests; integration test with real GGUF marked `skipif`.

| Test | What it proves |
|------|---------------|
| AT-11 | `isinstance(LlamaCppHook(...), TardigradeHook)` is True |
| AT-12 | `on_generate` with high-norm embeddings → `should_write=True` |
| AT-13 | `on_generate` with near-zero embeddings → `should_write=False` |
| AT-14 | `WriteDecision.key` has shape `(d_model,)` and dtype `float32` |
| AT-15 | After writing cells, `on_prefill` retrieves `MemoryCellHandle` list |
| AT-16 | Multi-view mode: `store_memory(text)` produces 4 cells (one per view at layers 0-3) |
| AT-17 | Multi-view retrieval: store a memory, query with related text, correct memory in top-k |
| AT-18 | Integration: real GGUF extraction produces `float32` array of expected dimension |

### Step 6: Implement LlamaCppHook

**File:** `python/tardigrade_hooks/llama_cpp_hook.py`

Key design:
- Constructor accepts `model: str | Path` — string triggers `GGUFModelResolver`
- `text_to_embeddings(text) → np.ndarray (n_tokens, d_model)` — extracts per-token embeddings
- `on_generate(layer, hidden_states)` — applies view at `layer` index, returns WriteDecision
- `on_prefill(layer, query_states)` — queries engine using view at `layer` index
- `store_memory(text, salience=None) → list[int]` — convenience: extract → all views → mem_write
- `query_memory(text, k=None) → list[MemoryCellHandle]` — convenience: extract → best match across views

Make AT-11 through AT-18 pass.

### Step 7: Update Exports and Example

- Update `python/tardigrade_hooks/__init__.py` to export `LlamaCppHook`, `GGUFModelResolver`
- Refactor `examples/llama_memory_test.py` to use `LlamaCppHook` instead of raw code

### Step 8: Refactor Pass

- Check for duplication between `hf_hook.py` and `llama_cpp_hook.py` (salience heuristic logic)
- Extract shared salience computation if appropriate
- Naming review: consistency of `owner` vs `agent_id`, `model` vs `model_path`

### Step 9: Gap Review

- Error handling: corrupt Ollama manifests, missing blobs, model loading OOM
- Edge cases: 1-token input (first/last views collapse), empty string, very long text exceeding n_ctx
- What if `llama-cpp-python` returns 1D embedding instead of 2D? (Some models only return pooled)

---

## Verification

### Unit tests
```bash
cd ~/Dev/tardigrade-db && source .venv/bin/activate
pytest tests/python/test_embedding_views.py tests/python/test_gguf_resolver.py tests/python/test_llama_cpp_hook.py -v
```

### Integration test (requires Ollama models on disk)
```bash
pytest tests/python/test_llama_cpp_hook.py -v -k "AT_18"
```

### End-to-end: Two-agent character test with LlamaCppHook
```bash
python examples/llama_memory_test.py clear
python examples/llama_memory_test.py store "memory one" "memory two"
python examples/llama_memory_test.py query "related question"
python examples/llama_memory_test.py info
```

### Recall comparison
Rerun the two-agent Kael test with multi-view LlamaCppHook and compare:
- Word-hash baseline: 92% recall
- GPT-2 (12 layers): 80% recall
- Llama 3.2 single-layer: 60% recall
- **Llama 3.2 multi-view (4 views): target ≥ 70% recall**

### All existing tests still pass
```bash
cargo test --workspace --exclude tdb-python
pytest tests/python/ -v
```
