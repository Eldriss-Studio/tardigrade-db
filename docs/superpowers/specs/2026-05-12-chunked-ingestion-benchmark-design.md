# Chunked Ingestion for Benchmark Adapter + Multi-Config LoCoMo Matrix

## Problem

The benchmark adapter truncates LoCoMo's 9K-token conversational contexts
to 256 tokens and stores one cell per item. This produces near-identical
hidden states across all memories — score ratio = 1.000 for every query.
Every technique tested (whitening, reweighting, multi-layer, RLS) was
running on degenerate scores and produced 0% improvement.

## Design

### Pattern: Template Method (existing adapter) + Composition (TextChunker)

The adapter's `ingest()` method changes from single-cell truncation to
multi-cell chunking using the existing `TextChunker` (built earlier this
session). No new patterns — reuse existing infrastructure.

### ATDD Acceptance Criteria

1. Chunked adapter ingests a 1000-word context into ≥5 cells (not 1)
2. All cells for one item map back to the same `BenchmarkItem` via `_cell_to_item`
3. Score ratio for queries is NOT 1.000 (scores differentiate between chunks)
4. Chunked baseline produces a LoCoMo score different from 68.2%
5. `TextChunker` chunk size is configurable via `TDB_BENCH_CHUNK_TOKENS` env var (default 128)
6. Overlap is configurable via `TDB_BENCH_CHUNK_OVERLAP` env var (default 16)

### Components

#### 1. Modified `ingest()` in `python/tdb_bench/adapters/tardigrade.py`

Replace the truncation path (lines 211-230) with:

```python
from tardigrade_hooks.chunker import TextChunker

_CHUNK_TOKENS = int(os.getenv("TDB_BENCH_CHUNK_TOKENS", "128"))
_CHUNK_OVERLAP = int(os.getenv("TDB_BENCH_CHUNK_OVERLAP", "16"))

# In ingest():
chunker = TextChunker(tokenizer, max_tokens=_CHUNK_TOKENS, overlap_tokens=_CHUNK_OVERLAP)
for item in items:
    chunks = chunker.chunk(item.context)
    for chunk in chunks:
        inputs = tokenizer(chunk.text, return_tensors="pt", truncation=True, max_length=_CHUNK_TOKENS)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        decision = self._hook.on_generate(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )
        if not decision.should_write:
            continue
        cell_id = self._engine.mem_write(
            1, query_layer, decision.key, decision.value, decision.salience, None,
        )
        self._cell_to_item[int(cell_id)] = item
    self._store.insert(item)
```

All cells from all chunks of one item map to the same `BenchmarkItem`.
The query side doesn't change — it finds the best-scoring cell and looks
up `_cell_to_item[cell_id]` to get the answer.

#### 2. Named Constants

```python
_CHUNK_TOKENS = int(os.getenv("TDB_BENCH_CHUNK_TOKENS", "128"))
_CHUNK_OVERLAP = int(os.getenv("TDB_BENCH_CHUNK_OVERLAP", "16"))
```

#### 3. Benchmark Matrix

Run 4 configs sequentially, all with chunked ingestion:

| # | Config | Env vars |
|---|--------|----------|
| 1 | Chunked baseline | `TDB_REFINEMENT_MODE=centered` |
| 2 | Chunked + whitened | `TDB_REFINEMENT_MODE=whitened` |
| 3 | Chunked + RLS keyword | `TDB_REFINEMENT_MODE=centered TDB_RLS_MODE=keyword` |
| 4 | Chunked + RLS generative | `TDB_REFINEMENT_MODE=centered TDB_RLS_MODE=generative` |

Each run produces a separate output JSON. Compare all against the old
68.2% / 90.9% baselines.

### Files

| File | Action | What |
|------|--------|------|
| `python/tdb_bench/adapters/tardigrade.py` | Modify | Replace truncation with chunked ingestion |
| `tests/python/test_tardigrade_adapter.py` | Create | ATDD tests for chunked ingestion |

### What Does NOT Change

- `TextChunker` (already built, tested)
- `query()` method (unchanged — cell_id lookup is the same)
- Benchmark runner, evaluator, datasets, CLI
- Engine, refinement pipeline, RLS strategies
- Other adapters (Mem0, Letta)

### SOLID

- **SRP:** TextChunker chunks. Adapter ingests. Engine stores. Each unchanged.
- **OCP:** Chunk size configurable via env var without code changes.
- **DIP:** Adapter depends on TextChunker abstraction (already injected).

### Verification

1. `pytest tests/python/test_tardigrade_adapter.py -v` — ATDD tests pass
2. Score ratio check on 20-item LoCoMo subset — ratios ≠ 1.000
3. Full LoCoMo + LongMemEval × 4 configs — compare all against 68.2% / 90.9%
