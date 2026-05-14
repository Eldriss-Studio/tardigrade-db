# Reflective Latent Search (RLS): Agentic Retrieval for TardigradeDB

> **⚠️ Note — 2026-05-14.** The premise of this design — bypassing a 67.2% LoCoMo "vocabulary-mismatch ceiling" via reformulation — is retracted. The baseline measured the lexical fallback adapter, the LIMIT-paper citation as theoretical support is unjustified by our data, and RLS is measured as net-negative (−5pp to −13pp) on a clean dataset. See [`../../experiments/2026-05-14-bench-audit.md`](../../experiments/2026-05-14-bench-audit.md). Preserved below as a historical design artifact.

## Problem

TardigradeDB's latent-space retrieval hits a theoretical ceiling on
vocabulary-mismatched queries (DeepMind LIMIT, ICLR 2026). LoCoMo sits
at 67.2%, below the vanilla GPT-4o baseline (74%). We proved that no
geometric transform (whitening, reweighting, multi-layer fusion) moves
the needle — the remaining failures require world-knowledge reasoning.

RLS bypasses this ceiling by letting the agent reformulate queries and
re-retrieve in latent space. All retrieval stays tensor-native.

## Design Patterns

### Strategy Pattern — `ReformulationStrategy`

Swappable reformulation algorithms. Each takes a query string and returns
a list of reformulated variants. The RLS loop doesn't know how
reformulation happens — it just gets alternative texts to try.

```
ReformulationStrategy (ABC)
├── KeywordExpansionStrategy   (synonym expansion, no generation)
└── MultiPhrasingStrategy      (template-based rephrasing)
```

OCP: new strategies added without modifying the RLS loop.

### Template Method — `ReflectiveLatentSearch`

The RLS loop follows a fixed sequence: RETRIEVE → EVALUATE → REFORMULATE
→ RE-RETRIEVE → FUSE. Subclasses (or injected strategies) control the
reformulation step. The loop structure is invariant.

### Decorator Pattern — Adapter Integration

`ReflectiveLatentSearch` wraps the existing hook + engine retrieval path.
When RLS is disabled (`TDB_RLS_MODE=none`), the adapter calls the hook
directly — zero overhead. When enabled, the RLS wrapper intercepts the
query and adds the reflective loop. The adapter doesn't change shape.

## ATDD Acceptance Criteria

### Reformulation Strategies:
1. `KeywordExpansionStrategy.reformulate("athletic achievements")` returns
   a list containing text with synonyms (e.g., "running", "marathon")
2. `MultiPhrasingStrategy.reformulate("What does Sonia know about languages?")`
   returns 2-3 distinct variants
3. Both strategies return empty list for empty/None input
4. Both strategies are substitutable (implement `ReformulationStrategy`)

### RLS Loop:
5. High-confidence retrieval (score ratio > threshold) returns single-shot
   results without reformulation
6. Low-confidence retrieval triggers reformulation and returns fused results
7. Fused results include memories found only by reformulated queries
8. Max attempts is configurable and respected (no infinite loops)
9. Default `max_attempts=2` (original + 1 reformulation)

### Integration:
10. `TDB_RLS_MODE=none` produces identical results to current behavior
11. `TDB_RLS_MODE=keyword` uses `KeywordExpansionStrategy`
12. `TDB_RLS_MODE=multiphrasing` uses `MultiPhrasingStrategy`
13. `TDB_RLS_MODE=both` chains keyword then multiphrasing

### Benchmark:
14. All RLS modes do not regress specific R@5 below 100% on 10-fact corpus
15. At least one RLS mode improves vague R@5 > 60% on 10-fact corpus

## Components

### 1. `ReformulationStrategy` ABC (`python/tardigrade_hooks/rls.py`)

```python
class ReformulationStrategy(ABC):
    @abstractmethod
    def reformulate(self, query_text: str) -> list[str]: ...
```

### 2. `KeywordExpansionStrategy` (`python/tardigrade_hooks/rls.py`)

Extracts content words (filtering stop words), expands each with a
built-in synonym map, concatenates into an expanded query string.

```
Input: "What does Sonia know about languages?"
Content words: ["sonia", "know", "languages"]
Expanded: "Sonia know languages translation linguistic foreign translated"
```

The synonym map covers the domains in the Sonia corpus and common
vocabulary bridges:
```python
SYNONYMS = {
    "languages": ["translation", "translated", "linguistic", "foreign"],
    "athletic": ["running", "marathon", "ultramarathon", "race", "endurance"],
    "mechanical": ["engine", "motor", "motorcycle", "machine", "restored"],
    "nature": ["ecology", "peatlands", "composting", "organic", "environmental"],
    "chemicals": ["spectrometer", "calibration", "lab", "chemistry", "equipment"],
    ...
}
```

Returns `[expanded_text]` — a single expanded version.

### 3. `MultiPhrasingStrategy` (`python/tardigrade_hooks/rls.py`)

Generates template-based variants without LLM generation:

```python
def reformulate(self, query_text: str) -> list[str]:
    content_words = extract_content_words(query_text)
    keyword_only = " ".join(content_words)
    subject = content_words[0] if content_words else ""
    who_what = f"What did {subject} do related to {' '.join(content_words[1:])}"
    return [keyword_only, who_what]
```

Returns 2 variants: keyword-only and a "What did X do" rephrasing.

### 4. `ReflectiveLatentSearch` (`python/tardigrade_hooks/rls.py`)

```python
class ReflectiveLatentSearch:
    def __init__(
        self,
        hook,
        model,
        tokenizer,
        query_layer: int,
        strategies: list[ReformulationStrategy],
        confidence_threshold: float = 1.5,
        max_attempts: int = 2,
    ): ...

    def query(self, question: str, top_k: int = 5) -> list[MemoryCellHandle]:
        # Step 1: initial retrieval via hook
        handles = self._retrieve(question)
        
        # Step 2: confidence check
        if self._is_confident(handles):
            return handles[:top_k]
        
        # Step 3-4: reformulate and re-retrieve
        all_results = [handles]
        for strategy in self.strategies:
            variants = strategy.reformulate(question)
            for variant in variants[:self.max_attempts - 1]:
                new_handles = self._retrieve(variant)
                all_results.append(new_handles)
        
        # Step 5: fuse via RRF
        return self._fuse(all_results, top_k)
```

**Confidence check:** `_is_confident()` computes score ratio
`handles[0].score / handles[1].score` if ≥2 results. Ratio above
`confidence_threshold` (default 1.5) means confident — skip reformulation.
If <2 results, always reformulate.

**Fusion:** Uses `rrf_fuse()` from `multi_layer_query.py`, adapted to
work with `MemoryCellHandle` lists instead of pack dicts.

### 5. Adapter Integration (`python/tdb_bench/adapters/tardigrade.py`)

New env var `TDB_RLS_MODE`:
- `none` (default): current behavior, no RLS
- `keyword`: `KeywordExpansionStrategy`
- `multiphrasing`: `MultiPhrasingStrategy`
- `both`: chains both strategies

When `TDB_RLS_MODE != none`, the adapter wraps the hook in
`ReflectiveLatentSearch` at init time. The `query()` method delegates
to `rls.query()` instead of `hook.on_prefill()` directly.

## Files

| File | Action | Pattern | What |
|------|--------|---------|------|
| `python/tardigrade_hooks/rls.py` | Create | Strategy + Template Method | `ReformulationStrategy` ABC, 2 strategies, `ReflectiveLatentSearch` loop |
| `python/tdb_bench/adapters/tardigrade.py` | Modify | Decorator | Add RLS wrapper when `TDB_RLS_MODE` is set |
| `tests/python/test_rls.py` | Create | — | ATDD tests for strategies + RLS loop |
| `experiments/rls_experiment.py` | Create | — | 10-fact corpus experiment |

## What Does NOT Change

- Rust engine (no changes)
- Benchmark runner, evaluator, datasets, CLI
- Other adapters (Mem0, Letta)
- Existing refinement pipeline (whitening, centering, reweighting)
- MCP tools
- Default behavior (`TDB_RLS_MODE=none`)

## Verification

1. `pytest tests/python/test_rls.py -v` — all ATDD tests pass
2. `TDB_RLS_MODE=none` benchmark matches previous results exactly
3. 10-fact corpus experiment with all RLS modes
4. If vague improves: full LoCoMo re-run with best mode
5. Compare against 67.2% baseline

## SOLID Analysis

- **SRP:** Each strategy reformulates. RLS loop orchestrates. Adapter delegates.
- **OCP:** New strategies register without modifying the loop.
- **LSP:** All strategies implement `ReformulationStrategy` — substitutable.
- **ISP:** Strategy has one method: `reformulate()`. Minimal.
- **DIP:** RLS depends on `ReformulationStrategy` abstraction, not concrete classes.
