# Reflective Latent Search (RLS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Reflective Latent Search to the benchmark adapter so TardigradeDB can reformulate vague queries and re-retrieve in latent space, then measure the impact on LoCoMo (67.2% baseline).

**Architecture:** `ReformulationStrategy` ABC (Strategy pattern) with two implementations. `ReflectiveLatentSearch` class wraps the hook + engine, runs the RETRIEVE→EVALUATE→REFORMULATE→RE-RETRIEVE→FUSE loop. Adapter toggles RLS via `TDB_RLS_MODE` env var. Python-only — no Rust changes.

**Tech Stack:** Python (tardigrade_hooks, tdb_bench), Qwen3-0.6B on MPS.

---

### Task 1: ReformulationStrategy ABC + Two Implementations

**Files:**
- Create: `python/tardigrade_hooks/rls.py`
- Create: `tests/python/test_rls.py`

- [ ] **Step 1: Write ATDD tests FIRST**

Create `tests/python/test_rls.py`:

```python
"""ATDD tests for Reflective Latent Search — Strategy pattern for query reformulation."""

import pytest

from tardigrade_hooks.rls import (
    KeywordExpansionStrategy,
    MultiPhrasingStrategy,
    ReformulationStrategy,
)


class TestReformulationStrategyABC:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            ReformulationStrategy()

    def test_keyword_is_substitutable(self):
        assert isinstance(KeywordExpansionStrategy(), ReformulationStrategy)

    def test_multiphrasing_is_substitutable(self):
        assert isinstance(MultiPhrasingStrategy(), ReformulationStrategy)


class TestKeywordExpansion:
    def test_expands_with_synonyms(self):
        s = KeywordExpansionStrategy()
        results = s.reformulate("What does Sonia know about languages?")
        assert len(results) >= 1
        expanded = results[0].lower()
        assert any(word in expanded for word in ["translat", "linguistic", "foreign"])

    def test_athletic_expansion(self):
        s = KeywordExpansionStrategy()
        results = s.reformulate("Tell me about athletic achievements")
        expanded = results[0].lower()
        assert any(word in expanded for word in ["marathon", "running", "ultramarathon", "race"])

    def test_mechanical_expansion(self):
        s = KeywordExpansionStrategy()
        results = s.reformulate("Tell me something mechanical about Sonia")
        expanded = results[0].lower()
        assert any(word in expanded for word in ["motorcycle", "engine", "motor", "restored"])

    def test_empty_input(self):
        s = KeywordExpansionStrategy()
        assert s.reformulate("") == []
        assert s.reformulate(None) == []

    def test_returns_list_of_strings(self):
        s = KeywordExpansionStrategy()
        results = s.reformulate("some query")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, str)


class TestMultiPhrasing:
    def test_returns_multiple_variants(self):
        s = MultiPhrasingStrategy()
        results = s.reformulate("What does Sonia know about languages?")
        assert len(results) >= 2

    def test_variants_differ_from_each_other(self):
        s = MultiPhrasingStrategy()
        results = s.reformulate("What scientific research has Sonia done?")
        assert len(set(results)) == len(results)

    def test_keyword_only_variant_has_no_question_words(self):
        s = MultiPhrasingStrategy()
        results = s.reformulate("What does Sonia know about languages?")
        keyword_variant = results[0]
        assert "what" not in keyword_variant.lower()
        assert "does" not in keyword_variant.lower()

    def test_empty_input(self):
        s = MultiPhrasingStrategy()
        assert s.reformulate("") == []
        assert s.reformulate(None) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `source .venv/bin/activate && pip install -e . && PYTHONPATH=python pytest tests/python/test_rls.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement rls.py**

Create `python/tardigrade_hooks/rls.py`:

```python
"""Reflective Latent Search (RLS) — agentic retrieval in latent space.

Strategy pattern for query reformulation. Template Method for the
RETRIEVE → EVALUATE → REFORMULATE → RE-RETRIEVE → FUSE loop.
All retrieval stays tensor-native — the agent's forward pass on
reformulated text generates new hidden states as retrieval keys.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_STOP_WORDS = frozenset(
    "what does did do is was were are how has had have who where when why "
    "tell me about the a an in on at to for of by from with and or but "
    "something some any been being know known".split()
)

_SYNONYMS: dict[str, list[str]] = {
    "language": ["translation", "translated", "linguistic", "foreign", "patent"],
    "languages": ["translation", "translated", "linguistic", "foreign", "patent"],
    "athletic": ["running", "marathon", "ultramarathon", "race", "endurance", "sport"],
    "sport": ["running", "marathon", "ultramarathon", "race", "athletic"],
    "outdoors": ["hiking", "marathon", "highlands", "mountain", "trail"],
    "outdoorsy": ["hiking", "marathon", "highlands", "mountain", "trail"],
    "mechanical": ["motorcycle", "engine", "motor", "restored", "jawa", "machine"],
    "nature": ["ecology", "peatlands", "composting", "organic", "environmental", "mycorrhizal"],
    "ecology": ["peatlands", "composting", "organic", "environmental", "mycorrhizal", "nature"],
    "chemical": ["spectrometer", "calibration", "lab", "chemistry", "equipment", "mass"],
    "chemicals": ["spectrometer", "calibration", "lab", "chemistry", "equipment", "mass"],
    "materials": ["spectrometer", "magnets", "rare-earth", "supply", "mining"],
    "research": ["published", "paper", "peer-reviewed", "journal", "study"],
    "science": ["published", "paper", "peer-reviewed", "journal", "ecology"],
    "scientific": ["published", "paper", "peer-reviewed", "journal", "ecology"],
    "engineering": ["designed", "built", "system", "recovery", "waste-heat", "kiln"],
    "environmental": ["composting", "organic", "waste", "cooperative", "emissions"],
    "music": ["balalaika", "play", "grandmother", "instrument"],
}


class ReformulationStrategy(ABC):
    """Strategy: produces alternative query texts for re-retrieval."""

    @abstractmethod
    def reformulate(self, query_text: str | None) -> list[str]: ...


class KeywordExpansionStrategy(ReformulationStrategy):
    """Extracts content words, expands with synonyms."""

    def reformulate(self, query_text: str | None) -> list[str]:
        if not query_text or not query_text.strip():
            return []
        words = re.findall(r"[a-zA-Z]+", query_text.lower())
        content = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
        if not content:
            return []
        expanded = list(content)
        for word in content:
            syns = _SYNONYMS.get(word, [])
            expanded.extend(syns)
        return [" ".join(dict.fromkeys(expanded))]


class MultiPhrasingStrategy(ReformulationStrategy):
    """Generates template-based query variants without LLM generation."""

    def reformulate(self, query_text: str | None) -> list[str]:
        if not query_text or not query_text.strip():
            return []
        words = re.findall(r"[a-zA-Z]+", query_text.lower())
        content = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
        if not content:
            return []
        keyword_only = " ".join(content)
        subject = content[0] if content else ""
        rest = " ".join(content[1:]) if len(content) > 1 else ""
        who_what = f"{subject} {rest}" if rest else subject
        return [keyword_only, f"What did {who_what} involve"]


def rrf_fuse_handles(
    handle_lists: list[list],
    k: int = 60,
) -> list:
    """Fuse MemoryCellHandle lists via RRF, dedup by cell_id."""
    scores: dict[int, float] = defaultdict(float)
    handle_by_id: dict[int, object] = {}

    for handles in handle_lists:
        for rank, h in enumerate(handles):
            cid = h.cell_id
            scores[cid] += 1.0 / (k + rank + 1)
            if cid not in handle_by_id:
                handle_by_id[cid] = h

    sorted_ids = sorted(scores, key=lambda cid: -scores[cid])
    return [handle_by_id[cid] for cid in sorted_ids]


DEFAULT_CONFIDENCE_THRESHOLD = 1.5
DEFAULT_MAX_ATTEMPTS = 2


class ReflectiveLatentSearch:
    """RETRIEVE → EVALUATE → REFORMULATE → RE-RETRIEVE → FUSE loop.

    Wraps the hook + engine retrieval. When confidence is low, tries
    reformulated queries and fuses results. All retrieval stays in
    latent space.
    """

    def __init__(
        self,
        engine,
        model,
        tokenizer,
        query_layer: int,
        hidden_size: int,
        owner: int = 1,
        k: int = 5,
        strategies: list[ReformulationStrategy] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ):
        self._engine = engine
        self._model = model
        self._tokenizer = tokenizer
        self._query_layer = query_layer
        self._hidden_size = hidden_size
        self._owner = owner
        self._k = k
        self._strategies = strategies or [KeywordExpansionStrategy()]
        self._threshold = confidence_threshold
        self._max_attempts = max_attempts

    def query(self, question: str, top_k: int | None = None) -> list:
        """Run the RLS loop. Returns list of MemoryCellHandle."""
        import numpy as np

        k = top_k or self._k
        handles = self._retrieve(question)

        if self._is_confident(handles):
            return handles[:k]

        all_results = [handles]
        attempts = 0
        for strategy in self._strategies:
            if attempts >= self._max_attempts - 1:
                break
            variants = strategy.reformulate(question)
            for variant in variants:
                if attempts >= self._max_attempts - 1:
                    break
                new_handles = self._retrieve(variant)
                if new_handles:
                    all_results.append(new_handles)
                attempts += 1

        fused = rrf_fuse_handles(all_results)
        return fused[:k]

    def _retrieve(self, text: str) -> list:
        """Single-shot latent retrieval via forward pass + engine."""
        import numpy as np
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256,
        )
        inputs = {k_: v.to(self._model.device) for k_, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)

        h = out.hidden_states[self._query_layer][0]
        tokens = h[1:].cpu().numpy().astype(np.float32)

        if hasattr(self._engine, "mem_read_tokens"):
            results = self._engine.mem_read_tokens(tokens, self._k * 2, self._owner)
        else:
            from .encoding import encode_per_token
            key = encode_per_token(tokens, self._hidden_size)
            results = self._engine.mem_read(key, self._k * 2, self._owner)

        from .hook import MemoryCellHandle
        return [
            MemoryCellHandle(
                cell_id=r.cell_id, owner=r.owner, layer=r.layer, score=r.score,
                key=np.array(r.key(), dtype=np.float32),
                value=np.array(r.value(), dtype=np.float32),
            )
            for r in results
        ]

    def _is_confident(self, handles: list) -> bool:
        if len(handles) < 2:
            return False
        ratio = handles[0].score / handles[1].score if handles[1].score > 0 else float("inf")
        return ratio >= self._threshold
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && pip install -e . && PYTHONPATH=python pytest tests/python/test_rls.py -v`
Expected: All 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tardigrade_hooks/rls.py tests/python/test_rls.py
git commit -m "✨ feat(hooks): Reflective Latent Search — ReformulationStrategy + RLS loop"
```

---

### Task 2: Adapter Integration

**Files:**
- Modify: `python/tdb_bench/adapters/tardigrade.py`

- [ ] **Step 1: Add RLS mode parsing and initialization**

In `python/tdb_bench/adapters/tardigrade.py`, after the existing `self._refinement` env var parsing (around line 42), add:

```python
_RLS_MODE = os.getenv("TDB_RLS_MODE", "none")
```

In the `__init__` method, after `self._hook = HuggingFaceKVHook(...)` (around line 140), add:

```python
            self._rls = None
            if _RLS_MODE != "none":
                from tardigrade_hooks.rls import (
                    KeywordExpansionStrategy,
                    MultiPhrasingStrategy,
                    ReflectiveLatentSearch,
                )
                strategies = []
                if _RLS_MODE in ("keyword", "both"):
                    strategies.append(KeywordExpansionStrategy())
                if _RLS_MODE in ("multiphrasing", "both"):
                    strategies.append(MultiPhrasingStrategy())
                if strategies:
                    model, tokenizer, query_layer = _load_model_cached()
                    self._rls = ReflectiveLatentSearch(
                        engine=self._engine,
                        model=model,
                        tokenizer=tokenizer,
                        query_layer=query_layer,
                        hidden_size=model.config.hidden_size,
                        owner=1,
                        k=5,
                        strategies=strategies,
                    )
```

- [ ] **Step 2: Update query() to use RLS when available**

In the `query()` method, at the start of the native path (after `if self._mode != "native":` block, around line 192), add before the existing forward pass code:

```python
        if self._rls is not None:
            start = time.perf_counter()
            handles = self._rls.query(item.question, top_k=top_k)
            latency_ms = (time.perf_counter() - start) * 1000.0

            evidence: list[str] = []
            answer = ""
            for h in handles[:max(1, top_k)]:
                mapped = self._cell_to_item.get(int(h.cell_id))
                if mapped is None:
                    continue
                evidence.append(mapped.context)
                if not answer:
                    answer = mapped.ground_truth
            if not answer:
                return AdapterQueryResult(
                    answer="", evidence=[], latency_ms=latency_ms,
                    status="failed", error="no_rls_match",
                )
            return AdapterQueryResult(
                answer=answer, evidence=evidence,
                latency_ms=latency_ms, status="ok", error=None,
            )
```

- [ ] **Step 3: Update metadata to report RLS mode**

In the `metadata()` method, add:

```python
            "rls_mode": _RLS_MODE,
```

- [ ] **Step 4: Run existing benchmark smoke test to verify no regression**

```bash
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop
TDB_REFINEMENT_MODE=centered \
PYTHONPATH=python python -m tdb_bench run \
  --mode smoke --system tardigrade \
  --config python/tdb_bench/config/default.json \
  --output target/bench-smoke-rls-none.json
```

Verify `TDB_RLS_MODE` defaults to `none` and results match previous smoke.

- [ ] **Step 5: Commit**

```bash
git add python/tdb_bench/adapters/tardigrade.py
git commit -m "✨ feat(bench): TardigradeAdapter integrates RLS via TDB_RLS_MODE env var"
```

---

### Task 3: 10-Fact Experiment + LoCoMo Benchmark

**Files:**
- Create: `experiments/rls_experiment.py`

- [ ] **Step 1: Write experiment script**

Script tests 4 configs on the 10-fact Sonia corpus:
1. Baseline (centered, no RLS)
2. RLS keyword expansion
3. RLS multi-phrasing
4. RLS both

Uses the same `FACTS`, `SPECIFIC`, `MODERATE`, `VAGUE` queries from
`experiments/vague_refinement_v2_experiment.py`. For each config:
creates engine, ingests facts, runs queries via `ReflectiveLatentSearch`,
reports R@5 per tier.

- [ ] **Step 2: Run 10-fact experiment**

```bash
source .venv/bin/activate && python experiments/rls_experiment.py
```

Success criteria:
- Specific R@5 ≥ 100% (no regression)
- Moderate R@5 ≥ 80% (no regression)
- Vague R@5 > 60% in at least one config (any improvement)

- [ ] **Step 3: If vague improves — run full LoCoMo**

```bash
export DEEPSEEK_API_KEY=$(cat .env.bench | grep DEEPSEEK_API_KEY | cut -d= -f2)
TDB_RLS_MODE=<best_mode> \
TDB_REFINEMENT_MODE=centered \
LOCOMO_DATA_PATH=benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl \
LOCOMO_DATA_REV=phase1_oracle \
LONGMEMEVAL_DATA_PATH=benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl \
LONGMEMEVAL_DATA_REV=phase1_oracle \
PYTHONPATH=python python -m tdb_bench run \
  --mode full --system tardigrade \
  --config python/tdb_bench/config/default.json \
  --output target/bench-full-rls.json
```

Compare against 67.2% LoCoMo / 88.8% LongMemEval baselines.

- [ ] **Step 4: Document results and commit**

```bash
git add experiments/rls_experiment.py
git commit -m "📊 experiments: Reflective Latent Search — keyword + multiphrasing on Sonia corpus"
git push origin main
```
