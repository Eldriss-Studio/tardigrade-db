# LLM Judge Provider Strategy Implementation Plan

> **⚠️ Note — 2026-05-14.** Premise retracted, implementation stands. This plan was motivated by validating the LoCoMo 68.2% baseline with a real LLM judge; that baseline measured the lexical fallback, not the native engine. The `JudgeProvider` Strategy / Chain-of-Responsibility infrastructure (DeepSeek + OpenAI providers, deterministic fallback) is intact and useful. See [`../../experiments/2026-05-14-bench-audit.md`](../../experiments/2026-05-14-bench-audit.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded evaluator with a Strategy-based provider chain so TardigradeDB benchmarks can use DeepSeek (or any LLM) as an impartial judge.

**Architecture:** `JudgeProvider` ABC (Strategy) with `DeepSeekProvider` and `OpenAIProvider`. `LLMGatedEvaluator` receives providers via constructor injection and tries them in order (Chain of Responsibility). Deterministic fallback as last resort.

**Tech Stack:** Python, `urllib.request` (zero deps), Chat Completions API (standard across DeepSeek + OpenAI).

---

### Task 1: JudgeProvider ABC + DeepSeekProvider + OpenAIProvider

**Files:**
- Create: `python/tdb_bench/evaluators/providers.py`
- Create: `tests/python/test_evaluator_providers.py`

- [ ] **Step 1: Write the ATDD acceptance tests**

Create `tests/python/test_evaluator_providers.py`:

```python
"""ATDD tests for JudgeProvider — Strategy pattern for LLM judge APIs."""

import os
from unittest.mock import patch

import pytest

from tdb_bench.evaluators.providers import (
    DeepSeekProvider,
    JudgeProvider,
    OpenAIProvider,
)


class TestJudgeProviderABC:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            JudgeProvider()

    def test_deepseek_is_substitutable(self):
        assert isinstance(DeepSeekProvider(), JudgeProvider)

    def test_openai_is_substitutable(self):
        assert isinstance(OpenAIProvider(), JudgeProvider)


class TestDeepSeekProvider:
    def test_name(self):
        assert DeepSeekProvider().name() == "deepseek"

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"})
    def test_available_when_key_set(self):
        assert DeepSeekProvider().is_available()

    @patch.dict(os.environ, {}, clear=True)
    def test_unavailable_when_key_missing(self):
        p = DeepSeekProvider()
        # Remove key if it exists in the patched env
        os.environ.pop("DEEPSEEK_API_KEY", None)
        assert not p.is_available()

    def test_judge_raises_when_unavailable(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DEEPSEEK_API_KEY", None)
            p = DeepSeekProvider()
            with pytest.raises(ValueError, match="not available"):
                p.judge("test prompt")


class TestOpenAIProvider:
    def test_name(self):
        assert OpenAIProvider().name() == "openai"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_available_when_key_set(self):
        assert OpenAIProvider().is_available()

    @patch.dict(os.environ, {}, clear=True)
    def test_unavailable_when_key_missing(self):
        p = OpenAIProvider()
        os.environ.pop("OPENAI_API_KEY", None)
        assert not p.is_available()

    def test_judge_raises_when_unavailable(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            p = OpenAIProvider()
            with pytest.raises(ValueError, match="not available"):
                p.judge("test prompt")

    def test_custom_model(self):
        p = OpenAIProvider(model="gpt-4o")
        assert p.name() == "openai"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && PYTHONPATH=python pytest tests/python/test_evaluator_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tdb_bench.evaluators.providers'`

- [ ] **Step 3: Implement providers.py**

Create `python/tdb_bench/evaluators/providers.py`:

```python
"""LLM judge providers — Strategy pattern for multi-provider evaluation.

Each provider encapsulates one LLM API's HTTP call. The evaluator
iterates providers via Chain of Responsibility until one succeeds.
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod


class JudgeProvider(ABC):
    """Strategy: sends a judge prompt to an LLM API, returns raw text."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def judge(self, prompt: str) -> str: ...


class DeepSeekProvider(JudgeProvider):
    """DeepSeek Chat Completions API."""

    _URL = "https://api.deepseek.com/v1/chat/completions"
    _MODEL = "deepseek-chat"
    _ENV_VAR = "DEEPSEEK_API_KEY"
    _TIMEOUT = 15

    def name(self) -> str:
        return "deepseek"

    def is_available(self) -> bool:
        return bool(os.getenv(self._ENV_VAR, "").strip())

    def judge(self, prompt: str) -> str:
        api_key = os.getenv(self._ENV_VAR, "").strip()
        if not api_key:
            raise ValueError(f"{self.name()} provider not available: {self._ENV_VAR} not set")
        return _chat_completions(self._URL, api_key, self._MODEL, prompt, self._TIMEOUT)


class OpenAIProvider(JudgeProvider):
    """OpenAI Chat Completions API."""

    _URL = "https://api.openai.com/v1/chat/completions"
    _ENV_VAR = "OPENAI_API_KEY"
    _TIMEOUT = 10

    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self._model = model

    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(os.getenv(self._ENV_VAR, "").strip())

    def judge(self, prompt: str) -> str:
        api_key = os.getenv(self._ENV_VAR, "").strip()
        if not api_key:
            raise ValueError(f"{self.name()} provider not available: {self._ENV_VAR} not set")
        return _chat_completions(self._URL, api_key, self._model, prompt, self._TIMEOUT)


def _chat_completions(url: str, api_key: str, model: str, prompt: str, timeout: int) -> str:
    """Shared Chat Completions call — same format for DeepSeek and OpenAI."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && PYTHONPATH=python pytest tests/python/test_evaluator_providers.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tdb_bench/evaluators/providers.py tests/python/test_evaluator_providers.py
git commit -m "✨ feat(bench): JudgeProvider ABC + DeepSeek/OpenAI providers (Strategy pattern)"
```

---

### Task 2: Refactor LLMGatedEvaluator to Use Providers

**Files:**
- Modify: `python/tdb_bench/evaluators/llm.py`
- Create: `tests/python/test_evaluator_llm.py`

- [ ] **Step 1: Write the ATDD acceptance tests**

Create `tests/python/test_evaluator_llm.py`:

```python
"""ATDD tests for LLMGatedEvaluator — Chain of Responsibility over providers."""

from unittest.mock import MagicMock

import pytest

from tdb_bench.evaluators.llm import LLMGatedEvaluator, _parse_score
from tdb_bench.evaluators.providers import JudgeProvider
from tdb_bench.models import BenchmarkItem


def _make_item():
    return BenchmarkItem(
        item_id="test-1",
        dataset="test",
        context="Test context.",
        question="What color is the sky?",
        ground_truth="blue",
    )


def _mock_provider(name: str, available: bool, response: str | None = None, error: Exception | None = None):
    p = MagicMock(spec=JudgeProvider)
    p.name.return_value = name
    p.is_available.return_value = available
    if error:
        p.judge.side_effect = error
    elif response is not None:
        p.judge.return_value = response
    return p


class TestChainOfResponsibility:
    def test_uses_first_available_provider(self):
        p1 = _mock_provider("alpha", available=True, response='{"score": 0.9}')
        p2 = _mock_provider("beta", available=True, response='{"score": 0.5}')
        ev = LLMGatedEvaluator(providers=[p1, p2])
        result = ev.score(_make_item(), "blue", [])
        assert result.evaluator_mode == "llm_alpha"
        assert result.score == 0.9

    def test_skips_unavailable_provider(self):
        p1 = _mock_provider("alpha", available=False)
        p2 = _mock_provider("beta", available=True, response='{"score": 0.8}')
        ev = LLMGatedEvaluator(providers=[p1, p2])
        result = ev.score(_make_item(), "blue", [])
        assert result.evaluator_mode == "llm_beta"
        p1.judge.assert_not_called()

    def test_falls_through_on_provider_error(self):
        p1 = _mock_provider("alpha", available=True, error=ConnectionError("timeout"))
        p2 = _mock_provider("beta", available=True, response='{"score": 0.7}')
        ev = LLMGatedEvaluator(providers=[p1, p2])
        result = ev.score(_make_item(), "blue", [])
        assert result.evaluator_mode == "llm_beta"

    def test_deterministic_fallback_when_all_fail(self):
        p1 = _mock_provider("alpha", available=True, error=ConnectionError("down"))
        ev = LLMGatedEvaluator(providers=[p1])
        result = ev.score(_make_item(), "blue", [])
        assert result.evaluator_mode == "deterministic_fallback"

    def test_deterministic_fallback_when_no_providers(self):
        ev = LLMGatedEvaluator(providers=[])
        result = ev.score(_make_item(), "blue", [])
        assert result.evaluator_mode == "deterministic_fallback"

    def test_judgment_pass_above_threshold(self):
        p = _mock_provider("alpha", available=True, response='{"score": 0.85}')
        ev = LLMGatedEvaluator(providers=[p])
        result = ev.score(_make_item(), "blue", [])
        assert result.judgment == "llm_pass"

    def test_judgment_fail_below_threshold(self):
        p = _mock_provider("alpha", available=True, response='{"score": 0.3}')
        ev = LLMGatedEvaluator(providers=[p])
        result = ev.score(_make_item(), "blue", [])
        assert result.judgment == "llm_fail"


class TestParseScore:
    def test_valid_json(self):
        assert _parse_score('{"score": 0.85}') == 0.85

    def test_json_in_markdown_code_fence(self):
        assert _parse_score('```json\n{"score": 0.9}\n```') == 0.9

    def test_json_with_surrounding_text(self):
        assert _parse_score('The answer is correct. {"score": 1.0}') == 1.0

    def test_malformed_json_returns_zero(self):
        assert _parse_score("not json at all") == 0.0

    def test_empty_string_returns_zero(self):
        assert _parse_score("") == 0.0

    def test_clamps_to_range(self):
        assert _parse_score('{"score": 1.5}') == 1.0
        assert _parse_score('{"score": -0.3}') == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && PYTHONPATH=python pytest tests/python/test_evaluator_llm.py -v`
Expected: FAIL — `LLMGatedEvaluator` constructor signature mismatch (expects `answerer_model, judge_model`, not `providers`).

- [ ] **Step 3: Rewrite llm.py**

Replace `python/tdb_bench/evaluators/llm.py` entirely:

```python
"""LLM-gated evaluator with provider chain and deterministic fallback.

Chain of Responsibility: iterates JudgeProvider instances until one
succeeds. Falls back to DeterministicEvaluator if all providers fail
or none are available.
"""

from __future__ import annotations

import json
import re

from tdb_bench.contracts import Evaluator
from tdb_bench.models import BenchmarkItem, ScoreResult

from .deterministic import DeterministicEvaluator
from .providers import JudgeProvider

_JUDGE_PROMPT = (
    'Score answer correctness from 0.0 to 1.0 as JSON {{"score": number}}.\n'
    "Question: {question}\n"
    "Ground truth: {ground_truth}\n"
    "Answer: {answer}"
)

_PASS_THRESHOLD = 0.8


class LLMGatedEvaluator(Evaluator):
    """Chain of Responsibility over LLM judge providers.

    Tries each provider in order. First successful response wins.
    Falls back to deterministic scoring if all providers fail.
    """

    def __init__(self, providers: list[JudgeProvider]) -> None:
        self._providers = providers
        self._fallback = DeterministicEvaluator()

    def score(self, item: BenchmarkItem, answer: str, evidence: list[str]) -> ScoreResult:
        prompt = _JUDGE_PROMPT.format(
            question=item.question,
            ground_truth=item.ground_truth,
            answer=answer,
        )

        for provider in self._providers:
            if not provider.is_available():
                continue
            try:
                text = provider.judge(prompt)
                parsed = _parse_score(text)
                verdict = "pass" if parsed >= _PASS_THRESHOLD else "fail"
                return ScoreResult(
                    score=parsed,
                    judgment=f"llm_{verdict}",
                    evaluator_mode=f"llm_{provider.name()}",
                )
            except Exception:
                continue

        fallback = self._fallback.score(item, answer, evidence)
        return ScoreResult(
            score=fallback.score,
            judgment=f"{fallback.judgment}_fallback",
            evaluator_mode="deterministic_fallback",
        )


def _parse_score(text: str) -> float:
    """Extract score from LLM judge response. Returns 0.0 on parse failure."""
    if not text:
        return 0.0
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text)
    match = re.search(r'\{\s*"score"\s*:\s*([0-9]*\.?[0-9]+)\s*\}', cleaned)
    if not match:
        return 0.0
    score = float(match.group(1))
    return max(0.0, min(1.0, round(score, 6)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && PYTHONPATH=python pytest tests/python/test_evaluator_llm.py -v`
Expected: All 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tdb_bench/evaluators/llm.py tests/python/test_evaluator_llm.py
git commit -m "♻️ refactor(bench): LLMGatedEvaluator uses provider chain (Chain of Responsibility)"
```

---

### Task 3: Registry + Exports Update

**Files:**
- Modify: `python/tdb_bench/evaluators/__init__.py`
- Modify: `python/tdb_bench/registry.py`

- [ ] **Step 1: Update __init__.py exports**

Replace `python/tdb_bench/evaluators/__init__.py`:

```python
"""Evaluation strategies."""

from .deterministic import DeterministicEvaluator
from .llm import LLMGatedEvaluator
from .providers import DeepSeekProvider, JudgeProvider, OpenAIProvider

__all__ = [
    "DeterministicEvaluator",
    "LLMGatedEvaluator",
    "JudgeProvider",
    "DeepSeekProvider",
    "OpenAIProvider",
]
```

- [ ] **Step 2: Update registry to build provider chain**

In `python/tdb_bench/registry.py`, update the import and factory method:

Change the import line:
```python
from tdb_bench.evaluators import DeterministicEvaluator, LLMGatedEvaluator, DeepSeekProvider, OpenAIProvider
```

Replace the `create_evaluator` method body:
```python
    @staticmethod
    def create_evaluator(evaluator_cfg: dict) -> Evaluator:
        mode = evaluator_cfg.get("mode", "deterministic")
        judge_model = evaluator_cfg.get("judge_model", "gpt-4.1-mini")

        if mode == "deterministic":
            return DeterministicEvaluator()
        if mode in ("llm", "llm_gated"):
            providers = [
                DeepSeekProvider(),
                OpenAIProvider(model=judge_model),
            ]
            return LLMGatedEvaluator(providers=providers)
        raise ConfigError(f"Unknown evaluator mode: {mode}")
```

- [ ] **Step 3: Run all evaluator tests**

Run: `source .venv/bin/activate && PYTHONPATH=python pytest tests/python/test_evaluator_providers.py tests/python/test_evaluator_llm.py -v`
Expected: All 24 tests PASS.

- [ ] **Step 4: Run the full Python test suite to check no regressions**

Run: `source .venv/bin/activate && pytest tests/python/ -m "not gpu" --tb=short -q`
Expected: Same pass/fail counts as before (4 pre-existing failures in `test_vllm_load_path.py`).

- [ ] **Step 5: Commit**

```bash
git add python/tdb_bench/evaluators/__init__.py python/tdb_bench/registry.py
git commit -m "🔧 chore(bench): registry builds provider chain for llm_gated evaluator"
```

---

### Task 4: Smoke Integration Test + Push

**Files:**
- No new files — run existing smoke benchmark with DeepSeek key

- [ ] **Step 1: Run smoke benchmark with DEEPSEEK_API_KEY**

```bash
source .venv/bin/activate
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
TDB_REFINEMENT_MODE=centered \
PYTHONPATH=python \
python -m tdb_bench run \
  --mode smoke \
  --system tardigrade \
  --config python/tdb_bench/config/default.json \
  --output target/bench-smoke-deepseek.json
```

- [ ] **Step 2: Verify evaluator_mode in output**

```bash
python3 -c "
import json
with open('target/bench-smoke-deepseek.json') as f:
    data = json.load(f)
modes = {item['evaluator_mode'] for item in data['items']}
print('Evaluator modes used:', modes)
assert 'llm_deepseek' in modes, f'Expected llm_deepseek, got {modes}'
print('PASS: DeepSeek provider used successfully')
"
```

Expected: `Evaluator modes used: {'llm_deepseek'}` and `PASS`.

- [ ] **Step 3: Push all commits**

```bash
git push origin main
```

---

### Task 5: Full LoCoMo + LongMemEval Re-run with DeepSeek Judge

**Files:**
- Create: `docs/bench/locomo-longmemeval-llm-judged.md` (results)

- [ ] **Step 1: Run full benchmark**

```bash
source .venv/bin/activate
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
LOCOMO_DATA_PATH=benchmarks/datasets/phase1_oracle/locomo_phase1.jsonl \
LOCOMO_DATA_REV=phase1_oracle \
LONGMEMEVAL_DATA_PATH=benchmarks/datasets/phase1_oracle/longmemeval_phase1.jsonl \
LONGMEMEVAL_DATA_REV=phase1_oracle \
TDB_REFINEMENT_MODE=centered \
PYTHONPATH=python \
python -m tdb_bench run \
  --mode full \
  --system tardigrade \
  --config python/tdb_bench/config/default.json \
  --output target/bench-full-deepseek.json
```

- [ ] **Step 2: Extract scores and compare**

```bash
python3 -c "
import json
with open('target/bench-full-deepseek.json') as f:
    data = json.load(f)
items = data['items']
locomo = [i for i in items if i['dataset'] == 'locomo' and i['status'] == 'ok']
longmem = [i for i in items if i['dataset'] == 'longmemeval' and i['status'] == 'ok']
loco_score = sum(i['score'] for i in locomo) / len(locomo) if locomo else 0
lm_score = sum(i['score'] for i in longmem) / len(longmem) if longmem else 0
print(f'LoCoMo (LLM-judged):     {loco_score:.1%} ({len(locomo)} items)')
print(f'LongMemEval (LLM-judged): {lm_score:.1%} ({len(longmem)} items)')
print(f'LoCoMo (deterministic):   68.2%')
print(f'LongMemEval (deterministic): 90.9%')
print(f'LoCoMo delta:  {loco_score - 0.682:+.1%}')
print(f'LongMemEval delta: {lm_score - 0.909:+.1%}')
"
```

- [ ] **Step 3: Write results doc**

Create `docs/bench/locomo-longmemeval-llm-judged.md` with the comparison table: deterministic vs LLM-judged, per-benchmark, plus evaluator metadata.

- [ ] **Step 4: Commit and push**

```bash
git add docs/bench/locomo-longmemeval-llm-judged.md
git commit -m "📊 bench: LoCoMo + LongMemEval with DeepSeek LLM judge — fair comparison"
git push origin main
```
