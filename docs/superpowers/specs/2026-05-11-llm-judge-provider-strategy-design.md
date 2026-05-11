# LLM Judge Provider Strategy: Multi-Provider Evaluator for Benchmarks

## Problem

The benchmark evaluator (`LLMGatedEvaluator`) has two issues:
1. **Hardcoded provider logic** — OpenAI Responses API and DeepSeek are baked into a single class, violating OCP. Adding a new provider means editing the evaluator.
2. **Broken OpenAI path** — Uses the non-standard Responses API (`/v1/responses`) which silently falls back to deterministic evaluation, making our LoCoMo 68.2% / LongMemEval 90.9% baseline potentially underscored.

The user wants to re-run benchmarks with DeepSeek as the LLM judge to get fair scores comparable to published competitor numbers. This requires a working LLM evaluator.

## Design

### Pattern: Strategy (Provider) + Chain of Responsibility (Evaluator)

**`JudgeProvider`** (Strategy ABC): encapsulates how to call a specific LLM API.
**`LLMGatedEvaluator`** (Chain of Responsibility): iterates providers until one succeeds, falls back to deterministic.

This follows the same Strategy pattern used throughout TardigradeDB: `RetrievalKeyStrategy`, `FramingStrategy`, `BoundaryStrategy`, `ConsolidationPolicy`.

### ATDD Acceptance Criteria

**Before any implementation, these tests define "done":**

#### Provider-level:
1. `DeepSeekProvider.is_available()` returns `True` when `DEEPSEEK_API_KEY` env var is set, `False` when unset.
2. `OpenAIProvider.is_available()` returns `True` when `OPENAI_API_KEY` env var is set, `False` when unset.
3. `DeepSeekProvider.name()` returns `"deepseek"`.
4. `OpenAIProvider.name()` returns `"openai"`.
5. A provider that is not available raises `ValueError` from `judge()` (no new exception class — YAGNI).

#### Evaluator chain-level:
6. With no available providers, evaluator falls back to deterministic: `evaluator_mode == "deterministic_fallback"`.
7. With DeepSeek available, evaluator uses it: `evaluator_mode == "llm_deepseek"`.
8. With first provider failing (exception), evaluator tries next provider.
9. Score parsing: `'{"score": 0.85}'` parses to `0.85`; malformed JSON defaults to `0.0`.
10. Evaluator receives providers via constructor injection (DIP).

#### Registry/config-level:
11. `"mode": "llm_gated"` in config still works (backward compatible).
12. Provider list is constructed by the registry based on available env vars.

#### Integration:
13. `tdb_bench run --mode smoke` with `DEEPSEEK_API_KEY` set produces `evaluator_mode` containing `"deepseek"` in the output JSON.

### Components

#### 1. `JudgeProvider` ABC (`python/tdb_bench/evaluators/providers.py`)

```python
class JudgeProvider(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def judge(self, prompt: str) -> str: ...
```

SRP: each provider handles one API's HTTP call and response parsing. Nothing else.

#### 2. `DeepSeekProvider` (`python/tdb_bench/evaluators/providers.py`)

- Env var: `DEEPSEEK_API_KEY`
- Endpoint: `https://api.deepseek.com/v1/chat/completions`
- Model: `deepseek-chat`
- Response: `choices[0].message.content`
- Timeout: 15s

#### 3. `OpenAIProvider` (`python/tdb_bench/evaluators/providers.py`)

- Env var: `OPENAI_API_KEY`
- Endpoint: `https://api.openai.com/v1/chat/completions` (Chat Completions, NOT Responses API)
- Model: configurable, default `gpt-4.1-mini`
- Response: `choices[0].message.content` (same format as DeepSeek)
- Timeout: 10s

Both providers use the identical Chat Completions format. The only differences: env var, base URL, default model, timeout.

#### 4. Refactored `LLMGatedEvaluator` (`python/tdb_bench/evaluators/llm.py`)

```python
class LLMGatedEvaluator(Evaluator):
    def __init__(self, providers: list[JudgeProvider]) -> None:
        self._providers = providers
        self._fallback = DeterministicEvaluator()

    def score(self, item, answer, evidence) -> ScoreResult:
        prompt = _JUDGE_PROMPT.format(...)
        for provider in self._providers:
            if not provider.is_available():
                continue
            try:
                text = provider.judge(prompt)
                score = _parse_score(text)
                verdict = "pass" if score >= 0.8 else "fail"
                return ScoreResult(
                    score=score,
                    judgment=f"llm_{verdict}",
                    evaluator_mode=f"llm_{provider.name()}",
                )
            except Exception:
                continue
        # All providers failed
        fallback = self._fallback.score(item, answer, evidence)
        return ScoreResult(
            score=fallback.score,
            judgment=f"{fallback.judgment}_fallback",
            evaluator_mode="deterministic_fallback",
        )
```

DIP: evaluator depends on `JudgeProvider` abstraction, not concrete providers.
Chain of Responsibility: try each provider in order, first success wins.

#### 5. `_parse_score` pure function (`python/tdb_bench/evaluators/llm.py`)

```python
def _parse_score(text: str) -> float:
    """Extract score from LLM judge response. Returns 0.0 on parse failure."""
```

Extracts `{"score": N}` from response text, handling JSON embedded in markdown code fences or surrounded by explanation text.

#### 6. Registry update (`python/tdb_bench/registry.py`)

For `"llm_gated"` mode, construct the provider chain:

```python
providers = []
providers.append(DeepSeekProvider())
providers.append(OpenAIProvider(model=judge_model))
return LLMGatedEvaluator(providers=providers)
```

Config stays backward-compatible. No new config keys required — provider availability is determined by env vars at runtime.

### SOLID Analysis

- **SRP:** Provider handles HTTP. Evaluator handles scoring logic. `_parse_score` handles JSON extraction.
- **OCP:** New providers (Anthropic, Ollama) added by creating a class + registering in the provider list.
- **LSP:** All providers implement `JudgeProvider` — substitutable.
- **ISP:** `JudgeProvider` has 3 methods: `name()`, `is_available()`, `judge()`. Minimal.
- **DIP:** `LLMGatedEvaluator` depends on `JudgeProvider` abstraction, not concrete classes.

### Files

| File | Action | What |
|------|--------|------|
| `python/tdb_bench/evaluators/providers.py` | Create | `JudgeProvider` ABC, `DeepSeekProvider`, `OpenAIProvider` |
| `python/tdb_bench/evaluators/llm.py` | Refactor | Inject providers, remove hardcoded API logic, extract `_parse_score` |
| `python/tdb_bench/evaluators/__init__.py` | Update | Export new classes |
| `python/tdb_bench/registry.py` | Update | Build provider chain in factory |
| `tests/python/test_evaluator_providers.py` | Create | Provider ATDD tests |
| `tests/python/test_evaluator_llm.py` | Create | Evaluator chain ATDD tests |

### What Does NOT Change

- `DeterministicEvaluator` (untouched, still the fallback)
- `BenchmarkRunner` (calls `evaluator.score()` — interface unchanged)
- Config JSON structure (backward compatible)
- CLI (`tdb_bench run` commands)
- Result JSON schema (`score`, `judgment`, `evaluator_mode` fields)

### Verification

1. `pytest tests/python/test_evaluator_providers.py tests/python/test_evaluator_llm.py -v` — all ATDD tests pass
2. `DEEPSEEK_API_KEY=sk-... tdb_bench run --mode smoke --system tardigrade` — `evaluator_mode` is `"llm_deepseek"` in output JSON
3. Full LoCoMo + LongMemEval re-run with DeepSeek judge — compare against 68.2% / 90.9% deterministic baseline
