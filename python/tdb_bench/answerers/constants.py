"""Named constants for the LLM-answerer subsystem.

No magic values: every literal that downstream behavior depends on
lives here with a sentence on why it's the chosen value. Env-var
overrides are documented next to the constant they shadow.
"""

from __future__ import annotations

# --- Provider routing ---

# Default answerer provider name. DeepSeek picked because (a) we
# already have a key in `.env.bench`, (b) deepseek-chat is ~$0.57/full
# LoCoMo run vs Mem0's gpt-4o-mini at ~$0.31 — same order of magnitude,
# strictly cheaper than Claude Haiku 4.5 at $2.25, (c) the bench
# already has a JudgeProvider for DeepSeek so the HTTP/JSON shape is
# proven on this stack.
LLM_GATE_DEFAULT_PROVIDER = "deepseek"

LLM_GATE_DEFAULT_MODEL = "deepseek-chat"

LLM_GATE_DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
LLM_GATE_OPENAI_URL = "https://api.openai.com/v1/chat/completions"

LLM_GATE_DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
LLM_GATE_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# --- Generation parameters ---

# Max tokens for the *answer*, not the prompt. LoCoMo answers are
# typically <30 tokens; LongMemEval <80. 256 gives 3-8× safety margin
# without inviting verbose hallucinations.
LLM_GATE_MAX_TOKENS = 256

# Deterministic eval — temperature 0 ensures the same retrieval +
# prompt yields the same answer, so re-runs are diff-able and cache
# hits are valid.
LLM_GATE_TEMPERATURE = 0.0

# How many candidates the RetrieveThenReadAdapter (Decorator) asks
# the inner adapter to retrieve, regardless of the bench's
# evaluator-reporting top_k. The Decorator owns this budget privately
# below the fairness-validator's abstraction line: every system still
# receives the bench's symmetric top_k from the runner, but the
# Decorator widens its own internal call so the LLM actually has
# material to answer from.
#
# 25 chosen because measured item-level R@25 ≈ 84% on LoCoMo vs R@5
# ≈ 30% (Phase 1B.2 audit, 2026-05-15). This is the retrieval-recall
# ceiling the LLM can possibly reason over. Bumping further yields
# diminishing returns and pushes prompt size up.
#
# Override per-run via TDB_LLM_GATE_INNER_TOP_K.
LLM_GATE_INNER_TOP_K = 25

# How many of the retrieved chunks reach the LLM prompt. 10 picked
# because (a) LoCoMo questions typically need 1-3 supporting cells
# and 10 gives room for misranking, (b) at ~500 tokens per chunk this
# stays inside DeepSeek's 64K context with comfortable headroom.
#
# Override per-run via TDB_LLM_GATE_PROMPT_TOP_K.
LLM_GATE_PROMPT_TOP_K = 10

# Back-compat alias for one release. Existing callers continue to
# work; new code should import LLM_GATE_PROMPT_TOP_K directly because
# the name unambiguously distinguishes "prompt-evidence budget" from
# "inner-retrieval budget" (LLM_GATE_INNER_TOP_K).
LLM_GATE_EVIDENCE_TOP_K = LLM_GATE_PROMPT_TOP_K

# --- HTTP behavior ---

# Per-request timeout. DeepSeek-chat answers <30 tokens in <2s
# typically; 30s allows for cold start and queue depth without
# silently masking outages.
LLM_GATE_HTTP_TIMEOUT_S = 30

# --- Retry policy ---

LLM_GATE_RETRY_MAX_ATTEMPTS = 3
LLM_GATE_RETRY_INITIAL_DELAY_S = 1.0
LLM_GATE_RETRY_BACKOFF_FACTOR = 2.0

# --- Prompt template ---

# Bump this string when the prompt format changes. Response cache
# keys include it, so old cached entries become invalid automatically.
# Format: "v{N}-{ISO-date}" so version order is reading-order and
# we can git-blame the change date.
PROMPT_TEMPLATE_VERSION = "v2-2026-05-16"

# --- Cache ---

LLM_GATE_CACHE_DIR_ENV = "TDB_LLM_GATE_CACHE_DIR"
