# Path 1: Synthetic-Fact KV Injection Verification

**Date:** April 27, 2026  
**Model:** Qwen3-0.6B (596M, 28 layers, 8 KV heads, head_dim=128) on CPU, float32  
**Script:** `experiments/synthetic_kv_injection_experiment.py`  
**Corpus:** `experiments/synthetic_facts_corpus.py`  
**Tests:** `tests/python/test_synthetic_kv_injection.py` (7 ATDD tests)  
**Status:** Complete — **PASS**

## The Question

Does KV injection actually transfer knowledge the model has never seen? Or does it only appear to work because the model can guess facts from training data?

Earlier experiments reported 8/10 novel facts recalled byte-identically to text RAG (Phase 25, `injection_vs_text_rag.py`; Phase 30, `multi_memory_experiment.py`). But those facts used real-world words — "Pachycephalosaurus", "Pachinko", "Fernando" — that a model could plausibly pattern-match from training data. This experiment removes that confound by using **fully synthetic gibberish** as the answer targets.

## Corpus Design

10 facts where every expected answer is pure nonsense — nonsense proper nouns, fake units, made-up numbers with invented words. No training corpus contains these strings.

| # | Fact | Query | Expected |
|---|------|-------|----------|
| 1 | The capital of Vrenthar is Zyphlox-9 | What is the capital of Vrenthar? | Zyphlox-9 |
| 2 | The qurblix density of compound 7T is 42.7 zennits | What is the qurblix density of compound 7T? | 42.7 zennits |
| 3 | Dr. Molvax discovered the Krellian frequency at 8.31 plonks | What is the Krellian frequency discovered by Dr. Molvax? | 8.31 plonks |
| 4 | The planet Gorflax-12 orbits the star Wumbelion every 347 drazeks | How often does Gorflax-12 orbit Wumbelion? | 347 drazeks |
| 5 | Agent Snibblex reported that the vault code is 9-Quornth-44 | What is the vault code reported by Agent Snibblex? | 9-Quornth-44 |
| 6 | The Blirvian treaty was signed in the year 7042 by Chancellor Prindok-3 | Who signed the Blirvian treaty and when? | Prindok-3 |
| 7 | The tallest building in Skorblex City is the Junthavex-7 Tower at 1.3 thrummels | What is the tallest building in Skorblex City? | Junthavex-7 Tower |
| 8 | Professor Glindavar invented the Thraxial engine using 5 klombs of purazine | What did Professor Glindavar use to build the Thraxial engine? | 5 klombs of purazine |
| 9 | The speed record on the Nelvox track is 88.2 frenzils set by racer Dwimtho-6 | What is the speed record on the Nelvox track? | 88.2 frenzils |
| 10 | The antidote for Crellish fever requires 3 drops of Yombliquid-X per dose | What is the antidote dosage for Crellish fever? | 3 drops of Yombliquid-X |

Design constraints validated by ATDD tests:
- Every expected answer is unique across the corpus
- Every answer is >= 4 characters and contains non-alpha characters (digits, hyphens, spaces)
- No answer is a single token in the tokenizer vocabulary (multi-token generation required)

## Method

Two paths compared head-to-head on identical facts:

**Text RAG:** Fact placed in a system message via chat template, query as user message, greedy generation with `max_new_tokens=100`. Thinking suppressed via `enable_thinking=False` where supported. `</think>` tags stripped from output.

**KV Injection:** Fact stored via `KnowledgePackStore.store(fact)` in a fresh engine (one per fact for clean isolation). Query sent through `KnowledgePackStore.generate(query + " /no_think")` which retrieves the stored KV pack and injects it as `past_key_values` into `model.generate()`.

Fresh engine per fact eliminates retrieval confusion — with exactly one stored memory, retrieval accuracy is 100%. This isolates the question to: **does the injected KV transfer knowledge?**

## Results

| # | Expected | RAG | KV | Tokens Saved |
|---|----------|-----|-----|-------------|
| 1 | Zyphlox-9 | Y | Y | 16 |
| 2 | 42.7 zennits | Y | Y | 22 |
| 3 | 8.31 plonks | Y | Y | 23 |
| 4 | 347 drazeks | Y | Y | 26 |
| 5 | 9-Quornth-44 | Y | Y | 22 |
| 6 | Prindok-3 | Y | Y | 26 |
| 7 | Junthavex-7 Tower | Y | Y | 29 |
| 8 | 5 klombs of purazine | Y | **N** | 21 |
| 9 | 88.2 frenzils | **N** | Y | 29 |
| 10 | 3 drops of Yombliquid-X | Y | Y | 22 |

**Text RAG: 9/10 | KV Injection: 9/10 | Recall ratio: 100% | Token savings: 236**

### Analysis of misses

**#8 (KV miss):** KV injection returned "Professor Glindavar used **purazine** to build the Thraxial engine." — correct substance but dropped the quantity "5 klombs of". The model extracted the key entity but lost the numeric detail. The compound expected answer ("5 klombs of purazine") is the longest in the corpus.

**#9 (RAG miss):** Text RAG returned only "The speed record" before hitting the `max_new_tokens` limit. The model entered `<think>` mode despite `enable_thinking=False` and consumed all 100 tokens on reasoning before producing the answer. KV injection avoided this because `KnowledgePackStore.generate()` appends `/no_think` to the query, and the injected KV bypasses the system message where thinking is triggered.

### Sample KV injection responses

```
[1] The capital of Vrenthar is **Zyphlox-9**.
[4] Gorflax-12 orbits Wumbelion every **347 drazeks**.
[5] The vault code reported by Agent Snibblex is **9-Quornth-44**.
[6] The Blirvian Treaty was signed by **Chancellor Prindok-3** on **7042**.
[10] The antidote for Crellish fever requires **3 drops of Yombliquid-X per dose**.
```

These responses contain gibberish strings that exist nowhere in training data. The only source is the injected KV cache tensors.

## Implications

**The core thesis holds.** KV injection via `model.generate(past_key_values=...)` transfers knowledge the model has provably never seen. This is not pattern-matching from training data — "Zyphlox-9", "9-Quornth-44", and "Yombliquid-X" are pure nonsense strings that can only come from the injected tensors.

**Token economics.** KV injection saves ~23.6 prompt tokens per query on average. At scale (thousands of queries against the same stored memories), the cumulative prefill savings are significant.

**KnowledgePackStore is the canonical path.** This experiment validates the full pipeline: chat-template wrapping → model forward pass → per-layer KV extraction → Q4-quantized storage → retrieval → DynamicCache reconstruction → `past_key_values` injection. All through the `KnowledgePackStore` facade.

## Test Coverage

| Test | What it validates |
|------|------------------|
| `test_corpus_answers_are_unique_gibberish` | Corpus integrity: unique, >= 4 chars, non-alpha markers |
| `test_corpus_answers_not_single_token` | No answer is a single token (multi-token generation required) |
| `test_kv_store_and_retrieve_structural` | Store 10 facts, retrieve_and_inject returns valid cache (GPT-2) |
| `test_kv_injection_returns_with_memory_flag` | `kps.generate()` returns `had_memory=True` (GPT-2) |
| `test_text_rag_baseline_produces_output` | Text RAG helper produces non-empty output (GPT-2) |
| `test_experiment_result_schema` | `run_experiment()` returns dict with required keys |
| `test_injection_recall_gate` | **THE GATE**: `kv_correct / max(rag_correct, 1) >= 0.70` (Qwen3-0.6B, marked `@pytest.mark.slow`) |

Tests 1-6 run on GPT-2 (fast, CI-safe). Test 7 requires Qwen3-0.6B (~3 min on CPU).

## Running

```bash
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -m crates/tdb-python/Cargo.toml

# Full experiment with output
python experiments/synthetic_kv_injection_experiment.py

# Structural tests (fast)
pytest tests/python/test_synthetic_kv_injection.py -v -k "not slow"

# Gate test (requires Qwen3-0.6B)
pytest tests/python/test_synthetic_kv_injection.py -v -m slow
```
