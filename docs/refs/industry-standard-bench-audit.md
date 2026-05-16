# Industry-Standard LoCoMo / LongMemEval Bench Audit

**Date:** 2026-05-16
**Author of this pass:** research audit for TardigradeDB
**Companion docs:** [`docs/experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md), [`docs/positioning/latency_first.md`](../positioning/latency_first.md), `~/.claude/plans/i-want-a-definitive-majestic-bear.md`
**Status:** authoritative for "what does the industry do, and what are we doing differently."

## 1 — Executive verdict

The leaderboard practice for **LoCoMo** has converged on a single
pipeline shape: retrieve → answer (GPT-4o-mini class) → LLM-Judge
(`gpt-4o-mini` with the Mem0 `ACCURACY_PROMPT`, CORRECT/WRONG binary,
`temperature=0`, **Category 5 excluded**, 10 independent runs). The
**LongMemEval** standard is different: retrieve → answer → LLM-Judge
with `gpt-4o-2024-08-06`, `temperature=0`, `max_tokens=10`, per-type
yes/no prompts, with the official protocol setting top-k high enough
to include all sessions (`TOPK=1000` in the paper's reader script).

What we are doing **right:** justify-then-judge with v2 prompt (matches
the field-leading ByteRover / Hindsight pattern; correctly resists
refusal-credit and wrong-evidence-penalty biases); honest retraction
of the inflated 1B.8 number; per-item evidence capping; versioned
prompts; cost-tracked DeepSeek answerer + judge; multi-stage Decorator
chain for the retrieve→answer pipeline.

What we are doing **wrong or non-standard,** ordered by likely impact:

1. **Judge model gap.** We use **DeepSeek-Chat (V3)** as judge; every
   leaderboard system uses `gpt-4o-mini` or `gpt-4o-2024-08-06`. The
   judge's calibration is the single most load-bearing variable in
   leaderboard scores; ours is uncalibrated against the others'.
2. **Single seed.** We report one run; Mem0 / Zep / Memobase report
   **mean ± std over 10 independent runs**. Our 30-item slice has
   ~3 pp single-item swing, our 1533-item full corpus has ~0.2 pp by
   their convention. Single-run claims are statistically weaker.
3. **No Category 5 exclusion.** The LoCoMo paper / Mem0 audit /
   dial481 audit all agree Category 5 (adversarial) is excluded from
   the headline number. Our prep treats all items uniformly. If our
   1533-row corpus includes adversarial items, our score is computed
   on a different question set than every comparator.
4. **No per-category reporting.** Every leaderboard system breaks
   LoCoMo into {single-hop, multi-hop, open-domain, temporal} and
   reports both per-category and overall. Mem0 single-hop = 67%,
   multi-hop = 51%, temporal = 56%. The headline alone is uninformative.
5. **Capture-model gap (load-bearing).** Mem0/Memobase/Letta all use
   GPT-4o-mini class. We use Qwen3-1.7B (challenger). ~25 pp of our
   deficit vs Mem0 is plausibly model-size + training-corpus quality,
   not retrieval quality.
6. **Answerer prompt is too strict.** Our prompt says "If the
   evidence does not contain the answer, respond with: I don't know."
   Mem0's `ANSWER_PROMPT` says "convert relative time references" and
   "prioritize the most recent memory" — i.e. it asks the model to
   *reason over* evidence, not just extract from it.

The honest read: a fair LoCoMo Judge comparison against Mem0 needs
(a) judge = `gpt-4o-mini`, (b) Category 5 excluded, (c) 10-run
mean ± std, (d) per-category table, and ideally (e) answerer =
`gpt-4o-mini` to factor out capture model. Without those, our 42%
number cannot be apples-to-apples with the published 66.88%.

---

## 2 — Per-pillar comparison tables

### Pillar 1 — Answerer + judge configuration

| System | Answerer | Judge | Judge temp | Judge max_tok | Scoring | Runs | Category 5? |
|---|---|---|---|---|---|---|---|
| Mem0 (paper) | `gpt-4o-mini` | `gpt-4o-mini` | 0.0 | (small) | LLM binary CORRECT/WRONG, F1, B1 also reported | 10 | **excluded** |
| Memobase | `gpt-4o` (suggested) | `gpt-4o` | n/a | n/a | LLM binary + F1 + BLEU | n/a | follows snap-research convention |
| Letta | `gpt-4o-mini` | `gpt-4.1` | n/a | n/a | SimpleQA-graded + penalty for extra tool calls | n/a | follows convention |
| MemMachine v0.2 | `gpt-4.1-mini` | `gpt-4o-mini` | n/a | n/a | LLM binary | n/a | follows convention |
| ByteRover 2.0 | Gemini 3 Pro (justifier) | Gemini 3 Flash (judge) | n/a | n/a | Hindsight justifier+judge prompts | n/a | 1,982 questions (full set including Cat 5) |
| ENGRAM | `gpt-4o-mini` | `gpt-4o-mini` | n/a | n/a | LLM binary | 3 (mean ± std) | follows convention |
| LongMemEval official | (config-specified) | `gpt-4o-2024-08-06` | 0 | 10 | yes/no binary, per-type prompts | n/a | n/a |
| **TardigradeDB challenger** | **DeepSeek-Chat (V3) + Qwen3-1.7B capture** | **DeepSeek-Chat (V3)** | **0.0** | **60 (judge) / 256 (answer) / 512 (justify)** | **Justify-then-judge, 0.0-1.0 float** | **1** | **all included** |

Key gaps for us:
- Judge model is DeepSeek; everyone else is OpenAI / Gemini.
- Single seed; convention is 3-10 runs with std.
- 0.0-1.0 float score; the LoCoMo convention is binary {0, 1}.
  Our 0.6 partial-credit window inflates middle scores vs binary.
- Capture model is 1.7B vs leaderboard 4o-mini (~8-70B class).

### Pillar 2 — Official benchmark protocol

| Field | LoCoMo (Maharana et al., ACL 2024, [snap-research/locomo](https://github.com/snap-research/locomo)) | LongMemEval (Wu et al., ICLR 2025, [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)) |
|---|---|---|
| Task | QA + event summarization + multimodal dialog | QA across 5 ability types (info extraction, multi-session reasoning, temporal, knowledge updates, abstention) |
| Conversations | 10 conversations, ~300 turns / 9K tokens each, up to 35 sessions | 500 curated questions over freely-scalable chat histories |
| Default answerer | GPT-4 class (paper); GPT-4o-mini (modern norm) | Long-context LLM (paper) or RAG (memory systems track) |
| Default judge | `gpt-4` initially; `gpt-4o-mini` adopted by Mem0; `gpt-4.1` and `gpt-oss-120b` mentioned | `gpt-4o-2024-08-06`, `temperature=0`, `max_tokens=10` |
| Top-K convention | per-system (paper doesn't fix it); leaderboards use 5-25 | `TOPK=1000` (i.e. all sessions; reader is long-context) |
| Session text | full session JSON; no chunking at corpus level | full session JSON or NL ("recommend JSON") |
| Score metric | F1 + BLEU + LLM-Judge (binary); paper warns F1 misleads ("Alice born March" vs "Alice born July" both score high on F1) | per-type `autoeval_label` yes/no; aggregate across 500 questions |
| Duplicate-context items | The audit doc (Phase 1B.4) found 27.6% share context — the generator LLM disambiguates from the question | non-issue: each question has its own session set |
| Adversarial (Cat 5) | **silently dropped** in published code per [dial481/locomo-audit](https://github.com/dial481/locomo-audit); 446 questions / 22.5% of dataset | the abstention category is **scored**, not skipped |

### Pillar 3 — Independent audits + retraction history

| Source | Finding | Cite |
|---|---|---|
| dial481/locomo-audit | **6.4 %** of LoCoMo gold answers wrong → 93.57 % theoretical ceiling | [dial481/locomo-audit](https://github.com/dial481/locomo-audit) |
| dial481/locomo-audit | LLM judge accepts **62.81 %** of intentionally wrong-but-topical answers | same |
| dial481/locomo-audit | Third-party EverMemOS reproduction: **38.38 %** vs claimed **92.32 %** (53.94 pp gap) | same |
| dial481/locomo-audit | **446 adversarial questions (22.5 %)** silently dropped by published eval code | same |
| dial481/locomo-audit | Token-cost claims show **2.9×** discrepancy (2,298 claimed vs 6,669 actual) | same |
| Zep retraction | Original claim **84 %** → corrected **58.44 % ± 0.20** after Mem0 audit; later self-corrected to **75.14 % ± 0.17** | [getzep/zep-papers#5](https://github.com/getzep/zep-papers/issues/5) |
| Zep retraction (root cause) | Included Cat 5 in numerator but not denominator → **+25.56 pp inflation**; single-run vs others' 10-run protocol; modified system prompt | same |
| TardigradeDB 1B.9 retraction | Judge v1 prompt rewarded "I don't know" answers (1.0) and penalized correct-but-evidence-mismatch (0.0). Re-measured 66 % → 42 % on 30-item smoke after v2 prompt | [`2026-05-14-bench-audit.md` §Phase 1B.9](../experiments/2026-05-14-bench-audit.md) |

These four episodes converge on a single principle: **LoCoMo headline
numbers are dominated by judge prompt + category-inclusion choices,
not by retrieval quality**. The same retrieval can score 38 % or 92 %
depending on judge.

### Pillar 4 — Retrieval-only metrics (BEIR-style alternative)

| System | Metric | Score | Methodology |
|---|---|---|---|
| **ENGRAM (Vectorize)** retrieval-only (per the [snap-research/locomo#38 thread](https://github.com/snap-research/locomo/issues/38)) | R@5 | **93.9 %** (1862/1982 questions) | Session chunks ~6 turns, 1-turn overlap; timestamp prefix; speaker injection; synthetic doc augmentation. The R@K is whether the *evidence turn* makes it into top-K. |
| ENGRAM (same) | R@10 | 95.0 % | same |
| ENGRAM (same) | NDCG@5 | 0.894 | same |
| ENGRAM v0.1.3 paper (2511.12960) | **LLM-Judge** (their primary) | 77.55 % LoCoMo, 71.40 % LongMemEval (~1K tokens) | They moved *away* from retrieval-only for the final paper, explicitly noting "lexical metrics are insensitive to factual inversions". |
| LongMemEval paper | Recall@k + NDCG@k | per indexing strategy | The paper reports retrieval and end-to-end separately. |
| **TardigradeDB (our internal probe)** | item-level R@25 | 83.8 % (LongMemEval, 500 items) | `scripts/probe_rank_diagnostic.py`; chunker fixes + whitening + chunk-text reranker (Phase 1A/B). |
| **TardigradeDB (our internal probe)** | item-level R@5 | ~30 % LongMemEval, ~30 % LoCoMo | same |

A retrieval-only BEIR-style table is a **legitimate alternative
headline**. ENGRAM's R@5=93.9% was their headline in the GitHub issue
thread before they shifted to LLM-Judge. Our item-level R@25=83.8% is
in the same order of magnitude as ENGRAM's R@5=93.9% (different K, so
not directly comparable) — but importantly, it is computed without an
answerer or judge, which means it is not vulnerable to either of the
two judge-bias failure modes that have retracted three different
systems' headline numbers.

---

## 3 — Concrete deltas (us → median leaderboard practice)

Sorted by likely score impact, highest first.

### High impact

| # | Delta | Our practice | Leaderboard median | Fix complexity |
|---|---|---|---|---|
| 1 | **Judge model** | `deepseek-chat` (V3) | `gpt-4o-mini` (5/7 systems) | **trivial** — change `evaluator.judge_model` in `default.json::challenger` and set `OPENAI_API_KEY`. Cost: ~$0.30/full run. |
| 2 | **Capture model size** | Qwen3-1.7B local | GPT-4o-mini class (~8B+) | **hard** — requires either remote-API capture flow (no KV access) or accepting that we're a different class of system. The KV-native architecture *requires* a local model; this is a structural tradeoff, not a fix. |
| 3 | **Category 5 exclusion** | All items scored | Cat 5 excluded (Mem0, Memobase, Letta, ENGRAM) | **trivial** — add `--exclude-category 5` flag to `prepare_phase1_datasets.py` and bench runner aggregation. Document choice in run metadata. |
| 4 | **Multi-seed reporting** | seed=42 single run | mean ± std over 3-10 runs | **medium** — script change in `scripts/run_challenger_headline.sh` to loop; aggregator in `reporting.py` to compute std. Cost multiplies linearly. |
| 5 | **Binary scoring vs partial credit** | 0.0-1.0 float with 0.6-0.9 partial-credit window | binary {0, 1} on LoCoMo Judge | **trivial** — judge prompt change. Note our v2 prompt already documents the partial-credit window as rule 2; remove it to align with Mem0/Memobase headline convention. Keep the float as a secondary diagnostic. |

### Medium impact

| # | Delta | Our practice | Leaderboard median | Fix complexity |
|---|---|---|---|---|
| 6 | **Per-category breakdown reporting** | aggregate `avg_score` only | every system reports per-category (single-hop / multi-hop / temporal / open-domain) | **medium** — requires preserving LoCoMo category labels through prep → bench → reporting. Mem0's per-category table (single-hop 67 %, multi-hop 51 %, temporal 56 %, open-domain 73 %) is what makes their results interpretable. |
| 7 | **Answerer prompt rigidity** | "respond with: I don't know" if evidence insufficient | Mem0's `ANSWER_PROMPT` actively asks model to *resolve* relative time references, prioritize recent memories, ignore character-name vs user disambiguation | **medium** — rewrite `prompt_builder.py::_ANSWER_INSTRUCTION` to match Mem0's style. Bump `PROMPT_TEMPLATE_VERSION`. Risk: trades refusal-rate improvement for hallucination risk; needs A/B measurement. |
| 8 | **Top-K convention for LongMemEval** | profile says `top_k=5`, decorator widens to inner=25, prompt sees 10 | LongMemEval paper's official reader uses `TOPK=1000` (i.e. long-context, all sessions) | **medium** — LongMemEval is *not designed for sparse retrieval*. Memory systems compete on compression to ~1K tokens (ENGRAM 71.4%) vs full-context (56.2%). Our top-25 sits between; we should report both. |
| 9 | **Session timestamp injection** | evidence-dated revision exists (`a9c4577`) for evidence mode; not applied to full-conv | every Mem0 prompt explicitly asks "convert relative time references to specific dates" with session timestamps in context | **medium** — add timestamp dating to full-conv prep mode. Already covered in evidence mode. |
| 10 | **Justify-then-judge as the default** | implemented (`justify_then_judge.py`) | ByteRover / Hindsight use it; Mem0 / Memobase do not | **already done** — keep this. Note that Mem0's single-stage `gpt-4o-mini` ACCURACY_PROMPT may be cheaper and still calibrated; consider supporting both modes for direct comparison. |

### Low impact

| # | Delta | Our practice | Leaderboard median | Fix complexity |
|---|---|---|---|---|
| 11 | **Cost transparency** | logged per-run in JSON | Mem0/Letta blog posts cite total $ cost | **trivial** — already have it. Surface in the positioning doc table. |
| 12 | **Conversation count reporting** | 1533 LoCoMo items | ByteRover specifies "272 sessions across 10 conversations, 1982 questions" | **trivial** — report both. Our 1533 is post-evidence-mode filter from 1542 raw; we should explicitly say "1542 source → 1533 after dropping items with empty evidence". |
| 13 | **Tool-call latency penalty** | no penalty | Letta penalizes "extraneous memory operations" | **N/A** — Letta is an agent loop; we are a retrieval engine. Different architecture. |

---

## 4 — Anti-patterns we should NOT adopt

Leaderboard practices that are controversial, retracted, or known-broken:

1. **Silently dropping Category 5.** The Mem0 / Memobase / Letta
   convention of excluding Category 5 *can be justified* (it's
   adversarial / abstention-testing, different task shape) but the
   `evaluate_gpts.sh` script does it *silently* with no
   `--exclude-adversarial` flag. dial481 calls this out as a
   methodology bug. **What to do:** exclude Category 5 from the
   headline (to match field), but **also publish a Cat-5-included
   number with a footnote**. Honest, transparent, and outperforms
   the field on rigor.

2. **Modifying the system prompt to favor your system.** Zep got
   retracted in part for this. The Mem0 audit cites "alterations to
   system_prompt that favor Zep's model, invalidating direct
   comparisons" as a methodology failure. **What to do:** Our
   answerer prompt is short and generic; we should not tune it past
   the Mem0 template style. Any prompt change must be versioned and
   noted in the headline.

3. **Single-stage judge with no reasoning trace.** Mem0's binary
   `ACCURACY_PROMPT` is fast and cheap but is also the source of the
   62.81 % wrong-but-vague-topical false-pass rate dial481 measured.
   **What to do:** Keep our justify-then-judge as the default,
   because the trace catches the false-positive case dial481 documents
   even if it scores us lower in the headline. Lower-honest beats
   higher-inflated.

4. **F1 / BLEU as primary metric.** The LoCoMo paper itself warns
   "Alice was born March" vs "Alice was born July" both score high on
   F1, despite being factually opposite. Memobase / Mem0 report F1
   + BLEU + LLM-Judge but only LLM-Judge is the headline. **What to
   do:** Never lead with F1 on LoCoMo; report as supplementary only.

5. **Including conversation hints in the question.** ByteRover
   explicitly notes "no conversation ID hints provided — the system
   retrieved context from all sessions independently." Some early
   LoCoMo leaderboard runs leaked which conversation contained the
   answer. **What to do:** Verify our prep script doesn't
   inadvertently include conversation IDs in the question text.

6. **Self-evaluation (same model as answerer + judge).** Multiple
   systems hit suspiciously high numbers when answerer and judge are
   the same instance of `gpt-4o-mini`. **What to do:** Our justify
   and judge currently both use DeepSeek-Chat — same model. We should
   either split to two providers, or use a different OpenAI model for
   answer vs judge to factor out self-bias. ByteRover uses Gemini 3
   Pro (justifier) + Gemini 3 Flash (judge), explicitly different.

7. **Token-cost reporting that doesn't include all sequential calls.**
   dial481 caught EverMemOS reporting 2,298 tokens/query when actual
   usage was 6,669 tokens (multi-call agentic pipeline). **What to
   do:** Our cost ledger should sum *all* DeepSeek calls per item —
   answer + justify + judge = ~3 calls per item, ~$0.0011 per item at
   DeepSeek pricing, ~$1.68 per 1533-item full run. Report this
   number; do not divide.

---

## 5 — Recommendations

Ordered by ROI:

### Immediate (next bench run)

1. **Switch judge to `gpt-4o-mini`** in the challenger profile.
   ~$0.30 added cost per run; aligns us with 5/7 leaderboard systems.
2. **Add Category 5 exclusion** with a documented flag and publish
   both numbers.
3. **Add per-category aggregation** to `reporting.py` so the headline
   table reads like Mem0's: single-hop / multi-hop / temporal /
   open-domain / overall.
4. **Switch justifier and judge to different providers** (or different
   OpenAI models) to break self-bias. E.g. justify with DeepSeek, judge
   with `gpt-4o-mini`.

### Short-term (next sprint)

5. **Run 3 seeds and report mean ± std.** Cost ~$5-15 per full run ×
   3 = ~$15-45. This is the cost of statistical legitimacy.
6. **Implement Mem0-style ANSWER_PROMPT** as a parallel
   `PromptBuilder` subclass (versioned), A/B against current. Keep
   whichever gives higher score *without* gaming, document both.
7. **Publish a retrieval-only BEIR-style table** as a parallel headline
   — R@5, R@10, NDCG@5 — derived from `scripts/probe_rank_diagnostic.py`.
   This number is robust against every judge-bias failure mode that
   has retracted three systems' headline numbers, and frames us
   alongside ENGRAM's retrieval-only positioning rather than Mem0's
   end-to-end LLM-Judge positioning.

### Long-term (architectural)

8. **Accept that LoCoMo Judge is GPT-4o-mini's home turf.** Our
   value prop is sub-millisecond retrieval + 5× compact footprint +
   KV-native API (per `latency_first.md`). The LoCoMo race is winnable
   only by paying GPT-4o-mini-class costs at capture time, which
   conflicts with the KV-native architecture's local-model
   requirement. The honest framing: TardigradeDB is a *retrieval
   engine* that any answerer can sit on top of; the canonical
   benchmark for retrieval engines is BEIR-style R@K, which we
   already do competitively at small scale.
9. **Consider adding a hybrid mode** where the bench harness can run
   our retrieval feeding `gpt-4o-mini` as the answerer. This factors
   capture-model quality out of the comparison and isolates retrieval
   quality. It's not the KV-native vision, but it produces a
   leaderboard-comparable number.

---

## 6 — Citations

### Papers

- Maharana et al., "Evaluating Very Long-Term Conversational Memory of
  LLM Agents", ACL 2024.
  [aclanthology.org/2024.acl-long.747](https://aclanthology.org/2024.acl-long.747/) ·
  [aclanthology.org/2024.acl-long.747.pdf](https://aclanthology.org/2024.acl-long.747.pdf)
- Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term
  Interactive Memory", ICLR 2025.
  [arxiv.org/abs/2410.10813](https://arxiv.org/abs/2410.10813)
- Mem0 paper (Mem0 / OpenAI / LangMem / MemGPT comparison, 2025).
  [arxiv.org/abs/2504.19413](https://arxiv.org/abs/2504.19413) ·
  [arxiv.org/html/2504.19413v1](https://arxiv.org/html/2504.19413v1)
- ENGRAM paper, "Effective, Lightweight Memory Orchestration for
  Conversational Agents" (2511.12960).
  [arxiv.org/abs/2511.12960](https://arxiv.org/abs/2511.12960) ·
  [arxiv.org/html/2511.12960v2](https://arxiv.org/html/2511.12960v2)
- Hindsight paper, "Hindsight is 20/20: Building Agent Memory that
  Retains, Recalls, and Reflects" (2512.12818).
  [arxiv.org/html/2512.12818v1](https://arxiv.org/html/2512.12818v1)

### Code repositories

- LoCoMo official: [github.com/snap-research/locomo](https://github.com/snap-research/locomo)
- LongMemEval official: [github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)
- Mem0 evaluation harness: [github.com/mem0ai/mem0/tree/main/evaluation](https://github.com/mem0ai/mem0/tree/main/evaluation)
- Mem0 `llm_judge.py` (ACCURACY_PROMPT verbatim):
  [github.com/mem0ai/mem0/blob/main/evaluation/metrics/llm_judge.py](https://github.com/mem0ai/mem0/blob/main/evaluation/metrics/llm_judge.py)
- Memobase LoCoMo experiment:
  [github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md)
- Mem0 cross-system benchmark harness:
  [github.com/mem0ai/memory-benchmarks](https://github.com/mem0ai/memory-benchmarks)
- LongMemEval `evaluate_qa.py` (uses `gpt-4o-2024-08-06`, T=0,
  max_tok=10): same repo, `src/evaluation/evaluate_qa.py`

### Blog posts + leaderboard claims

- ByteRover 2.0 LoCoMo 92.2 %, Gemini 3 stack, Hindsight prompts:
  [byterover.dev/blog/benchmark-ai-agent-memory](https://www.byterover.dev/blog/benchmark-ai-agent-memory)
- ByteRover LongMemEval-S 92.8 %:
  [byterover.dev/blog/benchmark_ai_agent_memory_real_production_byterover_top_market_accuracy_longmemeval](https://www.byterover.dev/blog/benchmark_ai_agent_memory_real_production_byterover_top_market_accuracy_longmemeval)
- MemMachine v0.2 LoCoMo 91.23 % (`gpt-4.1-mini` answerer):
  [memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark](https://memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/)
- Letta 74 % LoCoMo (`gpt-4o-mini` answerer, `gpt-4.1` judge,
  SimpleQA scoring):
  [letta.com/blog/benchmarking-ai-agent-memory](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- Mem0 cross-system comparison blog:
  [mem0.ai/blog/benchmarked-openai-memory-vs-langmem-vs-memgpt-vs-mem0-for-long-term-memory-here-s-how-they-stacked-up](https://mem0.ai/blog/benchmarked-openai-memory-vs-langmem-vs-memgpt-vs-mem0-for-long-term-memory-here-s-how-they-stacked-up)
- Mem0 evaluation framework wiki:
  [deepwiki.com/mem0ai/mem0/14.6-evaluation-framework](https://deepwiki.com/mem0ai/mem0/14.6-evaluation-framework)
- Hindsight Agent Memory Benchmark manifesto (vectorize.io):
  [hindsight.vectorize.io/blog/2026/03/23/agent-memory-benchmark](https://hindsight.vectorize.io/blog/2026/03/23/agent-memory-benchmark)
- AI Memory Benchmarks in 2026 (Mem0 blog):
  [mem0.ai/blog/ai-memory-benchmarks-in-2026](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)

### Audits, retractions, third-party reproductions

- dial481 LoCoMo audit (6.4 % wrong gold, 22.5 % Cat 5 silently
  dropped, 62.81 % judge false-pass rate, EverMemOS 38.38 % vs
  claimed 92.32 %):
  [github.com/dial481/locomo-audit](https://github.com/dial481/locomo-audit)
- Zep retraction (84 % → 58.44 % → 75.14 %; Cat 5 inflation, single-run
  bias, prompt modification):
  [github.com/getzep/zep-papers/issues/5](https://github.com/getzep/zep-papers/issues/5)
- Mem0 Zep audit blog ("Lies, damn lies, and statistics"):
  [blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- TardigradeDB Phase 1B.9 judge-bias retraction (66 % → 42 % after
  v2 prompt): [`docs/experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md)

### Retrieval-only methodology

- ENGRAM retrieval-only thread on snap-research/locomo:
  [github.com/snap-research/locomo/issues/38](https://github.com/snap-research/locomo/issues/38)
  (R@5=93.9 %, R@10=95.0 %, NDCG@5=0.894)

### Verbatim prompts (the contents are load-bearing)

Mem0 `ACCURACY_PROMPT` (LoCoMo judge),
`evaluation/metrics/llm_judge.py` — `gpt-4o-mini`, `temperature=0.0`,
`response_format={"type": "json_object"}`:

> Your task is to label an answer to a question as 'CORRECT' or
> 'WRONG'. You will be given the following data: (1) a question
> (posed by one user to another user), (2) a 'gold' (ground truth)
> answer, (3) a generated answer which you will score as
> CORRECT/WRONG. […] The generated answer might be much longer, but
> you should be generous with your grading - as long as it touches on
> the same topic as the gold answer, it should be counted as CORRECT.
> For time related questions, […] generous with your grading - as
> long as it refers to the same date or time period as the gold
> answer, it should be counted as CORRECT. […] First, provide a short
> (one sentence) explanation of your reasoning, then finish with
> CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your
> response, or it will break the evaluation script. Just return the
> label CORRECT or WRONG in a json format with the key as "label".

LongMemEval judge (`gpt-4o-2024-08-06`, T=0, max_tok=10, per-type
yes/no prompts):

> I will give you a question, a correct answer, and a response from a
> model. Please answer yes if the response contains the correct
> answer. Otherwise, answer no. […] Is the model response correct?
> Answer yes or no only.

Mem0 `ANSWER_PROMPT` (LoCoMo answer-generation, applies to all
two-speaker variants `ANSWER_PROMPT`, `ANSWER_PROMPT_GRAPH`,
`ANSWER_PROMPT_ZEP`), key directives extracted:

> Pay special attention to the timestamps; if the memories contain
> contradictory information, prioritize the most recent memory;
> convert relative time references (e.g. "last year") to specific
> dates based on memory timestamps; ignore the reference while
> answering the question; keep answers under 5-6 words; distinguish
> between character names in memories versus actual users.

---

**Final note.** This audit pairs with [`docs/experiments/2026-05-14-bench-audit.md`](../experiments/2026-05-14-bench-audit.md)
and [`docs/positioning/latency_first.md`](../positioning/latency_first.md):

- The 2026-05-14 audit is the *internal honesty pass* — what numbers
  we measured and why prior ones were retracted.
- The latency_first positioning doc is the *external framing* — what
  we legitimately compete on today.
- This document is the *third leg* — given the external standard,
  exactly what we'd need to do (and not do) to produce a comparable
  LoCoMo Judge number, and where the structural ceilings lie.

Together they form an internally consistent and externally defensible
record.
