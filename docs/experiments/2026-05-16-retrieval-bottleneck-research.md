# Retrieval Bottleneck Research — 2026-05-16

Companion to Phase 1B.10 of `docs/experiments/2026-05-14-bench-audit.md`.
After the chunk-text fix (commit `64be483`) exposed the honest
retrieval baseline (LoCoMo R@5 = 17% aggregate; ~28% LoCoMo-only),
this memo captures a focused research pass on **why** retrieval is
that low and what to change first.

Status: research only — no code changes recommended by this memo
have been made yet.

## Headline finding

**The retrieval metric is measuring the wrong target.**
LoCoMo's `evidence` dia_ids mark *supporting context*, not
*answer-bearing turns*. On a 10-item sample inspection, ~80% of
items have evidence text that doesn't literally contain the
answer. Example:

| Question | Ground truth | Marked evidence turn |
|---|---|---|
| When did Melanie paint a sunrise? | `2022` | "You'd be a great counselor!…" (no date) |
| What is Caroline's relationship status? | `Single` | "I've known these friends for 4 years…" (no status) |

Our retrieval recall (`python/tdb_bench/metrics/retrieval.py`)
substring-matches gold-evidence text against retrieved chunks.
When the answer lives in a *different* turn than the marked
evidence, the metric scores zero even if the retriever found
answer-bearing chunks. ~70-80% false-negative rate from this
mismatch alone.

**Implication:** our 17-28% R@5 is not directly comparable to
Mem0 (67%), Letta (74%), or ByteRover (92%). Those systems use
LLM-judge or answer-text-match, not oracle-evidence-match. Our
metric is harder. **We may be punishing ourselves for a target
mismatch, not for retrieval quality.**

## Ranked hypotheses

| # | Hypothesis | Evidence | Likely impact |
|---|---|---|---|
| 4 | **Oracle evidence ≠ answer-bearing** | 8/10 sampled items, gold text doesn't contain answer | **Highest — wrong target measured** |
| 3 | Reranker (`ms-marco-MiniLM-L-6-v2`) is web-search trained, not dialogue-trained. Memory note `project_dialogue_reranker_recommendation.md` recommends `Qwen3-Reranker-0.6B`. | Confirmed in `python/tardigrade_hooks/reranker.py:DEFAULT_MODEL`. | Medium — Stage 2 only |
| 2 | Per-token Top5Avg hidden states from Qwen3-1.7B may lack semantic capacity for dense retrieval at full-conversation scale. Layer 18 of 27 chosen heuristically. | `python/tardigrade_hooks/hf_kv_hook.py` uses `use_hidden_states=True` at the configured query layer. Position 0 correctly skipped per memory rule. | Medium — architecture is reasonable but model isn't retrieval-trained |
| 1 | Chunker splits mid-turn → substring match fails on boundary turns. | `ParagraphBoundaryStrategy` (max_tokens=128, overlap=16) is preserving turn boundaries cleanly — `\n\n` between speakers gets the highest priority. Empirically not the bottleneck. | **Low — not the cause** |
| 5 | Industry standard for conversational memory retrieval | Not investigated in depth — most production systems use hybrid (BM25 + dense) + LLM-as-judge for retrieval evaluation. | Out of scope for this memo |

## Recommended next move

**Change what the retrieval metric measures, not how it measures.**

The current metric tests "did the retriever surface the
LoCoMo-marked evidence turn?" That's not what predicts downstream
LLM-Judge performance. The better question is **"did the retriever
put a chunk containing the answer text in the LLM's window?"**

Concrete change (~30 LOC, no architectural impact):

1. Add an `answer_text_metrics` block alongside the existing
   `retrieval_metrics`. Compute recall@k by substring-matching
   `ground_truth` (or any token of it for short answers) against
   the retrieved chunks. Keep both metrics — the existing one is
   the audit-resistant ENGRAM-style number; the new one is the
   downstream-predictive number.
2. For short ground truths (1-2 words) that are common English
   words, fall back to "answer text appears with surrounding
   gold-evidence context" — guards against false positives on
   tokens like "Single" or "yes."

After this, **then** revisit the retrieval architecture
(hypotheses 2, 3) with the new metric in hand. With the wrong
metric, any architecture change is firing blind.

## Other bugs / smells found during research

- **Hypothesis 4 is itself a metric design issue, not a code bug.**
  The substring match is correct; it just answers the wrong
  question.
- **Per-token Top5Avg over Qwen3-1.7B at layer 18.** Choice of
  layer 18 (67% of 27) appears heuristic. No principled study of
  which layer encodes "search this dialogue for X" semantics
  best. Worth a small ablation before committing to alternative
  retrievers.
- **`cell_to_chunk_text` storage cost.** Each chunk text is held
  in a Python dict alongside the engine's tensor storage. At 5K
  cells × ~500 chars = ~2.5 MB — fine for benches, but not part
  of the engine's tracked storage footprint.

## Key code references

- Substring match (the target-mismatch site):
  `python/tdb_bench/metrics/retrieval.py:85-89`
- Runner attach site:
  `python/tdb_bench/runner.py:418-421`
- Chunker boundary strategy (working correctly):
  `python/tardigrade_hooks/chunker.py:81-108`
- Hook retrieval keys:
  `python/tardigrade_hooks/hf_kv_hook.py:215-266`
- Reranker default model:
  `python/tardigrade_hooks/reranker.py:DEFAULT_MODEL`
- Bench results (smoke #6):
  `target/bench-challenger-smoke30-dedup.json`

## What we are NOT changing this session

- The retrieval architecture (keys, reranker, chunker).
- The LLM-Judge scoring path.
- The bench dataset prep.

Those wait until the new answer-text metric is in place and we
can read which architecture change is actually moving the needle.

## Open task

Task #100 will track implementation of the answer-text retrieval
metric and an A/B comparison against the current evidence-text
metric on the smoke30 corpus.
