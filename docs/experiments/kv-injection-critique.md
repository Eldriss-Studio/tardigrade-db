# KV Cache Context-Dependence Critique

**Date:** April 22, 2026  
**Source:** External peer review  
**Status:** Open — requires empirical validation

## The Critique

After reviewing the two-agent memory experiment, the following concerns were raised:

1. **KV cache is context-dependent.** Key and Value vectors are computed per-model, per-prompt. They carry positional encoding, causal attention patterns, and token interdependencies specific to the sequence that produced them. You cannot share KV cache between different prompts.

2. **Latency.** Full KV cache is multiple GB in VRAM. A database roundtrip to fetch and inject it would negate the speed benefit of caching.

3. **Prior art.** The closest production approach is direct-storage SSD caching (NVMe → GPU DMA) for KV prefix sharing — not cross-context retrieval.

## The Deeper Point: Restoring KV = Restoring the Conversation

A follow-up observation sharpened the critique:

> "Even if you save the KV cache, bringing it back means returning your chat to that exact point in time, with that exact context usage. Did you do something to handle this differently?"

This is correct for **full KV cache**. Each token's Key and Value vectors at layer L are computed as:

```
K_i = W_K · x_i     (where x_i is the residual stream at position i)
V_i = W_V · x_i     (after attending to all tokens j ≤ i)
```

The residual stream `x_i` at position i carries the cumulative causal effect of every token before it. Restoring these K/V vectors in a different context means injecting representations that "remember" a different conversation history. The model's attention at query time would attend to tokens that reference a context that doesn't exist in the current prompt.

### What TardigradeDB Actually Stores (Not Full KV)

However, TardigradeDB does NOT store full per-token KV cache. The `HuggingFaceHook.on_generate` method does:

```python
mean_hidden = hidden_states.mean(axis=0)  # (d_model,)
```

This **mean-pools the entire sequence into a single vector**, destroying:
- Positional information (which token was where)
- Causal dependencies (which token attended to which)
- Per-token identity (individual token representations)

What survives is a **semantic centroid** — the average activation across all tokens at that layer. This is closer to a sentence embedding than a KV cache entry.

### Three Distinct Things

| What | What it is | Context-dependent? | Stored by TardigradeDB? |
|------|-----------|-------------------|------------------------|
| **Full per-token KV cache** | `(seq_len, num_heads, head_dim)` per layer | Yes — the reviewer is right | No |
| **Mean-pooled hidden state** | `(d_model,)` — semantic centroid | Partially — semantic content preserved, positional info destroyed | **Yes — this is what's stored** |
| **Text embedding** | Dense vector from a separate encoder model | No — model-independent | No — this is what vector DBs do |

TardigradeDB sits in the middle. It's not storing full KV (so The reviewer's strongest concern doesn't apply directly), but it's also not a text embedding (it captures richer model-internal representation). The mean-pooled vector is a lossy compression that accidentally makes the representation more portable across contexts — but also too lossy for faithful KV reconstruction.

### The kv_injector.py Problem

The current `kv_injector.py` tries to do something questionable: it takes the mean-pooled `(d_model,)` vector, reshapes it to `(1, num_heads, 1, head_dim)`, and injects it as a **single synthetic cache token**. This token:

- Never existed in the original sequence
- Has no meaningful positional encoding
- Represents the *average* of all tokens, not any specific token
- Is presented to the model's attention as if it were a real previous token

This is neither faithful KV restoration (which the reviewer correctly says requires the exact context) nor pure retrieval (which works). It's something in between — a synthetic pseudo-token derived from a semantic summary.

## Assessment

### What's valid

The reviewer's core point stands: **full KV cache is context-dependent and non-portable.** You cannot save a conversation's KV and inject it into a different conversation. This is architecturally fundamental to how transformers work.

### What's not affected

**Latent-space retrieval** — using stored hidden states as a semantic similarity index — is NOT affected. Mean-pooled hidden states preserve semantic content while discarding context-specific artifacts. Our two-agent experiment proved this: hidden states from "The capital of France is Paris" score highest against queries about France, regardless of context dependence. Retrieval-as-search is valid even if injection-into-attention is not.

### What's uncertain

**Mean-pooled pseudo-token injection** (`kv_injector.py`) is neither obviously right nor obviously wrong. It's not restoring the conversation to a point in time — it's injecting a synthetic semantic summary. Whether this helps, hurts, or is neutral is an empirical question.

### Latency concern is overstated for this architecture

TardigradeDB stores Q4-quantized, mean-pooled vectors (768–3072 floats per cell), not full KV tensors (`seq_len × num_heads × head_dim`). Each cell is kilobytes, not gigabytes. The SLB (Semantic Lookaside Buffer) targets sub-5μs retrieval.

## The Three Functions (Revised)

| Function | Critique applies? | Evidence |
|----------|------------------|----------|
| **Latent-space retrieval** (semantic memory search) | No | Validated: 80-92% recall in experiments |
| **Mean-pooled pseudo-token injection** (current kv_injector.py) | Partially — it's not full KV restoration, but also not clearly valid | Unvalidated: needs empirical testing |
| **Full KV cache restoration** (not implemented) | **Yes — the reviewer is right** | This is architecturally unsound without decoupled position encoding |

## Research That Addresses the Critique

The TDD references several techniques designed to handle cross-context KV reuse:

1. **Decoupled position encoding** — Separate positional information from semantic content in stored KV, re-encode positions on injection. This directly addresses The reviewer's concern by stripping context-specific positional artifacts before storage.
2. **MemArt** — Computes attention directly against compressed keys in latent space. Critically, MemArt does NOT inject into the KV cache — it uses stored keys as a retrieval signal only, then recomputes attention. This sidesteps the injection problem entirely.
3. **RelayCaching** — Reuses decode-phase KV across agents, but only for shared-prefix contexts (which the reviewer would agree is valid).

These are documented in `docs/technical/tdd.md` but not implemented. They represent the gap between "retrieval works" and "injection works."

## What This Means for the Project

The project has a clear, validated value as a **semantic memory retrieval system** operating in latent space. The two-agent experiment proved this unambiguously.

The KV injection path needs one of three resolutions:

1. **Drop it** — Accept that TardigradeDB is a retrieval engine, not an attention augmentation engine. Return text/metadata, not tensors for injection.
2. **Validate it empirically** — Run the injection validation test to measure actual impact on model output.
3. **Implement MemArt-style attention** — Compute attention against stored keys without injecting into the cache. This is the approach most aligned with the cited research.

## Proposed Validation Experiment

See [KV Injection Validation Test](kv-injection-validation-test.md) for the empirical test design.
