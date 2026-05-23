# vLLM connector recall divergence — 2026-05-23

**Status:** characterization complete. The `test_primed_request_recalls_synthetic_fact` failure is an architectural mismatch between vLLM's KV Connector v1 API and content-addressed memory recall, not a bug fix away from working.

---

## TL;DR

1. **HF KV injection works (9/10 on synthetic facts).** The HF hook stores `k_proj(hidden_states)` *before* RoPE is applied. Pre-RoPE K vectors are content-only — position-independent — so the model can attend to injected K from one sequence using Q from another.

2. **vLLM KV injection does not surface saved facts.** vLLM's attention kernel applies RoPE inside the kernel before writing to the K cache. The tensor the connector receives in `save_kv_layer` is post-RoPE. Copying that into a new request's block slots places content whose positional encoding belongs to the *original* fact's positions, not the new question's positions. Q·K dot products under attention are scrambled by the rotation mismatch.

3. **vLLM's KV Connector v1 API is built for prefix-cache reuse**, not semantic memory recall. Prefix-cache reuse only requires bit-identical prefixes at identical positions — RoPE alignment is trivial because positions match by construction. Content-addressed recall requires positions to *differ* between save and load, which is the case RoPE breaks.

4. **CLAUDE.md already flagged this** as a known design target: *"decoupled position encoding for safe historical KV block reuse"* (Storage Layer section). The fix is substantial — strip RoPE at save time and re-apply with target positions at load time — and is not a near-term roadmap item.

5. **The product surface for semantic memory under vLLM is the prefix client** (`VLLMMemoryClient` in `tardigrade_vllm/prefix_client.py`), validated by `test_vllm_prefix_e2e.py`. That path stuffs retrieved text into the prompt and lets the model encode it through normal attention. It works because the model produces its own RoPE-rotated K vectors at the correct positions.

---

## The failing test

`tests/python/test_vllm_integration.py::test_primed_request_recalls_synthetic_fact`

```
1. Cold:   llm.generate(["Who discovered the moons of Quthar?"])      → no Zorblax
2. Prime:  llm.generate(["Zorblax discovered the moons of Quthar in
                          the year 2089."])                            → save pack
3. Primed: llm.generate(["Who discovered the moons of Quthar?"])      → still no Zorblax
```

The scheduler-side match fires (GPU run output shows `[match-OK] returning seq_len=16 (matched pack 20)`). The worker-side `start_load_kv` copies the saved layer data into the allocated block slots without error. The model generates a coherent response that does not contain the synthetic answer token.

Match works. Copy works. Attention does not surface the content.

---

## What HF does — and why it works

`python/tardigrade_hooks/hf_kv_hook.py` captures K and queries with Q in the same projection space:

- `_project_k(hidden_states, layer_idx)` (line 75–93): calls `attn.k_proj(hidden_states)` and returns the result directly. Comment on line 76: *"Apply K projection (without RoPE) to hidden states."* The function never touches `rotary_emb` or any positional component.
- `_project_q(hidden_states, layer_idx)` (line 95–112): mirror on the Q side.

The K vectors stored in TardigradeDB are content-only. A token's K vector at position 5 of one prompt is identical (modulo k_norm) to the same token's K vector at position 12 of another prompt. Dot products between Q vectors and stored K vectors measure semantic similarity, independent of where each token lived in its source sequence.

This is why `test_kv_inject_hook_with_synthetic_facts` (HF path) sees 9/10 recall on Qwen3-0.6B with fully synthetic gibberish facts.

---

## What vLLM does — and why it does not work

`python/tardigrade_vllm/connector.py::TardigradeConnector.save_kv_layer` receives a `kv_layer` tensor handed in by vLLM's attention kernel. By the time the connector sees it, vLLM has already:

1. Computed `k = k_proj(hidden_states)`
2. Applied **rotary position embedding** to `k` based on the token's position in the current sequence
3. Written the post-RoPE `k` into the layer's paged K cache

The connector's save path treats this tensor as opaque content and writes it to disk. On load, `start_load_kv` reshapes that data via `flat_to_paged` and copies it into the block slots vLLM allocated for the *new* request. The copy succeeds; the tensor lands in the right slots.

The problem is what the next forward pass sees. When the model attends:

- Q for token *i* of the new request is rotated by RoPE for position *i*.
- K injected from the saved pack was rotated by RoPE for the position that token held in the *original* fact prompt — typically a different value.

Q·K under attention computes:

```
(R(θ_q_new) · q) · (R(θ_k_old) · k) = q · R(θ_q_new - θ_k_old) · k
```

The relative rotation `θ_q_new − θ_k_old` is wrong (it carries information about the *fact's* positional offsets, not the question's). The dot product is no longer measuring content similarity — it's measuring content similarity rotated through an arbitrary angle. For most layer indices and token offsets, that rotation puts the actual signal into the noise floor.

Result: the model has the saved K in memory but cannot meaningfully attend to it.

---

## Why this is structural, not a bug

The vLLM KV Connector v1 API is designed for a specific use case: **prefix-cache reuse across requests with identical prefix tokens**. The reference implementation (LMCache) ships KV between worker nodes when one node has already computed the prefix and another can skip the work. In that scenario:

- The prefix text is byte-identical between source and destination request.
- The positions of each token within that prefix are identical (both start at position 0).
- The post-RoPE K vectors land at the same positions they were rotated for. Attention is correct.

Content-addressed memory recall — "I stored a fact in conversation A, now retrieve it in conversation B" — violates the API's positional-identity assumption. The cache is built around the assumption that positions match. Connectors get post-RoPE K because that's the form that participates in attention; there's no way to get pre-RoPE K through the public connector hooks.

CLAUDE.md's "decoupled position encoding for safe historical KV block reuse" calls out the fix: strip RoPE at save time (requires pre-RoPE access, which means patching vLLM or running a custom attention layer) and reapply at load time with the destination positions. The investigation cost is substantial — vLLM's attention kernels are CUDA-side, fused, and version-volatile — and there is no roadmap commitment to it today.

---

## Three remediation paths

**Path A — accept the limitation, route semantic memory through the prefix client.**
The prefix client (`VLLMMemoryClient`) is already implemented, exercised by `test_vllm_prefix_e2e.py`, and works end-to-end on synthetic facts because the model produces its own RoPE-rotated K vectors at the correct positions for the prepended text. The vLLM KV connector remains useful for prefix-cache reuse (the API's actual purpose), but is not the semantic-memory surface.

Implication: rewrite or mark `test_primed_request_recalls_synthetic_fact` as testing a path it cannot validate. The architectural claim it makes ("KV save → retrieve → inject surfaces stored content via vLLM connector") is currently false and not addressable without decoupled position encoding.

**Path B — decoupled position encoding.**
Capture K *before* RoPE in vLLM. Requires intercepting `forward_decode` / `forward_prefill` at the attention layer rather than at the connector hooks, which means a custom attention class registered through vLLM's attention backend plugin system or a patched build. At load, apply RoPE with the destination request's positions. Cost: weeks of focused work, version-volatile, and the storage doubles (need to keep pre-RoPE K alongside the existing pack format). High leverage if the product target requires vLLM-native semantic memory.

**Path C — content storage in the prefix-client layer, with vLLM connector as transport only.**
Keep the connector as a thin pass-through that delivers retrieved text into the request, not retrieved KV. The connector still wins prefix-cache reuse for the prepended retrieved-text prefix (because vLLM's own prefix cache recognises identical token-id prefixes across requests). This collapses to Path A with a slightly more honest framing of the connector's role.

---

## Recommendation

**Path A.** The prefix client demonstrably works for semantic recall on synthetic facts; the vLLM KV connector demonstrably does not, for documented architectural reasons that require non-trivial engineering to address. Spending another session trying to make the connector path do content recall is throwing effort at a problem the API surface was not designed for.

Concretely:

1. Rewrite `test_primed_request_recalls_synthetic_fact` as a `test_primed_request_recalls_synthetic_fact_via_prefix_client` test, asserting the prefix-client path. The existing test should be deleted, not skipped — keeping a failing GPU test in the suite as a TODO marker rots into noise.
2. Update the `Status` section of the README and the v0.7.5 CHANGELOG retraction note to make the two-path story explicit: "vLLM KV connector path = prefix-cache reuse; prefix-client path = semantic memory."
3. Mark task #11 as **characterized; resolution path: Path A**.

Path B remains an option if a downstream consumer needs vLLM-native semantic memory specifically. It is not the right work for this session and probably not the right work for this minor version.

---

## Verification of the diagnosis

The hypothesis is grounded in source-reading, not empirical probing:

- HF pre-RoPE storage: `python/tardigrade_hooks/hf_kv_hook.py:75–93` (k_proj output, no rotary_emb call).
- vLLM post-RoPE storage: implicit in vLLM's attention backend design — the K cache contract is that stored K vectors have already had RoPE applied. The connector cannot reach inside the kernel to grab pre-RoPE K through the public `save_kv_layer` hook.
- Existing positive control: `test_kv_inject_hook_with_synthetic_facts` (HF path) shows 9/10 recall on Qwen3-0.6B with the exact same `SYNTHETIC_FACT` test pattern.
- Existing negative control: the failing test itself shows the model not surfacing the saved content despite a successful match + load + copy.

The minimum empirical test that would falsify the hypothesis would be: store post-RoPE K from the *same* sequence positions the fact occupies, then ask the question at exactly those positions — recall should jump if the diagnosis is correct. Building that experiment is more code than it's worth given the source-level evidence already points one way; the cost-justified action is to commit to Path A and move on.
