# SGLang KV Connector Investigation

**Date:** April 28, 2026
**Status:** Complete — **NOT VIABLE** for cross-prompt KV injection

## Question

Does SGLang's KV transfer/caching API allow injecting KV cache tensors that don't correspond to the literal prompt prefix? Can a connector provide pre-computed K/V tensors as "memory context" without those tokens being in the current prompt?

## Answer: No

SGLang's KV cache architecture is **strictly prefix-based**, identical to vLLM v1. The RadixAttention system uses a radix tree keyed on token sequence identity. `match_prefix()` operates on `RadixKey(token_ids)` — matching only happens when the exact same token sequence exists.

## Evidence

1. **RadixAttention:** Cache lookup is `RadixKey(token_ids)` → device indices. No semantic or arbitrary tensor injection mechanism.

2. **MatchResult contract:** Returns `device_indices` for matched prefix tokens only. No field for "extra context KV" or "memory augmentation."

3. **LMCache integration:** Uses content-hash-based prefix matching. It's a storage layer on the same prefix-matching tree — not semantic injection.

4. **Remote KV Connector RFC (#7746):** Proposed but not yet implemented. Even the RFC specifies `num_external_matched_tokens()` — prefix matching from remote storage, not arbitrary injection.

## Implication for TardigradeDB

Both major production LLM serving frameworks (vLLM and SGLang) are architecturally prefix-cache-only. Cross-prompt KV injection — TardigradeDB's core value proposition — is not possible through their current connector APIs.

**Confirmed deployment paths:**
- **Path 1 (HuggingFace direct injection):** Works. 9/10 synthetic facts, 46% token savings. The `model.generate(past_key_values=...)` approach bypasses the serving framework entirely.
- **Path 2 (Memory prefix):** Works. Governed text prefix served via stock prefix-cache. No KV injection — text approach.

**Closed paths:**
- **Path 3 (SGLang):** Not viable. Same prefix-only limitation as vLLM.
- **Path 4 (Custom attention plugin):** Remains the only theoretical path for KV injection in production serving. Requires forking vLLM/SGLang to modify the attention layer. Heavy, ongoing maintenance burden.

## Recommendation

Focus on Path 1 for applications with direct model access (research, single-model deployments). Use Path 2 for vLLM/SGLang production serving. Accept that production serving frameworks don't support cross-prompt KV injection today — this may change with vLLM v2 or future SGLang RFCs. Monitor upstream developments rather than maintaining a fork.
