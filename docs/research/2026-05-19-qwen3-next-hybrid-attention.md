# Research: Qwen3-Next hybrid attention architecture and KV-cache persistence

- **Slug:** `2026-05-19-qwen3-next-hybrid-attention`
- **Date:** 2026-05-19
- **Status:** complete
- **Triggered by:** (1) `AttributeError: 'LinearAttentionLayer' object has no attribute 'keys'` raised in `python/tardigrade_hooks/kp_injector.py::store` when swapping the casper-spike's model from Qwen3-1.7B to Qwen3.5-2B. The chat-template Adapter shipped in v0.3.2 fixes the *prompt-wrapping* crash on strict templates, but does not address the cache-attribute crash that follows. Scope question raised: how should tardigrade-db treat hybrid-attention models in general? (2) Follow-up question after the Phase 0 spike on RecurrentGemma-2B produced an unexpected H1 ≡ H2 byte-identical result: does the literature corroborate or contradict the "softmax slice carries all the retrieval signal" interpretation, and is there a measurement-bug risk?
- **Informed:** `CHANGELOG.md` v0.3.2 "Known Limitations" entry; `python/tardigrade_hooks/chat_template_adapter.py` module-docstring known-constraints section; `experiments/v11-hybrid-spike/README.md` (Phase 0 spike result write-up). Pending: scope decision for tardigrade-db's supported-architecture surface (refuse-cleanly vs. softmax-slice-only vs. full-empirical-spike vs. per-architecture-codec) — leaning toward softmax-slice-only based on both the spike and the literature, with measurement-bug controls pending.

## Question

Qwen3-Next (also marketed as Qwen3.5) is a modern open-weight LLM that crashes tardigrade-db's KV-capture path. Why does it crash, is the fix mechanical or architectural, does the same problem apply to other model families, and is "store-fact-then-reinject-at-runtime" even mathematically defined for the cache state these models hold?

## Sources

### [Qwen3-Next model card (Qwen/Qwen3-Next-80B-A3B-Instruct)](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- **Authors / Org:** Qwen Team (Alibaba)
- **Type:** vendor doc
- **Published:** 2025-09 (ongoing)
- **Accessed:** 2026-05-19
- **Relevance:** high
- **What this contributed:** Authoritative model-card description of the 3:1 hybrid layout — every 4th transformer layer is Gated Attention (softmax MHA with GQA: 16 Q-heads, 2 KV-heads, head_dim=256, partial RoPE on first 64 dims); the other 36 of 48 layers are Gated DeltaNet. Without this we'd be inferring the layout from `config.layer_types` indirectly.

### [Qwen3-Next blog (qwenlm.github.io)](https://qwenlm.github.io/blog/qwen3_next/)
- **Authors / Org:** Qwen Team (Alibaba)
- **Type:** engineering blog
- **Published:** 2025-09
- **Accessed:** 2026-05-19
- **Relevance:** high
- **What this contributed:** Establishes the rationale for the hybrid architecture (long-context efficiency at constant memory cost per token in the linear layers) and explicitly names "Gated DeltaNet" as the linear-attention mechanism. This is the primary anchor that pointed us at the Yang et al. 2024 paper rather than guessing between Mamba-2, RWKV, or Katharopoulos-style linear attention.

### [Gated Delta Networks: Improving Mamba2 with Delta Rule (arXiv 2412.06464)](https://arxiv.org/abs/2412.06464)
- **Authors / Org:** Songlin Yang, Jan Kautz, Ali Hatamizadeh (NVIDIA Research / MIT) — ICLR 2025
- **Type:** academic paper
- **Published:** 2024-12
- **Accessed:** 2026-05-19
- **Relevance:** high
- **What this contributed:** Primary source for the recurrent-state update rule used by Qwen3-Next's linear layers: `S_t = α_t·S_{t-1} + k_t ⊗ delta_t`. Establishes that the state `S` is a single fast-weight matrix of fixed shape — does not grow with sequence length — and that updates are non-associative across independent prefixes because the gating decay `α_t` depends on context. This is the mathematical basis for why "store fact, prepend at runtime" cannot be made exact for these layers.
- **Quoted:**
  > "Gated DeltaNet maintains a fast-weight memory matrix that is updated multiplicatively at each timestep by a data-dependent gating factor and additively by a delta term, generalizing both Mamba2 and the classic DeltaNet."

### [HuggingFace transformers source: cache_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)
- **Authors / Org:** Hugging Face transformers maintainers
- **Type:** open-source project (primary source)
- **Published:** ongoing (transformers ≥4.55)
- **Accessed:** 2026-05-19
- **Relevance:** high
- **What this contributed:** Confirms there is no `Qwen3NextDynamicCache` class. The generic `DynamicCache` reads `config.layer_types[i]` and instantiates per-layer cache objects from a registry: `DynamicLayer` for `"full_attention"` (carries `.keys` / `.values`), `LinearAttentionLayer` for `"linear_attention"` (carries `.conv_states` / `.recurrent_states`, no `.keys`), and `LinearAttentionAndFullAttentionLayer` for `"hybrid"`. The mutator API is `update_conv_state(t, idx)` / `update_recurrent_state(t, idx)`. This is the surface tardigrade-db must adapt to if it supports hybrid models.

### [HuggingFace transformers source: modeling_qwen3_next.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py)
- **Authors / Org:** Hugging Face transformers maintainers
- **Type:** open-source project (primary source)
- **Published:** ongoing
- **Accessed:** 2026-05-19
- **Relevance:** medium
- **What this contributed:** Confirms Qwen3-Next instantiates the cache via `DynamicCache(config=self.config)` at `:952` — no model-specific cache class. State shapes per linear layer: `conv_states ∈ (batch, conv_dim=8192, conv_kernel_dim=4)` (the 1D causal-conv rolling window) and `recurrent_states ∈ (batch, num_v_heads=32, head_k_dim=128, head_v_dim=128)`. These are the exact tensors tardigrade-db would need to persist if it supports the linear layers.

### [HuggingFace transformers source: configuration_qwen3_next.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/configuration_qwen3_next.py)
- **Authors / Org:** Hugging Face transformers maintainers
- **Type:** open-source project (primary source)
- **Published:** ongoing
- **Accessed:** 2026-05-19
- **Relevance:** medium
- **What this contributed:** Documents `config.layer_types: list[str]` as the canonical way to detect hybridness at construction time. A simple `any(t != "full_attention" for t in config.layer_types)` check at `KnowledgePackStore.__init__` is sufficient for the "refuse cleanly" path.

### [vLLM Issue #25874 — Enable APC for Qwen3-Next](https://github.com/vllm-project/vllm/issues/25874)
- **Authors / Org:** vLLM contributors (open issue thread)
- **Type:** open-source project (issue tracker)
- **Published:** 2025-10 (open)
- **Accessed:** 2026-05-19
- **Relevance:** high
- **What this contributed:** vLLM explicitly disabled Automatic Prefix Caching for Qwen3-Next pending a design that handles linear-attention state. Independent confirmation that this is not a tardigrade-db bug — the production inference ecosystem hit the same wall. Quote from maintainer: prefix caching of recurrent state "is not a solved problem and requires architectural work upstream."

### [vLLM Issue #36493 — prefix cache hit rate ~0% on Qwen3.5 35BA3B](https://github.com/vllm-project/vllm/issues/36493)
- **Authors / Org:** vLLM users
- **Type:** open-source project (issue tracker)
- **Published:** 2026 (open)
- **Accessed:** 2026-05-19
- **Relevance:** medium
- **What this contributed:** Quantifies the production impact — even when prefix caching is *enabled* on Qwen3-Next, the hit rate is effectively zero. Useful evidence that "softmax slice only" (option b in the solution space) probably degrades to "no useful caching" on hybrid models.

### [llama.cpp Issue #20225 — Qwen3.5 full prompt reprocessing](https://github.com/ggml-org/llama.cpp/issues/20225)
- **Authors / Org:** llama.cpp users
- **Type:** open-source project (issue tracker)
- **Published:** 2026 (open)
- **Accessed:** 2026-05-19
- **Relevance:** medium
- **What this contributed:** Confirms the same problem in a different inference engine — llama.cpp forces full prompt re-processing every turn on Qwen3.5, because the linear-attention state cannot be cached at intermediate positions. Cross-engine confirmation that the issue is the architecture, not any one implementation.

### [mlx-lm Issue #980 — prefix cache broken for all hybrid-arch models](https://github.com/ml-explore/mlx-lm/issues/980)
- **Authors / Org:** mlx-lm contributors
- **Type:** open-source project (issue tracker)
- **Published:** 2025-12 (open)
- **Accessed:** 2026-05-19
- **Relevance:** high
- **What this contributed:** Maintainer-level statement that prefix cache reuse is "broken for all hybrid-architecture models (sliding window, SSM/Mamba)" — calls out SSM/Mamba state as fundamentally not splittable. This is the strongest external confirmation that the mathematical obstacle is real, not implementation-side.
- **Quoted:**
  > "Cache reuse is broken for all hybrid-architecture models (sliding-window attention, SSM, Mamba, …). The recurrent state is not splittable at arbitrary token boundaries."

### [Beyond Standard LLMs — Sebastian Raschka](https://magazine.sebastianraschka.com/p/beyond-standard-llms)
- **Authors / Org:** Sebastian Raschka
- **Type:** engineering blog
- **Published:** 2025
- **Accessed:** 2026-05-19
- **Relevance:** medium
- **What this contributed:** Enumeration of modern hybrid-architecture model families — Jamba (AI21), Zamba (Zyphra), RecurrentGemma (Google), Falcon-Mamba (TII), Granite-4 (IBM), MiniMax-Text-01, Hunyuan-T1 (Tencent), Nemotron-H (NVIDIA), Bamba (IBM Research). This is the scope-of-impact answer: tardigrade-db isn't picking one model to refuse; it's picking a *class* of architectures to draw a line around.

### [Qwen3.5: Nobody Agrees on Attention Anymore — Maxime Labonne](https://huggingface.co/blog/mlabonne/qwen35)
- **Authors / Org:** Maxime Labonne
- **Type:** engineering blog
- **Published:** 2025
- **Accessed:** 2026-05-19
- **Relevance:** low
- **What this contributed:** Industry context — the broader trend is that "attention" no longer means one thing, and consumer-facing tooling (HF transformers, vLLM, llama.cpp) is in the middle of adapting cache abstractions to a heterogeneous reality. Useful framing for the scope decision but didn't change any specific technical conclusion.

### [Some Attention is All You Need for Retrieval — Michalak & Abreu](https://arxiv.org/abs/2510.19861)
- **Authors / Org:** Michalak, Abreu (independent / academic)
- **Type:** academic paper
- **Published:** 2025-10
- **Accessed:** 2026-05-19
- **Relevance:** high
- **What this contributed:** Direct corroboration of the Phase 0 spike's headline finding. Ran head-sparsification ablations on **RecurrentGemma-2B/9B and Jamba-Mini-1.6** — the same models in our cohort. Figure 2 shows that ablating all attention heads drops needle-in-a-haystack retrieval accuracy to **0%** across all three models, while the recurrent / SSM layers show no compensatory retrieval behavior even under prompting tricks (Just Read Twice) designed to elicit it. Figure 3 shows 15% of attention heads suffices to preserve near-perfect retrieval while keeping 84% MMLU. The paper's summary phrase — *"retrieval is exclusive to self-attention layers"* — is almost verbatim the conclusion the spike reached. Approaches the question from the opposite direction (zeroing attention, watching collapse) but reaches the same destination. Without this paper, the spike's H1 ≡ H2 result would be a one-off curiosity; with it, the conclusion is published consensus.

### [Functional Component Ablation Reveals Specialization Patterns in Hybrid Language Model Architectures](https://arxiv.org/abs/2603.22473)
- **Authors / Org:** (paper not closely inspected; cite verified via abstract)
- **Type:** academic paper
- **Published:** 2026 (recent)
- **Accessed:** 2026-05-19
- **Relevance:** medium
- **What this contributed:** Complementary asymmetry on Qwen3.5 / Falcon-H1. Table 1 / Table 2 show that removing the linear/SSM stream causes catastrophic perplexity degradation (35,200× / 53×) while removing attention causes much smaller degradation (82× / 3.2×) for *language modeling*. Combined with Michalak & Abreu's retrieval finding, the picture is: **the linear/SSM stream is the language-modeling backbone; attention layers are the retrieval/factuality refinement.** A tardigrade-db that captures only the softmax slice is capturing the part that does fact recall, which is exactly what the storage primitive is for.

### [Griffin: Mixing Gated Linear Recurrences with Local Attention — De et al.](https://arxiv.org/abs/2402.19427)
- **Authors / Org:** Soham De, Samuel L. Smith et al. (Google DeepMind)
- **Type:** academic paper
- **Published:** 2024-02
- **Accessed:** 2026-05-19
- **Relevance:** medium
- **What this contributed:** The architecture paper for RecurrentGemma. Section 6.2 / Figure 6: phonebook lookup succeeds on Griffin only "up to a context length that matches its local attention window size of 1024" — i.e., *the architecture authors themselves attribute recall to the attention layers, not to the recurrent state.* Beyond the 1024-token window, recall must route through RG-LRU state and degrades; below the window, attention does the work. This directly supports the "softmax slice sufficient for short prefills" finding **and** flags the load-bearing long-context caveat (the H1 ≡ H2 result holds at our 20-token prefill but should not extrapolate to multi-thousand-token prefills without re-validation).

### [Revisiting Associative Recall in Modern Recurrent Models — Arora et al.](https://arxiv.org/abs/2508.19029)
- **Authors / Org:** Simran Arora et al.
- **Type:** academic paper
- **Published:** 2025-08
- **Accessed:** 2026-05-19
- **Relevance:** medium
- **What this contributed:** Theoretical and empirical grounding for *why* the LRU recurrent state contributes nothing to fact recall in our spike. Recurrent models *can* in principle solve multi-query associative recall, but require narrow learning-rate windows and width-over-depth (Figs. 3-5). Attention is robust by default; LRU/Mamba/DeltaNet stacks may not have learned the associative-recall circuits during pretraining even when capable of it. Critical caveat (Fig. 6): Gated DeltaNet (Qwen3-Next's mechanism) shows Transformer-level robustness on MQAR — i.e., the softmax-dominance pattern observed on RecurrentGemma's Griffin/LRU may be *less pronounced* on Qwen3-Next. The spike's conclusion likely cross-architecture-portable for the "softmax suffices" direction but the magnitude of the recurrent-state contribution may differ.

### [RecurrentGemma technical report — Botev et al.](https://arxiv.org/abs/2404.07839)
- **Authors / Org:** Aleksandar Botev et al. (Google DeepMind)
- **Type:** academic paper / model report
- **Published:** 2024-04
- **Accessed:** 2026-05-19
- **Relevance:** medium
- **What this contributed:** Confirms the 26-layer 2:1 recurrent:attention layout for the 2B variant and the local-attention window size of 1024 tokens. Aligns with what `config.layers_block_type` reports at runtime.

### [HuggingFace transformers RecurrentGemma source — modeling_recurrent_gemma.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py)
- **Authors / Org:** Hugging Face transformers maintainers
- **Type:** open-source project (primary source)
- **Published:** ongoing
- **Accessed:** 2026-05-19
- **Relevance:** high
- **What this contributed:** **Critical measurement-bug surface.** `RecurrentGemmaRglru.recurrent_states` lives **on the module itself**, not inside the `DynamicCache` object that `past_key_values=` carries around. This means the spike's `zero_recurrent_layers(cache)` function — which iterates `cache.layers` and zeroes tensor attributes — may have done literally nothing to the recurrent state, because the state isn't *in* the cache to begin with. If true, the H1 ≡ H2 byte-identical result is a trivial consequence of zeroing nothing, not evidence of the recurrent state's irrelevance to next-token decisions. The HF docs add: *"If `past_key_values` are used, the user is expected to input only unprocessed `input_ids`"* — meaning the model may re-run the recurrent scan over the full input from zero state regardless of cache contents, which would *also* silently nullify our zero-out.

### [HuggingFace transformers RecurrentGemma docs](https://huggingface.co/docs/transformers/en/model_doc/recurrent_gemma)
- **Authors / Org:** Hugging Face docs team
- **Type:** vendor doc
- **Published:** ongoing
- **Accessed:** 2026-05-19
- **Relevance:** medium
- **What this contributed:** Confirms the "unprocessed `input_ids`" requirement when using `past_key_values`. Aligns with the measurement-bug risk above.

## Synthesis

**The crash is architectural, not a bug.** Qwen3-Next (and the family of hybrid models that share its design: Jamba, Zamba, RecurrentGemma, Falcon-Mamba, Granite-4, MiniMax, Hunyuan-T1, Nemotron-H, IBM Bamba) interleaves softmax-attention layers with linear/SSM layers. Per the [transformers source](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py), the linear layers hold a fundamentally different kind of state — `conv_states` (fixed-shape rolling convolution window) and `recurrent_states` (fixed-shape fast-weight matrix) — accessed via a different API than `.keys` / `.values`. Tardigrade-db's `kp_injector.py` walks `cache.layers[i].keys` uniformly, which works on Qwen3 / Llama-3 / Mistral / Gemma-2 (uniform softmax) and crashes on Qwen3-Next (hybrid).

**The math of "store fact, reinject at runtime" only works for softmax layers.** Per [Yang et al. 2024](https://arxiv.org/abs/2412.06464), the Gated DeltaNet update `S_t = α_t·S_{t-1} + k_t ⊗ delta_t` is non-associative across independent prefixes — the gating decay `α_t` is context-dependent, so a recurrent state captured at the end of an isolated fact-prefill is *not* what the state would have been mid-query. You cannot splice it in the way you can concatenate KV tensors along seq_len. The ecosystem confirms this with maintainer-level statements in [mlx-lm #980](https://github.com/ml-explore/mlx-lm/issues/980) and [vLLM #25874](https://github.com/vllm-project/vllm/issues/25874): every production inference engine that has tried to cache prefixes on hybrid models has either disabled it or reported ~0% hit rate.

**Solution space, ordered by engineering cost and uncertainty:**

1. **Refuse cleanly.** Detect `any(config.layer_types[i] != "full_attention")` at `KnowledgePackStore.__init__` and raise a typed `TardigradeError` with the supported-architecture list. Small, honest, matches the ecosystem. Tardigrade-db remains a softmax-attention KV memory engine.
2. **Softmax slice only.** Capture KV from full-attention layers; re-derive linear-layer state from a fact-text replay at injection. Approximate by construction; quality is empirical. The vLLM #36493 evidence that even partial caching collapses to ~0% hit rate on Qwen3-Next is discouraging for this path.
3. **Full empirical spike.** Persist both codecs, reinject both, measure recall. High uncertainty — the math says it shouldn't work, the model *might* be robust enough to treat injected recurrent state as a soft prior. Weeks of work to validate.
4. **Per-architecture codec adapter.** Refactor `kp_injector` behind a layer-cache codec interface keyed on `type(cache.layers[i])`. Substantial library work; the gain is general (any future architecture plugs in via a new codec). Doesn't resolve the math problem for hybrid models — only makes the failure mode pluggable.

The scope decision (which option to adopt) is pending; see "Downstream uses" below.

### Update after Phase 0 spike + literature check (2026-05-19, same day)

A Phase 0 spike was run on RecurrentGemma-2B (`experiments/v11-hybrid-spike/spike.py`) testing whether the soft-prior hypothesis holds empirically. Result on 20 synthetic facts at ~20-token prefill, greedy decoding:

| Path | R@1 | Notes |
|------|-----|-------|
| Floor (no inject) | 0/20 | Baseline |
| H3 (cache zeroed) | 0/20 | Control — cache structure preserved, contents zeroed |
| H2 (softmax only intact) | 11/20 | 55% — softmax slice alone produced recall |
| H1 (full inject) | 11/20 | 55% — byte-identical to H2 on every fact |
| Ceiling (re-prefill) | 14/20 | 70% upper bound |

The H1 ≡ H2 byte-identical finding pointed at "softmax slice is sufficient, recurrent state contributes nothing observable." A follow-up literature check returned two corroborating signals and one cautionary one:

**Corroborating:**

- [Michalak & Abreu 2025](https://arxiv.org/abs/2510.19861) explicitly tested attention ablation on RecurrentGemma-2B/9B and Jamba — needle-in-a-haystack recall collapses to 0% without attention; recurrent layers cannot compensate even under prompting tricks. Verbatim: *"retrieval is exclusive to self-attention layers."*
- The [Griffin paper itself](https://arxiv.org/abs/2402.19427) section 6.2 attributes phonebook lookup to the local attention layers, only up to the 1024-token window.
- [Arora et al. 2025](https://arxiv.org/abs/2508.19029) provides the theoretical why — LRU recurrent state *can* in principle do associative recall but pretrained Griffin/Mamba/LRU stacks generally haven't learned the circuit. Important Fig. 6 caveat: Gated DeltaNet (Qwen3-Next's mechanism) shows Transformer-level robustness, so the softmax-dominance may be *less pronounced* on Qwen3-Next.

**Cautionary (measurement-bug risk):**

- Per [the HF transformers source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py), `RecurrentGemmaRglru.recurrent_states` lives on the module, not in the `DynamicCache`. The spike's `zero_recurrent_layers()` iterates `cache.layers` and zeroes tensor attributes there — it may have zeroed *nothing relevant*, in which case H1 ≡ H2 is a trivial identity (zeroing nothing produces identical execution to zeroing nothing), not evidence of recurrent-state irrelevance.

**Net conclusion as of 2026-05-19:**

The "softmax slice is sufficient for fact recall in hybrid-attention models" interpretation is **supported by published ablation studies on the exact model family**, independent of whether the spike's mechanism was correctly wired. Phase 1's design (codec captures softmax layers, skips linear/SSM layers) is the right direction. But the *specific experimental evidence* in `experiments/v11-hybrid-spike/` needs a measurement correctness check (verify the zero-out actually zeros what it claims to zero) before being cited as load-bearing evidence.

The 55% recall headline is also lower than would be acceptable for a production retrieval system. Open question: how much of the gap from 55% → 100% is harness limitations (IT model auto-formatting, prompt format weakness), how much is method limitations (true ceiling of softmax-only injection), and how much is the load-bearing long-context regime we haven't tested. Phase 1 must address each.

## Open questions

These could not be resolved from public sources alone and would require running code against an actual Qwen3-Next checkpoint:

- Does reinjecting a DeltaNet `(conv_state, recurrent_state)` captured at position N — then continuing with a new query at position M — produce coherent generation as a soft prior, or pure noise? The math says no for exact composition, but empirical robustness is an open question.
- Does `DynamicCache` serialize cleanly via `torch.save` / `safetensors` when layers are mixed `DynamicLayer` + `LinearAttentionLayer`? Probably yes (both store plain tensors), but untested.
- Are DeltaNet recurrent states numerically stable across BF16 round-trips to disk?
- Does Qwen3.6 (when released) use the same 3:1 layout and the same `LinearAttentionLayer` class, or did HF's cache surface change?

## Downstream uses

- **`CHANGELOG.md` v0.3.2 "Known Limitations" entry** — documents that hybrid linear/standard attention architectures remain incompatible with the KV-capture step; links to this catalog for the why.
- **`python/tardigrade_hooks/chat_template_adapter.py` module docstring** — the "Qwen3.5's hybrid linear/standard attention architecture" note explicitly cites this as a separate library concern from the chat-template fix.
- **`experiments/v11-hybrid-spike/README.md`** — Phase 0 spike result write-up cites this catalog for the architectural background and for the literature corroboration of its R@1 finding.
- **Scope decision (resolved as of 2026-05-19)** — Phase 0 spike + literature check together point at **softmax-slice-only** as the right Phase 1 strategy. Measurement-bug controls and long-context validation must precede the library refactor.
- **Phase 1 plan (pending)** — fresh plan file at `~/.claude/plans/` to be opened, designing the codec interface for softmax-layer-only capture/restore with skip-on-linear-layer behavior. Will cite this catalog and the spike README.
