# Open Design Questions

Unresolved architectural decisions that need empirical validation or design choices before implementation. Each question includes the trade-offs and what evidence would resolve it.

---

## Q1: Full KV Storage Cost at Scale

**Context:** The KV injection validation proved that full per-token KV cache is necessary for injection (mean-pooled is broken). But full KV is 40-200x larger than mean-pooled vectors.

### The Numbers

Per single memory (~20 tokens), FP32:

| Model | KV heads | head_dim | Layers | Per token | 20 tokens | 20 tok Q4 |
|-------|----------|----------|--------|-----------|-----------|-----------|
| GPT-2 (117M) | 12 | 64 | 12 | 72 KB | 1.4 MB | 360 KB |
| Llama 3 8B (GQA) | 8 | 128 | 32 | 256 KB | 5.0 MB | 1.25 MB |
| Llama 3 70B (GQA) | 8 | 128 | 80 | 640 KB | 12.5 MB | 3.1 MB |

At scale (Llama 8B, Q4 quantized):

| Memories | Tokens each | Total Q4 storage |
|----------|-------------|-----------------|
| 100 | 20 | ~125 MB |
| 100 | 100 | ~625 MB |
| 1,000 | 20 | ~1.2 GB |
| 1,000 | 100 | ~6.2 GB |

GQA helps significantly — Llama 3's 8 KV heads (vs 32 Q heads) means KV is 4x smaller than `d_model` would suggest.

### Candidate Approaches

**A. Store everything, rely on governance to bound growth.**
AKL already decays importance over time. Old memories fall below eviction threshold and get swept. Storage stays bounded by governance policy, not accumulation. Simple, no new code needed — just configure sweep aggressiveness.

*Trade-off:* Simple but potentially wasteful. Agent with high write volume accumulates GB before sweep catches up.

**B. Selective layer storage.**
Not all layers contribute equally to recall. Store KV from only the most informative layers (e.g., middle layers 8-24 in a 32-layer model). Cuts storage proportionally.

*Trade-off:* Needs empirical evidence — which layers matter for injection quality? May vary by model and memory type.

**C. Token compression (learned summary tokens).**
The validation showed naive selective token injection fails (salience heuristic picks wrong tokens). But learned compression — e.g., attention-pooling over tokens to produce fewer "summary tokens" — might work. This is what MemArt does.

*Trade-off:* Most complex to implement. Requires training or a fixed compression strategy. Quality unknown.

**D. Hybrid: mean-pooled index + lazy full-KV load.**
Store mean-pooled vectors for fast retrieval (search index, tiny). Store full KV separately on disk (cold storage, large). Only load full KV for the top-k retrieval hits at injection time. Most memories stay cold — only the relevant ones get loaded.

*Trade-off:* Maps naturally to TardigradeDB's existing architecture (SLB/Vamana = fast index, BlockPool = cold storage). Adds a two-tier storage model. Read path becomes: query index → get cell IDs → load full KV from disk → inject.

### What Would Resolve This

1. Measure which layers matter for injection quality (layer ablation study)
2. Measure real-world agent memory accumulation rates (how many memories per day?)
3. Benchmark lazy-load latency: how fast can BlockPool serve a full KV read for top-5 results?
4. Test governance-bounded storage: does AKL sweep keep storage under control with realistic write patterns?

---

## Q2: RoPE Position Re-Encoding for Modern Models

**Context:** The KV injection validation used GPT-2, which has absolute learned position embeddings. Position offset (adjusting `position_ids`) is sufficient. Modern models (Llama, Qwen, Mistral) use RoPE — position is baked INTO the K vectors via rotation. Injecting historical K vectors at new positions requires unrotating and re-rotating.

### The Problem

```
Historical K at position 3:   K_3 = RoPE(W_K · x_3, pos=3)
Injection target position 15: need K' = RoPE(W_K · x_3, pos=15)

Required: unrotate(K_3, pos=3) → K_raw → rotate(K_raw, pos=15) → K'
```

### Open Sub-Questions

- Does RoPE unrotation + re-rotation preserve attention quality? (rounding errors accumulate through two rotations)
- For models with extrapolated RoPE (e.g., Llama 3's 128K context via RoPE scaling), does re-encoding at a very different position distort retrieval?
- Can we skip re-encoding for models that use ALiBi (Falcon, MPT) since ALiBi applies position bias at attention time, not in K vectors?

### What Would Resolve This

1. Implement `RoPEPositionEncoder` with unrotate/re-rotate
2. Run the KV injection validation experiment with Llama 3.2:3b
3. Compare injection quality with and without position re-encoding
4. Measure attention score distribution — are re-encoded K vectors in-distribution?

---

## Q3: GQA Head Count Mismatch

**Context:** Llama 2+ uses Grouped Query Attention — fewer KV heads (8) than Q heads (32). The injection pipeline assumes `num_kv_heads` for reshaping. Does the head count mismatch cause issues when injecting into models that broadcast KV heads across Q head groups?

### What Would Resolve This

1. Test injection with a GQA model (Llama 3.2:3b has 8 KV heads, 32 Q heads)
2. Verify that HuggingFace's `DynamicCache` handles the broadcast correctly when injected KV has fewer heads

---

## Q4: Sliding Window Attention Compatibility

**Context:** Mistral and some Qwen variants use sliding window attention — the model only attends to the last N tokens. Injected KV blocks that fall outside the window are invisible to the model.

### Open Sub-Questions

- Is this a hard blocker, or can injected KV be placed inside the window?
- If the window is 4096 tokens and the current prompt is 100 tokens, that leaves 3996 positions for injected memory — is this enough?
- Does the window apply to `past_key_values` or only to the current input?

---

## Q5: Cross-Model KV Portability

**Context:** Can memories stored from one model be injected into a different model? E.g., store KV from Llama 8B, inject into Llama 70B.

### Why It Probably Doesn't Work

Different models have different `W_K`/`W_V` projection matrices. K/V vectors from model A live in a completely different vector space than model B's attention expects. Even models from the same family but different sizes have different head dimensions and layer counts.

### Why It Might Work (Partially)

Models fine-tuned from the same base (e.g., Llama 3 8B Instruct vs Llama 3 8B Base) share `W_K`/`W_V` weights. KV portability within the same base model might be viable.

### What Would Resolve This

1. Store from Llama 3 8B Base, inject into Llama 3 8B Instruct — same architecture, different fine-tune
2. Store from Llama 3 8B, inject into Llama 3 70B — same family, different size
3. Measure attention score distribution for injected cross-model KV
