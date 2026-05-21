# Research: Learnings for TardigradeDB from production KV-cache projects (2025–2026)

- **Slug:** `2026-05-21-production-kv-cache-learnings`
- **Date:** 2026-05-21
- **Status:** complete
- **Triggered by:** User question "Is there anything tardigrade can learn today from production-grade kv cache projects?" raised in chat on 2026-05-21.
- **Informed:** Plan memo at `~/.claude/plans/is-there-anything-dreamy-pike.md`. Recommendations not yet picked up — each top-3 item would land as its own ATDD-first plan when scheduled.

## Question

What concrete, adoptable ideas can TardigradeDB take from 2025–2026 production-grade KV-cache projects (LMCache, vLLM PagedAttention, SGLang RadixAttention, Mooncake, CacheGen/CacheBlend, NVIDIA KVTC and Dynamo, DistServe), given that TardigradeDB is a persistent KV-native *memory engine* (sub-ms p99 at 5K cells, Q4 quant, Vamana ANN, per-token Top5Avg retrieval, 751 B/cell) and not a serving prefix cache? For each candidate technique: adopt, reject, or defer — and why?

## Sources

### [LMCache](https://github.com/LMCache/LMCache)
- **Authors / Org:** LMCache Lab (project lead Yihua Cheng et al., University of Chicago + community contributors). De-facto KV layer for vLLM / SGLang / NVIDIA Dynamo / llm-d / KServe.
- **Type:** open-source project + technical report
- **Published:** project ongoing; consolidated tech report 2025-10-13
- **Accessed:** 2026-05-21
- **Relevance:** high
- **What this contributed:** The bandwidth-vs-message-size curve (4 GB/s @ 64 KB → 49 GB/s @ 16 MB, a 12× gap) is the empirical anchor for the recommendation to bench TardigradeDB's per-cell append bandwidth and tune `BufferConfig::max_batch_size`. Also the source for the refcounting-instead-of-mutex pattern and the layer-wise pipelining technique behind recommendation #2 (refcounted read handles) and the `mem_read_multi_layer` adopt note.

### [LMCache technical report (arXiv 2510.09665v2)](https://arxiv.org/html/2510.09665v2)
- **Authors / Org:** Yihua Cheng, Kuntai Du, Junchen Jiang, et al. (University of Chicago, LMCache Lab)
- **Type:** academic paper / tech report
- **Published:** 2025-10
- **Accessed:** 2026-05-21
- **Relevance:** high
- **What this contributed:** Published workload-relative speedup numbers (1.9–8.1× TTFT for CPU offload; 2.3–14× throughput). Used to inform the "we should keep publishing absolute latency, they publish relative speedup — different axes, different positioning" framing in the synthesis.

### [CacheBlend (EuroSys '25)](https://arxiv.org/pdf/2405.16444)
- **Authors / Org:** Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, Junchen Jiang (University of Chicago)
- **Type:** academic paper
- **Published:** 2025 (EuroSys); arXiv preprint 2024-05
- **Accessed:** 2026-05-21
- **Relevance:** medium
- **What this contributed:** Selective-recomputation-of-critical-tokens idea (recompute attention only for ~15% of tokens to fuse non-prefix KV chunks). Used as the conceptual reference for "stealing the framing" — adjacent to our cross-encoder rerank over text-bearing candidates in vague-query refinement.

### [CacheGen (SIGCOMM '24)](https://arxiv.org/abs/2310.07240)
- **Authors / Org:** Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, Junchen Jiang (University of Chicago, Microsoft, Stanford)
- **Type:** academic paper
- **Published:** 2024-08 (SIGCOMM)
- **Accessed:** 2026-05-21
- **Relevance:** high — drives top-3 recommendation #1
- **What this contributed:** Variable-bitrate per-layer/per-channel quantization with 3.5–4.3× compression and quality preservation. This is the direct source for the #1 recommendation (variable-bitrate quantization for Validated/Core writes); the "calibrate once per model, store a per-channel bit-budget table" pattern is theirs.

### [vLLM PagedAttention — Kwon dissertation (Berkeley TR EECS-2025-192)](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.html)
- **Authors / Org:** Woosuk Kwon (UC Berkeley)
- **Type:** academic dissertation (canonical reference for PagedAttention)
- **Published:** 2025
- **Accessed:** 2026-05-21
- **Relevance:** medium
- **What this contributed:** The empirical "<4% fragmentation at fixed block size" datum behind the "uniform physical cell size in the arena" adopt note. Also the source for rejecting per-sequence page tables (we have owners, not sequences).

### [vLLM RFC #16016: Cache Salting](https://github.com/vllm-project/vllm/issues/16016) + [PR #15297: SHA-256 default hashing](https://github.com/vllm-project/vllm/pull/15297)
- **Authors / Org:** vLLM project (community RFC and merged PR)
- **Type:** open-source project artifacts (RFC + PR)
- **Published:** 2025
- **Accessed:** 2026-05-21
- **Relevance:** high
- **What this contributed:** The `cache_salt` pattern (per-tenant prefix-cache isolation by salting the content hash) is the source for the "owner-salted content-addressed packs" adopt item — architectural cross-owner leakage prevention rather than convention-based.

### [SGLang RadixAttention (LMSYS blog 2024)](https://lmsys.org/blog/2024-01-17-sglang/)
- **Authors / Org:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al. (LMSYS / Stanford / Berkeley)
- **Type:** engineering blog (project announcement)
- **Published:** 2024-01
- **Accessed:** 2026-05-21
- **Relevance:** medium
- **What this contributed:** "Turning radix prefix cache off costs 70-90% throughput" — used as the calibration for *why* the technique works for serving caches but *doesn't* translate to a latent-similarity memory engine. Also seeded the narrower "radix-tree dedup in `FileIngestor`" idea: prefix matching belongs at ingest, not at retrieval.

### [Mooncake (arXiv 2407.00079 / ACM ToS 2025)](https://arxiv.org/abs/2407.00079)
- **Authors / Org:** Ruoyu Qin, Zheming Li, Weiran He, et al. (Moonshot AI + Tsinghua)
- **Type:** academic paper (production system writeup)
- **Published:** 2024-07 arXiv, 2025 ACM ToS
- **Accessed:** 2026-05-21
- **Relevance:** low–medium (future-horizon for TardigradeDB)
- **What this contributed:** KVCache-aware scheduling across a fleet (100B+ tokens/day). Cited in the synthesis as the basis for the "future-horizon multi-node memory sharing" reject — useful framing for why we are *not* doing that today.

### [NVIDIA KVTC (arXiv 2511.01815, ICLR '26)](https://arxiv.org/abs/2511.01815) + [GitHub](https://github.com/OnlyTerp/kvtc)
- **Authors / Org:** NVIDIA Research team (full author list at paper; project repo `OnlyTerp/kvtc`)
- **Type:** academic paper + open-source reference implementation
- **Published:** 2025-11 arXiv, accepted ICLR 2026
- **Accessed:** 2026-05-21
- **Relevance:** high — drives honourable-mention recommendation
- **What this contributed:** PCA-decorrelate KV on a learned orthonormal basis, DP-based bit allocation, GPU-accelerated DEFLATE decode via nvCOMP — 20× compression (up to 40× in some cases) with preserved reasoning/long-context accuracy. Direct source for the "Frozen" archival-tier honourable mention. Open-source implementation is what brings the risk floor down enough to seriously consider this.

### [NVIDIA Dynamo + LMCache integration (LMCache blog, 2025-09-07)](https://blog.lmcache.ai/en/2025/09/07/nvidia-dynamo-lmcache-accelerating-the-future-of-llm-inference/)
- **Authors / Org:** LMCache team (joint announcement with NVIDIA Dynamo)
- **Type:** engineering blog (vendor / project)
- **Published:** 2025-09-07
- **Accessed:** 2026-05-21
- **Relevance:** medium
- **What this contributed:** The "session reloads hours/days later after GPU restart" vocabulary used by Dynamo's long-term persistence tier. Source for the "frame snapshot/checkpoint as `session_resume`" naming/API recommendation.

### [DistServe (OSDI '24)](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)
- **Authors / Org:** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang (Peking University, UCSD, StepFun)
- **Type:** academic paper (OSDI)
- **Published:** 2024 (OSDI)
- **Accessed:** 2026-05-21
- **Relevance:** medium
- **What this contributed:** Confirms the "decoupled position encoding" design target listed in `docs/technical/tdd.md` is actively researched — the literature to mine when picking up that roadmap item. Doesn't change today's design but anchors it.

### [Agent Memory Below the Prompt (arXiv 2603.04428)](https://arxiv.org/abs/2603.04428)
- **Authors / Org:** Author identities to be verified at the source; abstract attributes the work to an edge-AI research group (claim made by the agent that surfaced this source; primary verification still owed).
- **Type:** academic paper (preprint)
- **Published:** 2026-02
- **Accessed:** 2026-05-21
- **Relevance:** high — drives top-3 recommendation #3
- **What this contributed:** Closest external work to TardigradeDB's product pitch — persists per-agent KV cache to disk in 4-bit quantized form, reloads directly into the attention layer, claims up to 136× TTFT reduction vs re-prefill on edge. The 136× figure is the proposed external anchor for Track A positioning measurement. **Caveat:** the agent that surfaced this paper did not produce a verified author list; before this paper is cited in a positioning doc, the author list and the 136× claim should be re-read directly from the source.

## Synthesis

The full distilled answer lives in the consuming artifact (`~/.claude/plans/is-there-anything-dreamy-pike.md`). Condensed:

- **Adopt (concrete):** variable-bitrate quantization for cold-tier writes (CacheGen); refcounted read handles on `TextStore` (LMCache); layer-wise pipelining inside `mem_read_multi_layer` (LMCache); uniform physical cell size in the mmap arena (PagedAttention); owner-salted content-addressed packs (vLLM cache-salt RFC); radix-tree dedup at *ingest* (SGLang inspiration, not retrieval); write-side bandwidth bench at varied batch sizes (LMCache curve as reference).
- **Adopt (vocabulary / API):** frame an existing checkpoint API as `session_resume` (Dynamo + LMCache); expose `PUT(hash) / GET(hash) / EVICT` tiered-backend surface on the vLLM connector so production stacks can adopt TardigradeDB as a disk tier without rearchitecting.
- **Defer / honourable mention:** NVIDIA KVTC PCA+DEFLATE for a future "Frozen" archival tier (20× compression, never on hot path). Open-source reference implementation lowers risk meaningfully.
- **Defer (architectural):** FP8 promotion tier coupled to Core governance (FP8 hot cells, Q4 elsewhere) — wants a design memo before code.
- **Reject:** per-sequence page tables, radix-tree primary retrieval, remote tier in the hot path, LRU-only governance, stacking transfer compression over Q4, multi-node disaggregation today.

**Top-3 to pick up next**, ranked by value × effort: (1) variable-bitrate quantization (CacheGen); (2) refcounted read handles (LMCache); (3) head-to-head bench against the Feb 2026 edge-agent KV paper, with the caveat that its provenance must be re-verified before citation.

## Downstream uses

- Plan memo: [`~/.claude/plans/is-there-anything-dreamy-pike.md`](file:///home/flagrare/.claude/plans/is-there-anything-dreamy-pike.md) — the actionable synthesis the user reads.
- **Design archive: [`docs/refs/frozen-tier-design.md`](../refs/frozen-tier-design.md)** — the Frozen-tier design (state machine + on-demand-calibrated codec) extracted from this research and committed as an in-repo design reference. Scheduled-but-deferred work; rationale preserved before the code lands.
- When any other top-3 recommendation (CacheGen-style variable-bitrate quant; refcounted read handles; head-to-head bench vs. edge-agent KV paper) is picked up, its plan and the resulting code/doc changes should back-link to this catalog entry.
