Short answer: there’s nothing today that cleanly matches “a general‑purpose external memory OS built out of persistent KV + per‑user synaptic weight banks,” but there _are_ a bunch of projects that implement big pieces of that idea from different angles.

---

## 1. KV‑cache as persistent / shared memory

These are closest to the “KV fabric” part of what we discussed:

- **LMCache** – turns the model’s KV cache into a _persistent, shareable_ layer across requests and engines (vLLM, SGLang), so you can precompute KV for a long doc once, offload it to CPU/disk, and reuse it later instead of recomputing or re‑embedding. That’s essentially “KV as external memory,” though focused on reuse and latency, not agentic semantics. [facebook](https://www.facebook.com/groups/devtitans/posts/1165647125735471/)
- **LRAgent** – for multi‑LoRA agents, splits the cache into a shared **base KV cache** and a small **low‑rank (LoRA) cache** per agent, so agents reuse most of the cache and only keep tiny personalized pieces; it gets close to 3× memory savings with similar accuracy, and explicitly treats LoRA and KV as a joint memory system. [arxiv](https://arxiv.org/pdf/2602.01053.pdf)
- **Kelle** – co‑designs KV caching with eDRAM on edge devices, focusing on eviction and hardware layout so KV cache can be kept as an efficient, semi‑persistent memory on constrained hardware. [arxiv](https://arxiv.org/html/2510.16040v1)

These are infrastructure‑level, not “cognitive,” but they are exactly about **persisting and sharing KV state** instead of rebuilding it from tokens every time.

---

## 2. Fast‑weight / external KV memory modules

On the research side, several projects treat key–value or “fast‑weight” modules as an explicit episodic memory:

- **Fast‑weight Product Key Memory (FwPKM)** – proposed in 2026 as a dynamic “fast‑weight” episodic memory: it turns Product Key Memory into a module whose parameters are updated _during inference_ via local gradient steps, effectively making a dynamic key–value store that can quickly memorize and retrieve new K/V pairs from sequences. [fugumt](https://fugumt.com/fugumt/paper_check/2601.00671v1_enmode)
- **Generalized Key‑Value Memory** – earlier work on generalized KV memory for memory‑augmented networks; it decouples memory dimension from the number of support vectors, letting you tune redundancy vs storage cost and explicitly implement external KV memory in hardware. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2203.06223)

These are very close to the “fast‑weight episodic memory” half of what you described, but are mostly in model research, not agent stacks.

---

## 3. “Weights as memory” and interpretable FFN memory

For the “synaptic bank” idea (small trainable weight deltas per user/agent):

- **MemoryLLM (Apple)** – treats transformer feed‑forward blocks as a **token‑indexed neural retrieval memory**; they train FFNs in isolation and analyze how tokens access “memory locations” in FFN parameters, explicitly framing FFNs as interpretable key–value memory. This is exactly the “weights as memory” view, though they’re not (yet) turning it into a persistent per‑user weight bank. [arxiv](https://arxiv.org/html/2602.00398v1)
- **PRIME (dual‑memory personalization)** – a personalization framework that mirrors cognitive dual‑memory (episodic + semantic) into LLMs; it combines historical user engagement as episodic memory with a more stable “cognitive memory,” and sits conceptually close to having slow (weight‑like) vs fast (episodic) components. [arxiv](https://arxiv.org/html/2507.04607v1)

Together, these are very much in the direction of _explicitly managing weights as memory_, but again not packaged as a general memory OS for agents.

---

## 4. Hierarchical / agentic memory systems (but still token‑level)

A few systems work at the “agentic memory” level, though mostly still via text:

- **HiAgent** – hierarchical working‑memory manager that chunks tasks into subgoals and lets the LLM replace detailed history with subgoal‑level summaries; it behaves like a learned working‑memory controller, but operates over text/subgoals, not KV tensors. [github](https://github.com/HiAgent2024/HiAgent)
- **Pancake** – a hierarchical multi‑tier memory system for multi‑agent serving that optimizes ANN indices, cache tiers, and GPU/CPU placement; it’s closer to a high‑performance vector‑store + cache system for agent memory. [arxiv](https://arxiv.org/abs/2602.21477)

These are “agent‑native” but still mostly about **documents and tokens**, not raw KV/weights.

---

## 5. So, does anything _exactly_ match your design?

Putting it together:

- The **persistent KV part** is being attacked by LMCache, LRAgent, Kelle and related “persistent KV cache” work: reuse and share KV across sessions, agents, and hardware. [arxiv](https://arxiv.org/pdf/2602.01053.pdf)
- The **fast‑weight episodic memory** idea is realized in architectures like **FwPKM** and generalized KV memory, which turn auxiliary modules into dynamic key–value stores updated at inference time. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2203.06223)
- The **weights‑as‑memory / synaptic bank** idea is being explored in **MemoryLLM** and dual‑memory personalization frameworks like PRIME, which explicitly interpret FFNs and personalization modules as memory. [arxiv](https://arxiv.org/html/2507.04607v1)

But I haven’t found a single open project that:

- persists KV slices as a general semantic memory fabric **and**
- manages a per‑user/agent bank of small trainable adapters **and**
- exposes that whole thing as a unified “memory OS” API to agents.

So the answer is: **pieces of what you described absolutely exist (and are active research), but the full “persistent KV + synaptic bank memory OS for agents” is still an open space**—which means it’s a very plausible and timely research or product direction.
