# Research/Design Memo: Could We Ingest Files as Tensor Memory in TardigradeDB?

## Why this memo exists

You asked two related questions:

1. **"Could we RAG/vectorize memory files and store them as tensors, like
   TardigradeDB does for internal memory states?"** — i.e., take a text
   document and turn it into the same kind of stored object the engine
   already keeps, instead of doing classical retrieval-augmented generation.

2. **"Would re-forming existing memories as text and re-inserting them give
   us an additional retrieval path?"** — not as a replacement for what we
   have, but a parallel way to find a memory.

This document is a *research and design memo*, not an implementation plan.
Nothing gets built from this directly. The goal is to make sure we both
understand: what's actually going on, what's already been tried by other
people, and what tradeoffs we'd be signing up for.

I've tried to define every term as it comes up. If anything is still unclear,
that's a defect in the memo — flag it.

---

## A glossary so we're using the same words

Before any of this makes sense, we need a shared vocabulary. None of this is
TardigradeDB-specific; it's just standard transformer terminology.

**Token.** A piece of text the model sees as one unit. Roughly a word or a
sub-word ("the", "cat", "##ing"). When the model "reads" a sentence, it sees
a list of tokens.

**Hidden state.** When a token goes into the model, the model computes a
vector of numbers for it at each internal layer. That vector is the model's
"thought" about the token at that point in processing. A vector here is just
a list of numbers — typically 768, 1024, 4096 long depending on the model.

**Attention.** The mechanism the model uses to decide which earlier tokens
matter for the current token. For each token, the model produces three
vectors: a **Query (Q)**, a **Key (K)**, and a **Value (V)**. To figure out
what to do at position N, the model compares position N's Q against every
earlier position's K (a similarity score), then mixes their V's together
weighted by those scores. K and V are what get stored.

**KV cache.** During inference, instead of recomputing K and V for the same
tokens over and over, models save them. This cache is normally throwaway —
it lives only for one generation. **TardigradeDB's whole bet is: don't throw
it away. Persist it. Treat it as memory.**

**Embedding (in classical RAG).** A separate, smaller model takes a chunk of
text and produces *one* fixed-size vector summarising it. Vector databases
store these embeddings and find similar ones by cosine similarity. This is
not the same thing as K and V — K and V come from the actual generative model
during inference, and there's one pair *per token per layer*, not one per
chunk.

**Layer.** Transformers are stacks of attention blocks (a 28-layer model has
28 such blocks). Each layer has its own K and V for each token. Lower layers
tend to capture surface features (syntax, lexical patterns); middle/higher
layers capture more abstract meaning. Picking which layer to store from is a
design choice.

**Position encoding.** Transformers don't naturally know that the 5th token
came before the 50th token. The model injects position info either by adding
a position vector (older models, "absolute" position) or by rotating the K
and Q vectors by an angle that depends on position (newer models, called
"RoPE" — Rotary Position Embedding). RoPE is now standard for Llama, Qwen,
Mistral, etc. **The detail that matters: with RoPE, the K vectors stored in
the cache already have position rotation baked in.** If you want to reuse
that K at a *different* position later, you have to first un-rotate it, then
re-rotate at the new position. This is a real engineering hassle that comes
up a lot in this memo.

**Prefix.** The text at the beginning of the input. "Prefix caching" means
saving K/V for shared prefixes (like a system prompt that's identical across
many requests) and reusing them. Prefix caching is the easy case because
position 0 is position 0 in every request. Reusing K/V somewhere other than
the beginning ("non-prefix" reuse) is the hard case.

**RAG (Retrieval-Augmented Generation).** The classical pattern: when the
user asks a question, look up relevant documents using a vector database,
then paste those documents into the prompt as text and let the model read
them normally. The model has to re-tokenize and re-process those documents
every single time, which is expensive.

**CAG (Cache-Augmented Generation).** Newer alternative: pre-compute the
K and V cache for the *entire knowledge base* once, then stuff that cache
into context at every query. No retrieval step. Works only if the knowledge
base fits in the model's context window.

**kvRAG (KV-prefill-assisted RAG).** A hybrid: pre-compute K/V *per chunk*
once at ingest, store on disk, retrieve and inject the K/V (not the text) at
query time. Skips the re-prefill cost of classical RAG, doesn't need
everything to fit in context like CAG. **This is the family TardigradeDB
already operates in for agent memories, and it's exactly what you're
asking about for files.**

---

## What TardigradeDB actually does today, walked through step by step

Forget the architecture docs for a moment and imagine watching the engine
work in slow motion.

**A memory gets written:**

1. The agent is having a conversation, or processing some input. The
   transformer runs forward through its layers as normal.
2. There's a *hook* sitting on one layer (let's say layer 19 of 28, which is
   roughly the 67% depth heuristic the codebase uses). Every time the model
   produces K/V at that layer, the hook gets a copy.
3. A salience function decides whether this is worth keeping. If yes, the
   hook hands the K vectors (per token, at that one layer) to the engine.
4. The engine quantizes those K vectors down to 4 bits per number (Q4) to
   save space, writes them to an append-only file segment on disk, and
   indexes them.
5. It also stores the original text in a separate `TextStore`, plus metadata
   about who owns the memory, when it was written, what tier it's in, and
   any edges to other memories.

So a "memory" in TardigradeDB is: **a slice of K/V tensors from one specific
transformer layer, plus the source text, plus metadata.** That's the unit.

**A memory gets read:**

1. The agent gets a new query. The same transformer runs forward as normal.
2. The hook intercepts again — but this time during the *prefill* phase,
   before generation starts.
3. The hook takes the query's K vectors at layer 19 and asks the engine:
   "what stored memories have K vectors most similar to these?"
4. The engine runs a similarity search (currently per-token "Top5Avg":
   compare each query token to each stored token, average the top-5 scores
   per memory, rank). It returns the top matches.
5. The matched memories' K and V tensors get **injected directly back into
   the model's attention** at layer 19, as if those tokens had been there all
   along. (This is the part that's genuinely unusual — most memory systems
   stop at step 4 and paste retrieved *text* into the prompt.)

The key thing: there's no separate embedding model. The "search vector" *is*
the model's own internal K vector. There's no re-encoding when you read,
because the K already exists from the live forward pass.

---

## Your question, restated in plain terms

"What if instead of only storing memories that come from live agent
inference, we also fed text files through the model in a fake forward pass
just to capture their K/V, and stored those? That way the same retrieval
machinery would find them."

Yes — that's a clean, well-defined extension. In the literature it's called
**kvRAG** or **KV prefill-assisted RAG**, and a small but real research
community is working on it. TardigradeDB has *most* of the moving parts to
do it already; the missing bits are well-understood.

Let me walk through what would actually happen.

---

## What "ingesting a file as KV memory" would look like, mechanically

Imagine a 10,000-word document about, say, a research paper.

1. **Chunk it.** You can't feed the whole document to the model in one
   forward pass — it might exceed the model's context window, and even if it
   doesn't, you want each chunk to be a separately retrievable memory. So
   split the document into chunks of, let's say, 256 to 1024 tokens each,
   maybe with some overlap so you don't cut a sentence in half. (We don't
   currently have this chunker; we'd need one.)

2. **Pick a layer.** Same 67%-depth layer the rest of the engine uses, at
   least to start. Whether that's right for documents specifically vs. agent
   conversations is an open question — see "what we don't know" below.

3. **For each chunk:**

   a. Tokenize the chunk.

   b. Run a forward pass through the model. (This is the cost: one forward
   pass per chunk. For a 10K-word document split into ~40 chunks, that's
   40 forward passes — not free, but offline / one-time.)

   c. The hook captures K and V at the chosen layer.

   d. Engine writes a memory cell with those K/V tensors, the chunk's source
   text, and metadata pointing back to the document and the chunk's position
   in it.

4. **Wire chunks together.** A document is not a bag of independent chunks;
   chunk 7 follows chunk 6. Add `Supports` edges between consecutive chunks
   so retrieval can walk from one to the next. The edge primitive
   (`add_pack_edge` per `CLAUDE.md`) already exists.

5. **Done.** The document is now a connected graph of memory cells, each
   queryable by the same per-token K-similarity search the engine already
   uses for agent memories.

That's the happy path. There are three real complications we should talk
about before pretending this is easy.

---

## The three real engineering complications

### Complication 1: Position encoding (the RoPE problem)

Remember from the glossary: K vectors come out of the model with the
position rotation already applied. If chunk 7 was tokens 100–127 of the
original document, those K vectors are rotated as if they live at positions
100–127. When we later inject them into a *new* forward pass, where are they
going to live? Probably positions 5–32 of whatever the user is currently
asking. That mismatch matters — the model will compute attention scores
using wrong relative positions, and quality degrades.

The current TardigradeDB workaround: at inject time, un-rotate K from its
original position and re-rotate it at the target position (`position.py`
implements this). It works, but every chunk is fragile to where it lands,
and you pay a small computational tax on every read.

The cleaner fix is called **decoupled position encoding**: store K *without*
position rotation, and apply rotation lazily inside the attention kernel at
read time. The TDD lists this as a design target that hasn't been
implemented. The good news is the research community has now published
working kernels for this (the technique is sometimes called "Lazy-Attention"
or "position-independent KV reuse"). So if we wanted to do file ingest
properly, the time may be right to also tackle decoupled positions.

### Complication 2: Cross-attention loss (the CacheBlend problem)

When chunk 7's K/V was originally computed, the model was looking at chunk
7 *in the context of chunks 1–6*. Specifically, for each token in chunk 7,
the attention mechanism looked back at every token in chunks 1–6 to decide
how to compute that token's hidden state.

When we store just chunk 7's K and V and then later inject them next to a
totally different context (say, the user's query), the K and V are being
used in a setting their hidden state was never computed for. This causes
quality loss — sometimes mild, sometimes serious. A 2024 paper called
**CacheBlend** documents this carefully and proposes a fix: at read time,
*selectively* recompute K/V for a small subset of tokens (about 5–15%) whose
representation is most position-dependent, while reusing the rest from
storage. They report 2.2–3.3× faster time-to-first-token vs. full recompute,
without quality degradation.

A 2025 paper called **KVLink** offers a more elegant solution: train a few
special "link tokens" (think of them as glue tokens) that, when inserted
between cached chunks, restore the missing cross-attention. They report
**+4% QA accuracy** over the previous best non-prefix-reuse method, with
**96% reduction in time-to-first-token**. The cost is a small amount of
fine-tuning — but TardigradeDB already has a SynapticBank for LoRA adapters,
so link tokens could plausibly live there.

So: this problem is real, it's known, and there are at least two viable
fixes published.

### Complication 3: Storage size

A K and V tensor for one chunk, at full FP16 precision, is *not* small. For
a 28-layer model with hidden dimension 1024, a 1000-token chunk costs
roughly 1000 × 28 × 1024 × 2 (K and V) × 2 bytes = 115 MB. That's per
chunk. A 10,000-word document at 40 chunks would be ~4.5 GB raw.

Several mitigations stack:

- **Q4 quantization** (already in TardigradeDB): 4× reduction → ~1 GB.
- **Single-layer storage** (the engine only stores one layer, not all 28):
  another 28× reduction → ~40 MB. This is already what TardigradeDB does.
- **Further KV compression** (CacheGen, KVTC, TurboQuant): another 5–20×
  reduction with negligible quality loss. Not currently in the engine.

So a 10K-word document, in the engine as it stands today, would cost
roughly 40 MB. Not free, but not prohibitive. With CacheGen-class
compression on top, more like 5–10 MB. Compare to a classical RAG
embedding (~3 KB per chunk): the KV approach is ~1000× larger per chunk,
but it skips re-tokenization and re-prefill at read time, which is a real
latency and compute saving when you're querying frequently.

---

## Your follow-up: would re-encoding existing memories as text give us extra retrieval paths?

This is a different question and it's a *good* idea. Let me walk through
why.

### What you're proposing, in plain terms

Today, when a memory gets written, it captures one specific K-vector
representation: whatever the model's hidden state happened to be at the
moment of capture, in whatever framing the agent was working in. If the
agent was answering a coding question and incidentally said something
useful about Paris, the K vectors for that Paris fact are colored by the
"coding question" context.

Later, if a different agent asks "what's the capital of France?" — a query
in a totally different framing — the search has to bridge that gap. The K
vectors for the original capture and the K vectors for this new query come
from the same model, but they were computed in different contexts, so their
similarity is weaker than you'd want.

Your proposal: take the stored memory's *text* (we have it, in `TextStore`)
and re-feed it through the model under several different framings —
"summarize this", "what question does this answer", "rephrase this" — and
capture the K vectors from each pass. Store all of those as parallel
retrieval keys, all pointing to the same underlying memory. Now a query in
*any* of those framings can find the memory.

You called it an "additional neural path" — that's the right intuition.

### Why this works (and what it's called in the literature)

This idea has names in the research community:

- **HyDE (Hypothetical Document Embeddings, 2022)** — at query time, ask the
  model "what would a good answer to this query look like?" and search for
  documents similar to that hypothetical answer instead of the raw query.
  Bridges the question/answer vocabulary gap.

- **HyPE (Hypothetical Prompt Embeddings, 2025)** — the index-time variant.
  *Pre-compute* multiple hypothetical framings per chunk, store them all as
  parallel retrieval keys for the same chunk. **This is essentially what
  you proposed, just at the embedding level rather than the KV level.**

- **Doc2Query** — generate synthetic queries the chunk could answer, append
  them before indexing.

- **SimpleMem** — a recent agent-memory project that does parallel
  multi-view retrieval across semantic / lexical / symbolic indexes.

- **Memory consolidation / hippocampal replay** — the biological analog. In
  brains, memories are reactivated during sleep and re-encoded into
  different cortical assemblies, broadening the cue set that can retrieve
  them later. Given TardigradeDB's biological framing, this analogy is fair
  to lean on.

I couldn't find any published work doing HyPE specifically in KV space.
That would be a novel application of an established idea to a substrate
that's well-suited to it.

### Why it would specifically help TardigradeDB

The codebase already documents a problem this would directly attack. The
"vague-query refinement" results in `CLAUDE.md` show:

- For *specific* queries (vocabulary directly overlaps the memory): R@5 = 100%.
- For *vague* queries: R@5 = 46%.

The diagnosis is "vocabulary overlap, not vagueness — retrieval needs
augmentation for non-specific queries." You can attack that gap on either
end:

- **Query side:** rephrase the query at retrieval time (HyDE). This is what
  the existing centering + cross-encoder reranker does, lifting vague R@5
  to 64%.

- **Index side:** rephrase the *memory* at write time (HyPE / your idea).
  Each memory gets several K-vector views; queries in any framing have a
  shot at hitting one of them.

These two are *stackable*, not redundant. They attack the same gap from
opposite directions.

### How it would slot into the existing code, in plain terms

- **TextStore already keeps the source text.** Re-encoding is just another
  forward pass over that text — the storage layer doesn't change.

- **The retrieval-key strategy abstraction is already pluggable.** The
  `compute_for_save` template method (mentioned in `CLAUDE.md`) is exactly
  where a "MultiViewKey" strategy would fit: instead of producing one K per
  memory, produce N. Or, equivalently, write each view as its own memory
  cell linked to the canonical one by a `Supports` edge. Either works; the
  edge approach is probably cleaner.

- **The KV capture hook needs no changes.** The same `on_generate` path that
  fires during agent inference fires during a re-encoding pass.

- **It runs offline.** Consolidation is a sweep that happens after writes,
  not during them. There's already a sweep module (`sweep.py`) that handles
  importance decay. Multi-view consolidation belongs there.

- **AKL still works.** Tier (Draft / Validated / Core), importance,
  recency — all of them still mean what they meant. Views inherit the
  canonical memory's tier, or the canonical memory's importance is the max
  over its views' access counts. Either policy is sensible.

### Honest tradeoffs

- **Storage multiplier.** N views means N times the storage for retrieval
  keys. The V tensor only needs to be stored once if views share a canonical
  cell (the edge approach), so it's mostly K-vector overhead, which is
  half the total. Q4 still applies. At 4 views per memory, this is
  roughly 4× key storage — manageable.

- **Forward-pass cost at consolidation time.** N forward passes per memory,
  done offline. Real GPU time, but throttle-able: only Validated and Core
  tier get the full N views; Drafts get one (the original).

- **Query drift risk.** This is the one I'd watch most carefully. If the
  framings you generate are too generic ("here's some text"), all memories
  start to look alike in K-space and recall *degrades*. The existing
  `latent-PRF` experiment in the repo hit this failure mode at β=0.3 —
  specific R@5 collapsed from 100% to 30%. Any consolidation strategy needs
  the same kind of careful evaluation that PRF got.

- **Overlap with existing wins.** Centering + cross-encoder rerank already
  delivered +18 percentage points on vague R@5. Multi-view *might* compound
  that further or *might* mostly overlap with it. Only measurement will
  say.

- **Choice of framings is empirical.** Reasonable starting framings:
  "question this memory could answer", "one-sentence summary", "paraphrase
  in different words". The HyPE / Doc2Query literature has decent priors
  for documents; whether they generalize to *agent* memories (which are
  often fragmentary, conversational, or noisy) is open.

---

## How TardigradeDB compares to existing systems

This question has more context if you understand what already exists.
Roughly three families of related work:

### Inference-acceleration KV reuse (LMCache, CacheBlend, PromptCache, KVLink)

These are infrastructure projects whose goal is "make LLM inference
faster". They share the substrate (persistent K/V) but don't care about
agent memory or lifecycle.

- **LMCache** — a production KV cache layer for vLLM and SGLang. Stores K/V
  across GPU/CPU/disk/S3, handles non-prefix reuse, reports 3–10× TTFT
  reduction and up to 15× throughput.

- **CacheBlend** — the cross-attention recompute paper described above.
  2.2–3.3× TTFT reduction, used in RAG settings.

- **PromptCache** (MLSys 2024) — schema-defined "prompt modules" with
  explicit valid position ranges. 60× CPU speedup, 8× GPU speedup on Llama-2.

- **KVLink** (NeurIPS 2025) — trainable link tokens for non-prefix
  cross-attention restoration. +4% QA accuracy, −96% TTFT.

What we'd take from this family if we did file ingest: KVLink's link tokens
(plausibly fits in our SynapticBank LoRA adapter), CacheBlend's selective
recompute (a fallback if link tokens don't pan out), PromptCache's
position-range schema (per-chunk metadata for valid offsets).

### Agent memory systems (Mem0, Letta, MemGPT, ByteRover, Zep)

These do care about agent memory and lifecycle, but they all operate on
*text* — they store documents/strings, retrieve documents/strings, paste
them into context. None of them work in KV space.

The benchmark numbers for 2026 (LongMemEval and LoCoMo are the standard
tests):

- **Letta / MemGPT:** ~83.2% on LongMemEval. Best-in-class for the
  agent-as-OS architecture.
- **Mem0:** 49.0% on LongMemEval, 66.9% on LoCoMo. Mem0g (graph variant)
  reaches 68.4% on LoCoMo.
- **SuperLocalMemory:** 74.8% on LoCoMo, no cloud dependency.
- **Plain GPT-4o-mini agent + prompt tuning baseline:** 74.0% on LoCoMo.

This last number is the uncomfortable one. A vanilla agent with no
specialized memory system already beats Mem0 on LoCoMo. **Any new memory
system needs to credibly clear that bar before its complexity is justified.**
TardigradeDB has not yet been run on either benchmark. Closing that gap —
running it, publishing the number — should arguably come *before* shipping
file ingest or multi-view consolidation, because we don't currently know
where TardigradeDB lands relative to the field.

### Latent-attention / weights-as-memory research (MemArt, MemoryLLM, PRIME)

These explore using attention or model weights themselves as memory, but
they're early-stage research, not productized.

### Where TardigradeDB sits

The differentiator from the agent-memory family is the substrate (tensor-
native vs. text). The differentiator from the inference-acceleration family
is the lifecycle (AKL tiers, edges, governance vs. just caching). Nobody
else combines all three: persistent KV substrate + agent-native lifecycle +
unified API. That's the bet.

File ingest naturally extends that bet: it lets the same substrate hold
external knowledge alongside agent-captured knowledge, with the same
governance applied to both.

---

## What's already wired in the repo we could reuse

(Citing actual files and line numbers from the Phase 1 exploration.)

- **KV capture hook** — `python/tardigrade_hooks/hf_kv_hook.py:139-183`.
  This is the entry point. It already does what we need; we'd just call it
  from a different driver.

- **End-to-end demo of arbitrary-text → KV → store → retrieve** —
  `examples/e2e_demo.py:42-68`. Captures KV on "The capital of France is",
  retrieves with "What is the main city of France". This is essentially a
  one-chunk file ingest already, just dressed up as a demo.

- **Synthetic-fact gibberish recall test** — proves the ingest path works
  on content the model has *never* seen as English. Strongest existing
  evidence that file-derived KV is recoverable.

- **Position handling** — `python/tardigrade_hooks/position.py:63-170`.
  Strategy interface for absolute (GPT-2) and RoPE (Llama, Qwen) position
  encoders. Already model-agnostic; just doesn't yet support the cleaner
  "decoupled" approach.

- **Q4 quantization, segment storage, fsync** — already in place.

- **`mem_read_tokens` direct token API** — bypasses the Python encode/decode
  round-trip; the right primitive for an ingest pipeline.

- **Existing chunking precedent** — `python/tdb_bench/adapters/letta.py:214-232`
  already splits on whitespace at a 4000-char limit. Adapter-internal, not
  engine-level, but a starting point.

- **Edge primitives** — `add_pack_edge` with Supports/Contradicts already
  exposed.

- **SynapticBank** — already exposed to Python with f32↔f16 conversion.
  This is where KVLink-style link tokens would live if we went that route.

## What we'd need to add

In rough order of effort:

1. **A high-level ingest API.** Something like `engine.ingest_text(text,
   owner=…, chunk_size=…)`. Today the caller has to wrap text in a chat
   template and run inference manually — fine for examples, not fine as a
   public API.

2. **A document chunker.** Whitespace-respecting, configurable chunk size,
   optional overlap. Probably ~100 lines of Python.

3. **Cross-chunk edge wiring.** Auto-add `Supports` edges between
   consecutive chunks of the same document.

4. **Source / trust metadata.** Should an ingested file enter as Draft (the
   AKL default) or skip to Validated based on source trust? Policy
   decision.

5. **Decoupled position encoding.** The big one. Listed in the TDD as a
   design target. Not strictly required for v0 of file ingest (we could
   pay the unrotate/re-rotate cost), but cleaner long-term.

6. **Cross-attention restoration.** Either CacheBlend-style selective
   recompute or KVLink-style link tokens. Required if we want quality to
   match (or beat) classical RAG.

7. **Multi-view consolidation** (your follow-up idea). Probably best as a
   second feature after v0 of file ingest, because it's orthogonal and the
   evaluation is more involved.

---

## What we *don't know* — open questions worth answering before any code

1. **What's TardigradeDB's number on LongMemEval and LoCoMo?** This should
   be measured first. Without it, we're shipping features without knowing
   our baseline.

2. **What chunk size minimizes cross-attention loss?** Reproduce
   CacheBlend's measurement on the existing Qwen3-0.6B / 100-cell setup
   before committing to a number.

3. **Does the 67% layer heuristic still hold for documents?** The heuristic
   was chosen for agent memory, where the input is short conversational
   exchanges. Documents have different statistics. Sweep layers on the
   synthetic-fact corpus to verify.

4. **Does mean-centering refinement help on file-derived KV?** The +18pp
   vague R@5 win from centering was measured on agent memories. Files have
   different K-distribution properties; the result may not transfer.

5. **Is multi-view consolidation additive or overlapping with the existing
   refinement wins?** Centering + reranker already lifted vague R@5 from
   46% to 64%. Multi-view might compound that or might saturate it. Only
   measurement says.

6. **For multi-view: which framings are right for *agent* memories?**
   HyPE/Doc2Query priors are for documents. Agent memories are often
   fragmentary, conversational, noisy — different beast.

---

## My honest recommendation

Before building either feature:

1. **Run TardigradeDB on LongMemEval and LoCoMo.** Establish a baseline
   number. Without this, any feature we ship is hard to justify.

2. **Decide between two roads.** Both are good; they answer different
   questions:

   - **Road A: File ingest first.** Bigger capability expansion. Turns
     TardigradeDB from "agent memory" into "agent + document memory" using
     the same substrate. Engineering effort is higher (chunker, edges,
     ideally decoupled positions, ideally cross-attention restoration).

   - **Road B: Multi-view consolidation first.** Higher-leverage near-term
     win on the existing 100-cell corpus. Directly attacks the documented
     vague-query gap. Pure additive feature — doesn't change any existing
     behavior. Smaller engineering surface (a sweep extension + a new
     retrieval-key strategy).

   I'd lean toward Road B as the next step, *if* the LongMemEval/LoCoMo
   baseline shows a vague-query weakness similar to what the 100-cell
   corpus shows. If those benchmarks are fine and the gap is somewhere
   else, the priority changes.

3. **Treat the position-encoding rework as its own project.** Decoupled
   position encoding is in the TDD as a design target for good reason —
   it cleans up a lot of stuff including but not limited to file ingest.
   Worth tackling on its own merits, with its own ATDD plan, rather than
   bundled into a feature.

This is a memo, not a plan. Nothing builds from this directly. The next
step, if you want one, is choosing among Road A, Road B, or "benchmark
first, then decide" — and then I'd do a proper implementation plan for
whichever.

---

## Sources

### File ingest / KV reuse infrastructure

- [LMCache repo](https://github.com/lmcache/lmcache) — production KV cache
  layer for vLLM and SGLang
- [LMCache tech report (arxiv 2510.09665)](https://arxiv.org/abs/2510.09665)
- [CacheBlend (arxiv 2405.16444)](https://arxiv.org/abs/2405.16444) — the
  cross-attention recompute paper
- [CacheGen (Stanford SIGCOMM 2024)](https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final1571-acmpaginated.pdf)
  — KV compression and streaming
- [PromptCache (arxiv 2311.04934)](https://arxiv.org/abs/2311.04934) —
  schema-defined modular attention reuse
- [PromptCache MLSys 2024 PDF](https://proceedings.mlsys.org/paper_files/paper/2024/file/a66caa1703fe34705a4368c3014c1966-Paper-Conference.pdf)
- [KVLink (arxiv 2502.16002)](https://arxiv.org/abs/2502.16002) —
  trainable link tokens for non-prefix KV reuse
- [KVLink repo (UCSB-NLP-Chang)](https://github.com/UCSB-NLP-Chang/KVLink)
- [Don't Do RAG / CAG (arxiv 2412.15605)](https://arxiv.org/html/2412.15605v1)
  — preload entire knowledge into KV
- [DroidSpeak (arxiv 2411.02820)](https://arxiv.org/html/2411.02820v4) —
  cross-LLM KV cache sharing
- [K-V Cache Alignment (arxiv 2601.06123)](https://www.arxiv.org/pdf/2601.06123)
  — direct skill transfer between models via aligned KV
- [KV-Latent (arxiv 2507.11273)](https://arxiv.org/html/2507.11273) —
  latent-space KV reduction
- [TurboQuant overview](https://www.mindstudio.ai/blog/what-is-google-turboquant-kv-cache-compression)
  — 5× KV compression to 3-bit
- [KV Cache Transform Coding (arxiv 2511.01815)](https://arxiv.org/pdf/2511.01815)
  — up to 20× compression
- [NVIDIA kvpress repo](https://github.com/NVIDIA/kvpress)

### Multi-view / index-time augmentation

- [HyDE docs (Haystack)](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)
- [HyPE: Hypothetical Prompt Embeddings paper](https://www.researchgate.net/publication/389032824_Bridging_the_Question-Answer_Gap_in_Retrieval-Augmented_Generation_Hypothetical_Prompt_Embeddings)
- [SimpleMem repo](https://github.com/aiming-lab/SimpleMem) — parallel
  multi-view agent memory retrieval
- [Memory Retrieval and Consolidation through Function Tokens (arxiv 2510.08203)](https://arxiv.org/abs/2510.08203)
- [FOREVER: Forgetting Curve-Inspired Memory Replay (arxiv 2601.03938)](https://arxiv.org/html/2601.03938)
- [IBM: Memory augmentation for LLMs](https://research.ibm.com/blog/memory-augmented-LLMs)

### Agent memory benchmarks (2026)

- [State of AI Agent Memory 2026 (Mem0 blog)](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [Letta: Benchmarking AI Agent Memory](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- [5 AI Agent Memory Systems Compared (DEV.to, 2026)](https://dev.to/varun_pratapbhardwaj_b13/5-ai-agent-memory-systems-compared-mem0-zep-letta-supermemory-superlocalmemory-2026-benchmark-59p3)
- [The State of AI Agent Memory in 2026 (DEV.to)](https://dev.to/vektor_memory_43f51a32376/the-state-of-ai-agent-memory-in-2026-what-the-research-actually-shows-3aja)
- [Best LLM for RAG 2026](https://pricepertoken.com/leaderboards/rag)
- [RAG Benchmarks Leaderboard 2026](https://awesomeagents.ai/leaderboards/rag-benchmarks-leaderboard/)
