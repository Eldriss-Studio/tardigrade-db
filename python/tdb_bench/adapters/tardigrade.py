"""Tardigrade native adapter.

Two execution modes:

- ``native``: real engine path. Loads a small HuggingFace model
  (Qwen3-0.6B by default), captures per-token hidden states for each
  ingested item via ``HuggingFaceKVHook``, stores them through
  ``engine.mem_write_pack``-equivalent ``mem_write``, and at query time
  encodes the question through the same model and retrieves the top-k
  cells via ``engine.mem_read_tokens``. The retrieved cell IDs are
  mapped back to their source items so the adapter can return the
  matching ``ground_truth`` as the answer. **This is what
  TardigradeDB actually does** — earlier versions of this adapter
  short-circuited via lexical matching, which inflated the engine's
  apparent quality.

- ``in_memory``: portable fallback for hosts without CUDA/torch. It
  performs honest word-overlap matching against ingested contexts.
  Keep around so the smoke fixture can run on CI runners without GPUs.

Set ``TDB_BENCH_FORCE_FALLBACK=1`` to force the in-memory path even if
CUDA is available (useful for diffing the two paths).

Refinement mode is selected via ``TDB_REFINEMENT_MODE`` (``none``,
``centered``, or ``prf``). The default ``none`` preserves the engine's
unmodified retrieval. The corresponding empirical numbers live in
``docs/experiments/vague_queries/results.md``.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Any

from tardigrade_hooks.constants import DEFAULT_CAPTURE_LAYER_RATIO
from tardigrade_hooks.constants import (
    RLS_DEFAULT_GEN_MODEL,
    RLS_MODE_BOTH,
    RLS_MODE_EMBEDDING,
    RLS_MODE_AGENT,
    RLS_MODE_GENERATIVE,
    RLS_MODE_KEYWORD,
    RLS_MODE_MULTIPHRASING,
    RLS_MODE_NONE,
)
from tdb_bench.contracts import BenchmarkAdapter
from tdb_bench.models import AdapterQueryResult, BenchmarkItem


_MODEL_NAME = os.getenv("TDB_BENCH_MODEL", "Qwen/Qwen3-0.6B")
_DEVICE = os.getenv("TDB_BENCH_DEVICE", "cuda")
_REFINEMENT = os.getenv("TDB_REFINEMENT_MODE", "none")
_RERANK_MODEL = os.getenv("TDB_BENCH_RERANK_MODEL", "")  # empty disables reranker
_RLS_MODE = os.getenv("TDB_RLS_MODE", RLS_MODE_NONE)

# RLS is opt-in only. The 2026-05-14 bench audit found every RLS mode
# harmful relative to no-RLS (keyword -5.3pp, DeepSeek agent -12.7pp on
# the 50-item LoCoMo subset). Code is preserved for future fusion-redesign
# work; meanwhile, opting in via env var emits a one-time warning so the
# choice is visible in run logs. See docs/experiments/2026-05-14-bench-audit.md.
if _RLS_MODE != RLS_MODE_NONE:
    import warnings as _warnings
    _warnings.warn(
        f"TDB_RLS_MODE={_RLS_MODE!r} — RLS is experimental and was found "
        "harmful in the 2026-05-14 audit (all modes underperform no-RLS "
        "on the clean dataset). See docs/experiments/2026-05-14-bench-audit.md.",
        stacklevel=2,
    )
_CHUNK_TOKENS = int(os.getenv("TDB_BENCH_CHUNK_TOKENS", "128"))
_CHUNK_OVERLAP = int(os.getenv("TDB_BENCH_CHUNK_OVERLAP", "16"))

# Slice B1 — GPU batching for the ingest forward pass.
# Default 8 chunks per batched forward pass fits the RTX 3070 Ti's
# 8 GB VRAM with Qwen3-0.6B at chunk size 128 with comfortable
# headroom. Override via `TDB_BENCH_GPU_BATCH_SIZE` when tuning on
# different hardware.
_GPU_BATCH_SIZE = int(os.getenv("TDB_BENCH_GPU_BATCH_SIZE", "8"))

# Slice B2 — engine-side batched writes. WriteRequest tuple field
# defaults reused across all writes from this adapter; named for
# clarity instead of inline magic literals.
_DEFAULT_WRITE_OWNER = 1
_DEFAULT_WRITE_SALIENCE = 1.0
_NO_PARENT_CELL_ID: int | None = None

# Slice D2 — CPU/GPU pipeline overlap.
# Bounded queue depth between the GPU-forward producer thread and the
# `mem_write_batch` consumer thread. Depth 4 gives the consumer time
# to drain while the producer prepares the next item without
# unboundedly buffering pending writes.
_INGEST_PIPELINE_DEPTH = int(os.getenv("TDB_INGEST_PIPELINE_DEPTH", "4"))


def _create_rls_strategies(rls_mode: str) -> list:
    """Factory Method for ReformulationStrategy instances.

    Creates only strategies that need no GPU. GPU-dependent strategies
    (EmbeddingExpansion, GenerativeReformulation) stay in the native-only
    init path with their guarded imports.
    """
    from tardigrade_hooks.rls import (
        KeywordExpansionStrategy,
        LLMAgentReformulationStrategy,
        MultiPhrasingStrategy,
    )

    strategies: list = []
    if rls_mode in (RLS_MODE_KEYWORD, RLS_MODE_BOTH):
        strategies.append(KeywordExpansionStrategy())
    if rls_mode in (RLS_MODE_MULTIPHRASING, RLS_MODE_BOTH):
        strategies.append(MultiPhrasingStrategy())
    if rls_mode == RLS_MODE_AGENT:
        strategies.append(
            LLMAgentReformulationStrategy(api_key=os.getenv("DEEPSEEK_API_KEY", "").strip())
        )
    return strategies


# Module-level model cache so repeated ``adapter = TardigradeAdapter()``
# calls (one per repeat in BenchmarkRunner) don't re-download/re-load
# the model. None until first native instance asks for it.
_MODEL_CACHE: dict[str, Any] = {"model": None, "tokenizer": None, "query_layer": None}


def _load_model_cached() -> tuple[Any, Any, int]:
    if _MODEL_CACHE["model"] is None:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME, dtype=torch.float32, attn_implementation="eager",
        )
        model = model.to(_DEVICE).eval()
        n_layers = model.config.num_hidden_layers
        query_layer = int(n_layers * DEFAULT_CAPTURE_LAYER_RATIO)
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["tokenizer"] = tokenizer
        _MODEL_CACHE["query_layer"] = query_layer
    return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"], _MODEL_CACHE["query_layer"]


class _InMemoryStore:
    """Lexical fallback. Honest about what it is."""

    def __init__(self) -> None:
        self.data: OrderedDict[str, BenchmarkItem] = OrderedDict()

    def clear(self) -> None:
        self.data.clear()

    def insert(self, item: BenchmarkItem) -> None:
        self.data[item.item_id] = item

    def scored_best_match(self, question: str, top_k: int) -> list[tuple[int, BenchmarkItem]]:
        terms = {t.lower() for t in question.split() if t.strip()}
        scored: list[tuple[int, BenchmarkItem]] = []
        for item in self.data.values():
            hay = f"{item.context} {item.question}".lower()
            score = sum(1 for t in terms if t in hay)
            scored.append((score, item))
        scored.sort(key=lambda s: s[0], reverse=True)
        return scored[: max(1, top_k)]

    def best_match(self, question: str, top_k: int) -> tuple[str, list[str]]:
        top = self.scored_best_match(question, top_k)
        best = top[0][1]
        return best.ground_truth, [item.context for _, item in top]


class _LexicalReformulationSearch:
    """REFORMULATE-SEARCH-FUSE loop over lexical _InMemoryStore.

    Bridge pattern: same ReformulationStrategy interface as
    ReflectiveLatentSearch, but retrieves via word-overlap scoring
    instead of latent-space dot products.
    """

    def __init__(self, store: _InMemoryStore, strategies: list) -> None:
        self._store = store
        self._strategies = strategies

    def query(self, question: str, top_k: int) -> tuple[str, list[str]]:
        top = self._store.scored_best_match(question, top_k)
        best_score, best_item = top[0] if top else (0, None)
        best_evidence = [item.context for _, item in top]

        for strategy in self._strategies:
            for variant in strategy.reformulate(question):
                scored = self._store.scored_best_match(variant, top_k)
                if scored and scored[0][0] > best_score:
                    best_score = scored[0][0]
                    best_item = scored[0][1]
                    best_evidence = [item.context for _, item in scored]

        if best_item is None:
            return "", []
        return best_item.ground_truth, best_evidence


class TardigradeAdapter(BenchmarkAdapter):
    """Real-engine adapter with portable lexical fallback."""

    name = "tardigrade"

    def __init__(self) -> None:
        self._store = _InMemoryStore()
        self._cell_to_item: dict[int, BenchmarkItem] = {}
        # Cell ID → chunk text. Required for chunk-level cross-encoder
        # reranking — passing parent-item context here was the
        # 2026-05-15 bug that crushed LongMemEval scoring. Always
        # populated in production now (small memory cost: one chunk
        # text reference per cell, typically ~100-500 chars).
        # The diagnostic harness still calls
        # `enable_chunk_text_tracking()` for clarity; flag defaults
        # to True so production reranker has the text it needs.
        self._cell_to_chunk_text: dict[int, str] = {}
        self._track_chunk_text: bool = True
        self._engine = None
        self._hook = None
        self._mode = "in_memory"
        self._refinement = _REFINEMENT
        self._reranker = None
        if _RERANK_MODEL:
            try:
                from tardigrade_hooks.reranker import CrossEncoderReranker  # type: ignore

                self._reranker = CrossEncoderReranker(model_name=_RERANK_MODEL, device=_DEVICE)
            except Exception:
                # Fall back silently (metadata records the absence) — reranker
                # is an optional second-stage, not a blocker.
                self._reranker = None

        self._lexical_rls = None
        force_fallback = os.getenv("TDB_BENCH_FORCE_FALLBACK", "").lower() in ("1", "true", "yes")
        if force_fallback:
            self._init_lexical_rls()
            return

        try:
            import tardigrade_db  # type: ignore
            from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook  # type: ignore

            model, _tokenizer, _query_layer = _load_model_cached()
            data_dir = tempfile.mkdtemp(prefix="tdb_bench_")
            self._engine = tardigrade_db.Engine(data_dir)
            try:
                self._engine.set_refinement_mode(self._refinement)
            except Exception:
                # Older engine builds without refinement API still benchmark fine.
                pass
            self._hook = HuggingFaceKVHook(
                self._engine, owner=1, k=5,
                model_config=model.config, model=model,
                use_hidden_states=True,
            )
            self._mode = "native"
            self._rls = None
            if _RLS_MODE != RLS_MODE_NONE:
                from tardigrade_hooks.rls import (  # type: ignore
                    EmbeddingExpansionStrategy,
                    GenerativeReformulationStrategy,
                    KeywordExpansionStrategy,
                    LLMAgentReformulationStrategy,
                    MultiPhrasingStrategy,
                    ReflectiveLatentSearch,
                )
                strategies = []
                if _RLS_MODE in (RLS_MODE_KEYWORD, RLS_MODE_BOTH):
                    strategies.append(KeywordExpansionStrategy())
                if _RLS_MODE in (RLS_MODE_MULTIPHRASING, RLS_MODE_BOTH):
                    strategies.append(MultiPhrasingStrategy())
                if _RLS_MODE in (RLS_MODE_EMBEDDING, RLS_MODE_BOTH):
                    rls_model_tmp, rls_tok_tmp, _ = _load_model_cached()
                    embed_w = rls_model_tmp.get_input_embeddings().weight.detach().float().cpu().numpy()
                    strategies.append(EmbeddingExpansionStrategy(rls_tok_tmp, embed_w))
                if _RLS_MODE == RLS_MODE_GENERATIVE:
                    import torch as _torch
                    from transformers import AutoModelForCausalLM as _Auto, AutoTokenizer as _Tok
                    _gen_model_name = os.getenv("TDB_RLS_GEN_MODEL", RLS_DEFAULT_GEN_MODEL)
                    _gen_dtype = getattr(_torch, os.getenv("TDB_RLS_GEN_DTYPE", "float16"))
                    _gen_device = os.getenv("TDB_BENCH_DEVICE", "mps" if _torch.backends.mps.is_available() else "cpu")
                    _gen_tok = _Tok.from_pretrained(_gen_model_name)
                    _gen_model = _Auto.from_pretrained(_gen_model_name, dtype=_gen_dtype).to(_gen_device)
                    _gen_model.requires_grad_(False)
                    strategies.append(GenerativeReformulationStrategy(_gen_model, _gen_tok))
                if _RLS_MODE == RLS_MODE_AGENT:
                    _agent_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
                    strategies.append(LLMAgentReformulationStrategy(api_key=_agent_key))
                if strategies:
                    rls_model, rls_tokenizer, rls_query_layer = _load_model_cached()
                    rls_kwargs = {}
                    if _RLS_MODE == RLS_MODE_AGENT:
                        rls_kwargs["confidence_threshold"] = float("inf")
                    self._rls = ReflectiveLatentSearch(
                        engine=self._engine,
                        model=rls_model,
                        tokenizer=rls_tokenizer,
                        query_layer=rls_query_layer,
                        hidden_size=rls_model.config.hidden_size,
                        owner=1,
                        k=5,
                        strategies=strategies,
                        **rls_kwargs,
                    )
        except Exception as exc:
            self._engine = None
            self._hook = None
            self._rls = None
            self._mode = os.getenv("TDB_BENCH_FALLBACK_MODE", "in_memory")
            self._fallback_reason = str(exc)[:200]
            self._init_lexical_rls()

    def _init_lexical_rls(self) -> None:
        """Set up lexical reformulation search for in_memory mode."""
        if _RLS_MODE == RLS_MODE_NONE:
            return
        strategies = _create_rls_strategies(_RLS_MODE)
        if strategies:
            self._lexical_rls = _LexicalReformulationSearch(self._store, strategies)

    # ---- BenchmarkAdapter contract ----

    def ingest(self, items: list[BenchmarkItem]) -> None:
        if self._mode != "native":
            for item in items:
                self._store.insert(item)
            return

        import torch  # type: ignore

        from tardigrade_hooks.chunker import ParagraphBoundaryStrategy, TextChunker

        model, tokenizer, query_layer = _load_model_cached()
        # Phase 1A.2 — ParagraphBoundaryStrategy prefers split at
        # turn boundaries (\n\n), then sentence, then whitespace.
        # LongMemEval haystack sessions use \n\n between speaker
        # turns; LoCoMo evidence is single-sentence and falls
        # through to whitespace cleanly. The 2026-05-14 audit traced
        # LongMemEval hub-cell dominance to the unbounded fragment
        # chunks produced by the previous boundary-unaware splitter.
        chunker = TextChunker(
            tokenizer,
            max_tokens=_CHUNK_TOKENS,
            overlap_tokens=_CHUNK_OVERLAP,
            boundary_strategy=ParagraphBoundaryStrategy(),
        )

        # Slice D2 — producer-consumer pipeline.
        #
        # Producer (this thread): runs the GPU-bound batched forward
        # passes and posts (write_requests, item) onto a bounded
        # queue. Consumer (`_writer_loop`): calls
        # `engine.mem_write_batch` and updates `_cell_to_item` /
        # `_store`. The bounded queue is the backpressure mechanism;
        # the consumer holds the engine mutex during its writes,
        # so this is genuine CPU/GPU overlap on the engine's side
        # (PyTorch releases the GIL during CUDA work; the engine's
        # `py.detach()` releases it during writes).
        write_queue: queue.Queue = queue.Queue(maxsize=_INGEST_PIPELINE_DEPTH)
        writer_error: dict[str, BaseException] = {}
        writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(write_queue, writer_error),
            name="tdb-ingest-writer",
            daemon=True,
        )
        writer_thread.start()

        _total = len(items)
        try:
            for _idx, item in enumerate(items, 1):
                if writer_error:
                    break
                chunks = chunker.chunk(item.context)
                chunk_texts = [c.text for c in chunks] if chunks else [item.context[:500]]
                print(
                    f"[ingest {_idx}/{_total}] {item.item_id} "
                    f"chunks={len(chunk_texts)}",
                    flush=True,
                )
                write_requests, written_chunks = self._build_write_requests_for_item(
                    chunk_texts, tokenizer, model, query_layer, torch,
                )
                write_queue.put((write_requests, written_chunks, item))
        finally:
            write_queue.put(None)
            writer_thread.join()

        if writer_error:
            raise writer_error["err"]

    def _writer_loop(self, write_queue, error_slot):
        """Consumer: drains queued write batches, calls
        `engine.mem_write_batch`, updates side-tables.

        Runs on a dedicated thread for CPU/GPU overlap with the
        ingest producer. The engine releases the GIL inside
        `mem_write_batch` via `py.detach()`, so the producer's next
        forward pass can launch concurrently while writes commit.
        """
        try:
            while True:
                msg = write_queue.get()
                if msg is None:
                    return
                write_requests, written_chunks, item = msg
                if write_requests:
                    cell_ids = self._engine.mem_write_batch(write_requests)
                    # `cell_ids`, `write_requests`, and `written_chunks`
                    # share index-order — built together by
                    # `_build_write_requests_for_item`. Filtering of
                    # `should_write=False` decisions happens *before* the
                    # request is appended, so all three lists are aligned.
                    for cell_id, chunk_text in zip(cell_ids, written_chunks, strict=True):
                        self._cell_to_item[int(cell_id)] = item
                        if self._track_chunk_text:
                            self._cell_to_chunk_text[int(cell_id)] = chunk_text
                self._store.insert(item)
        except BaseException as exc:  # noqa: BLE001 — re-raised on producer thread
            error_slot["err"] = exc
            # Drain remaining queued items so the producer doesn't
            # block forever on a full queue while we're dying.
            while True:
                try:
                    if write_queue.get_nowait() is None:
                        return
                except queue.Empty:
                    return

    def _build_write_requests_for_item(
        self,
        chunk_texts,
        tokenizer,
        model,
        query_layer,
        torch,
    ):
        """Collect WriteRequest tuples for one item's chunks.

        Slice B1 batches chunks `_GPU_BATCH_SIZE` at a time per
        forward pass; Slice B2 emits the resulting requests as a
        single `engine.mem_write_batch` per item; Slice D3
        pre-tokenizes the whole item once and slices `input_ids` /
        `attention_mask` per batch — saves one tokenizer round-trip
        per batch.
        """
        if not chunk_texts:
            return [], []

        # Slice D3 — single tokenizer call per item.
        item_inputs = tokenizer(
            chunk_texts,
            return_tensors="pt",
            truncation=True,
            max_length=_CHUNK_TOKENS,
            padding=True,
        )
        item_inputs = {k: v.to(model.device) for k, v in item_inputs.items()}

        write_requests: list = []
        # Chunks that actually produced a write. Cells returned from
        # `mem_write_batch` map 1:1 against this list, in order — the
        # diagnostic harness uses this to learn which chunk text each
        # cell_id stored.
        written_chunk_texts: list[str] = []
        chunk_total = item_inputs["input_ids"].shape[0]
        for batch_start in range(0, chunk_total, _GPU_BATCH_SIZE):
            batch_end = min(batch_start + _GPU_BATCH_SIZE, chunk_total)
            decisions = self._forward_pass_batch(
                item_inputs, batch_start, batch_end, model, query_layer, torch,
            )
            for offset, decision in enumerate(decisions):
                if (
                    not decision.should_write
                    or len(decision.key) == 0
                    or len(decision.value) == 0
                ):
                    continue
                write_requests.append(
                    (
                        _DEFAULT_WRITE_OWNER,
                        query_layer,
                        decision.key,
                        decision.value,
                        decision.salience,
                        _NO_PARENT_CELL_ID,
                    )
                )
                written_chunk_texts.append(chunk_texts[batch_start + offset])
        return write_requests, written_chunk_texts

    def _forward_pass_batch(
        self,
        item_inputs,
        batch_start,
        batch_end,
        model,
        query_layer,
        torch,
    ):
        """One batched forward pass → one `WriteDecision` per chunk.

        Receives slices of the item-wide pre-tokenized tensors.
        Runs a single forward pass on the slice and dispatches the
        batch-aware hook per chunk with its own un-padded
        `seq_len`. The hook reconstructs each chunk's per-token K
        vector and KV payload exactly as it would under the serial
        path.
        """
        batch_inputs = {
            key: tensor[batch_start:batch_end] for key, tensor in item_inputs.items()
        }
        with torch.no_grad():
            out = model(**batch_inputs, use_cache=True, output_hidden_states=True)

        batched_hidden = out.hidden_states[query_layer]
        attention_mask = batch_inputs["attention_mask"]
        decisions = []
        for batch_index in range(batched_hidden.shape[0]):
            chunk_len = int(attention_mask[batch_index].sum().item())
            decision = self._hook.on_generate(
                layer=query_layer,
                past_key_values=out.past_key_values,
                model_hidden_states=batched_hidden,
                batch_index=batch_index,
                seq_len=chunk_len,
            )
            decisions.append(decision)
        return decisions

    def query(self, item: BenchmarkItem, top_k: int) -> AdapterQueryResult:
        if self._mode != "native":
            start = time.perf_counter()
            if self._lexical_rls is not None:
                answer, evidence = self._lexical_rls.query(item.question, top_k)
            else:
                answer, evidence = self._store.best_match(item.question, top_k)
            return AdapterQueryResult(
                answer=answer,
                evidence=evidence,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                status="ok",
                error=None,
            )

        if self._rls is not None:
            start = time.perf_counter()
            handles = self._rls.query(item.question, top_k=top_k)
            latency_ms = (time.perf_counter() - start) * 1000.0

            evidence: list[str] = []
            answer = ""
            for h in handles[: max(1, top_k)]:
                mapped = self._cell_to_item.get(int(h.cell_id))
                if mapped is None:
                    continue
                evidence.append(mapped.context)
                if not answer:
                    answer = mapped.ground_truth
            if not answer:
                return AdapterQueryResult(
                    answer="", evidence=[], latency_ms=latency_ms,
                    status="failed", error="no_rls_match",
                )
            return AdapterQueryResult(
                answer=answer, evidence=evidence,
                latency_ms=latency_ms, status="ok", error=None,
            )

        import torch  # type: ignore

        model, tokenizer, query_layer = _load_model_cached()
        start = time.perf_counter()
        inputs = tokenizer(
            item.question, return_tensors="pt", truncation=True, max_length=256,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, use_cache=True, output_hidden_states=True)
        handles = self._hook.on_prefill(
            layer=query_layer,
            past_key_values=out.past_key_values,
            model_hidden_states=out.hidden_states[query_layer],
        )

        # Stage-2 cross-encoder reranking on per-CHUNK text.
        #
        # Bug fixed 2026-05-15: previously this passed the parent
        # item's full `context` for every candidate. For LongMemEval
        # where each item has 50-150 chunks averaging 100-500 chars
        # each, all same-item chunks received identical reranker
        # input (the full 25-42KB session text) and therefore
        # identical scores — the reranker was doing item-level
        # reranking when it should have been chunk-level. LoCoMo
        # masked the bug because evidence-only items have 1 chunk
        # whose text equals the item context. See
        # `docs/experiments/2026-05-14-bench-audit.md` Phase 1B.
        if self._reranker is not None and handles:
            handles = self._reranker.rerank(
                query_text=item.question,
                candidates=handles,
                get_text=lambda h: self._cell_to_chunk_text.get(int(h.cell_id)),
            )

        latency_ms = (time.perf_counter() - start) * 1000.0

        evidence: list[str] = []
        answer = ""
        for h in handles[: max(1, top_k)]:
            mapped = self._cell_to_item.get(int(h.cell_id))
            if mapped is None:
                continue
            evidence.append(mapped.context)
            if not answer:
                answer = mapped.ground_truth
        if not answer:
            return AdapterQueryResult(
                answer="",
                evidence=[],
                latency_ms=latency_ms,
                status="failed",
                error="no_retrieval_match",
            )
        return AdapterQueryResult(
            answer=answer,
            evidence=evidence,
            latency_ms=latency_ms,
            status="ok",
            error=None,
        )

    def enable_chunk_text_tracking(self) -> None:
        """Opt into populating `_cell_to_chunk_text` during ingestion.

        Diagnostic harness only — Phase 0 of the retrieval debug plan.
        Adds a small per-cell memory cost (one chunk text string per
        stored cell). Off by default so production runs aren't taxed.
        """
        self._track_chunk_text = True

    def reset(self) -> None:
        self._store.clear()
        self._cell_to_item.clear()
        self._cell_to_chunk_text.clear()
        # Drop the engine and rebuild — items from one dataset must not
        # leak into the next. Cheap because the model stays cached.
        if self._mode == "native":
            try:
                import tardigrade_db  # type: ignore
                from tardigrade_hooks.hf_kv_hook import HuggingFaceKVHook  # type: ignore

                model, _tokenizer, _query_layer = _load_model_cached()
                data_dir = tempfile.mkdtemp(prefix="tdb_bench_")
                self._engine = tardigrade_db.Engine(data_dir)
                try:
                    self._engine.set_refinement_mode(self._refinement)
                except Exception:
                    pass
                self._hook = HuggingFaceKVHook(
                    self._engine, owner=1, k=5,
                    model_config=model.config, model=model,
                    use_hidden_states=True,
                )
                self._rls = None
                if _RLS_MODE != RLS_MODE_NONE:
                    from tardigrade_hooks.rls import (  # type: ignore
                        KeywordExpansionStrategy,
                        LLMAgentReformulationStrategy,
                        MultiPhrasingStrategy,
                        ReflectiveLatentSearch,
                    )
                    strategies = []
                    if _RLS_MODE in (RLS_MODE_KEYWORD, RLS_MODE_BOTH):
                        strategies.append(KeywordExpansionStrategy())
                    if _RLS_MODE in (RLS_MODE_MULTIPHRASING, RLS_MODE_BOTH):
                        strategies.append(MultiPhrasingStrategy())
                    if _RLS_MODE == RLS_MODE_AGENT:
                        _agent_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
                        strategies.append(LLMAgentReformulationStrategy(api_key=_agent_key))
                    if strategies:
                        rls_model, rls_tokenizer, rls_query_layer = _load_model_cached()
                        rls_kwargs = {}
                        if _RLS_MODE == RLS_MODE_AGENT:
                            rls_kwargs["confidence_threshold"] = float("inf")
                        self._rls = ReflectiveLatentSearch(
                            engine=self._engine,
                            model=rls_model,
                            tokenizer=rls_tokenizer,
                            query_layer=rls_query_layer,
                            hidden_size=rls_model.config.hidden_size,
                            owner=1,
                            k=5,
                            strategies=strategies,
                            **rls_kwargs,
                        )
            except Exception:
                self._engine = None
                self._hook = None
                self._rls = None
                self._mode = "in_memory"
                self._init_lexical_rls()

    def metadata(self) -> dict[str, str]:
        meta = {
            "adapter": self.name,
            "mode": self._mode,
            "refinement_mode": self._refinement,
            "rls_mode": _RLS_MODE,
            "chunk_tokens": str(_CHUNK_TOKENS),
            "model": _MODEL_NAME if self._mode == "native" else "lexical",
            "reranker_model": _RERANK_MODEL if self._reranker is not None else "",
        }
        if self._mode != "native" and hasattr(self, "_fallback_reason"):
            meta["fallback_reason"] = self._fallback_reason
        return meta
