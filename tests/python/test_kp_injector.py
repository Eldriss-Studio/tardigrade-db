# ATDD acceptance tests for KnowledgePackStore.
#
# Design pattern: Facade (wraps chat template + KV computation + pack
# storage + DynamicCache reconstruction + injection).
#
# Uses GPT-2 with a minimal chat template for structural tests.
# GPT-2 is too small to recall novel facts, so these tests verify
# the contract (shapes, pack counts, token accounting) — not the
# quality of generated answers. Correctness was validated by
# experiments/injection_vs_text_rag.py on Qwen3-0.6B.

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db
from tardigrade_hooks.kp_injector import KnowledgePackStore
from tardigrade_hooks.chat_template_adapter import (
    ChatTemplateAdapter,
    LegacySystemAdapter,
    UserMessageAdapter,
    select_chat_template_adapter,
)
from tardigrade_hooks.retrieval_key_strategy import (
    HiddenStateKeyStrategy,
    KVectorKeyStrategy,
    RetrievalKeyStrategy,
)

# Minimal chat template — GPT-2 doesn't ship one.
CHAT_TEMPLATE = '{% for message in messages %}{{ message["content"] }}{% endfor %}'


@pytest.fixture
def gpt2():
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    return model


@pytest.fixture
def tokenizer():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.chat_template = CHAT_TEMPLATE
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def engine(tmp_path):
    return tardigrade_db.Engine(str(tmp_path))


@pytest.fixture
def kps(engine, gpt2, tokenizer):
    return KnowledgePackStore(engine, gpt2, tokenizer, owner=1)


# -- 1: store creates one pack with all layers --------------------------------


def test_kp_store_creates_pack_for_all_layers(kps, engine):
    """GIVEN KnowledgePackStore with GPT-2 (12 layers),
    WHEN store("some fact"),
    THEN pack_count == 1 and store() returns a pack_id."""
    pack_id = kps.store("The wifi password is mango-cathedral-7")

    assert isinstance(pack_id, int)
    assert engine.pack_count() == 1


# -- 2: store wraps fact in chat template --------------------------------------


def test_kp_store_uses_chat_template(kps, engine, tokenizer):
    """GIVEN KnowledgePackStore,
    WHEN store("fact text"),
    THEN the stored KV seq_len matches chat-template-formatted token count,
    not raw text token count."""
    fact = "Eduardo's apartment number is 4B on the third floor"
    kps.store(fact)

    # Expected seq_len from chat template
    messages = [{"role": "system", "content": fact}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    expected_seq_len = len(tokenizer.encode(formatted))

    # Retrieve the pack and check stored KV dimensions
    query_input = tokenizer.encode(fact, return_tensors="pt")
    with torch.no_grad():
        out = kps.model(query_input, output_hidden_states=True)
    hidden = out.hidden_states[kps.query_layer][0]
    from tardigrade_hooks.encoding import encode_per_token
    h_tokens = hidden[1:].numpy().astype(np.float32)
    query_key = encode_per_token(h_tokens, kps.hidden_size)

    packs = engine.mem_read_pack(query_key, 1, 1)
    assert len(packs) == 1

    # Each layer's data is [K_flat | V_flat], each half = seq_len * kv_dim
    layer_data = np.array(packs[0]["layers"][0]["data"], dtype=np.float32)
    half = len(layer_data) // 2
    actual_seq_len = half // kps.kv_dim

    assert actual_seq_len == expected_seq_len


# -- 3: retrieve_and_inject reconstructs valid DynamicCache --------------------


def test_kp_retrieve_returns_valid_cache(kps, engine):
    """GIVEN a stored fact,
    WHEN retrieve_and_inject(query),
    THEN returns (DynamicCache, query_ids, attention_mask) with correct shapes."""
    kps.store("The pharmacy closes at 8:30pm on Tuesdays")

    cache, query_ids, attn_mask = kps.retrieve_and_inject(
        "When does the pharmacy close?"
    )

    assert cache is not None
    assert len(cache.layers) == 12  # GPT-2 has 12 layers

    kv_len = cache.get_seq_length()
    q_len = query_ids.shape[1]

    # Attention mask covers KV cache + query tokens
    assert attn_mask.shape == (1, kv_len + q_len)
    assert attn_mask.sum().item() == kv_len + q_len  # all ones

    # Each layer has correct head/dim shape
    layer0 = cache.layers[0]
    assert layer0.keys.shape[1] == kps.num_kv_heads
    assert layer0.keys.shape[3] == kps.head_dim


# -- 4: generate uses zero memory prompt tokens -------------------------------


def test_kp_generate_uses_zero_memory_prompt_tokens(kps, engine):
    """GIVEN a stored fact,
    WHEN generate(query),
    THEN prompt_tokens == query-only token count (no memory text in prompt)
    AND had_memory == True."""
    kps.store("Lucia's favorite dinosaur is the Pachycephalosaurus")

    query = "What is Lucia's favorite dinosaur?"
    text, prompt_tokens, had_memory = kps.generate(query, max_new_tokens=5)

    assert had_memory is True

    # prompt_tokens should be just the query portion — no memory text pasted
    _, query_ids, _ = kps.retrieve_and_inject(query)
    assert prompt_tokens == query_ids.shape[1]


# -- 5: generate is transparent without memories -------------------------------


def test_kp_generate_transparent_without_memories(kps, engine):
    """GIVEN an empty engine (no stored memories),
    WHEN generate(query),
    THEN returns valid text, had_memory == False, and generation completes."""
    assert engine.pack_count() == 0

    text, prompt_tokens, had_memory = kps.generate(
        "What is the meaning of life?", max_new_tokens=10
    )

    assert had_memory is False
    assert isinstance(text, str)
    assert len(text) > 0
    assert prompt_tokens > 0


# -- 6: multiple stores create independent packs ------------------------------


def test_kp_multiple_stores_create_independent_packs(kps, engine):
    """GIVEN two different facts stored,
    WHEN querying each,
    THEN pack_count == 2 and retrieval returns different packs."""
    kps.store("The wifi password is mango-cathedral-7")
    kps.store("Eduardo's apartment is 4B on the third floor")

    assert engine.pack_count() == 2

    # Both facts stored, both retrievable
    cache1, _, _ = kps.retrieve_and_inject("What is the wifi password?")
    cache2, _, _ = kps.retrieve_and_inject("What is Eduardo's apartment?")

    assert cache1 is not None
    assert cache2 is not None


# -- 7: generate clones cache (no in-place mutation) ---------------------------


def test_kp_generate_clones_cache(kps, engine):
    """GIVEN a stored fact,
    WHEN generate() is called twice with the same query,
    THEN both calls succeed (cache clone prevents in-place mutation)."""
    kps.store("The pharmacy closes at 8:30pm on Tuesdays")
    query = "When does the pharmacy close?"

    text1, _, had1 = kps.generate(query, max_new_tokens=5)
    text2, _, had2 = kps.generate(query, max_new_tokens=5)

    assert had1 is True
    assert had2 is True
    assert isinstance(text1, str)
    assert isinstance(text2, str)


# -- 8: KnowledgePackStore + ChatTemplateAdapter across model families ---------
#
# Parameterized regression coverage exercising the Adapter pattern that makes
# kp_injector model-family-agnostic. Original coverage (the 4 cases below) was
# added 2026-05-18 for the CUDA device-placement fix; expanded the same day
# with the (adapter × model) matrix when Qwen3.5 surfaced that the previous
# system-only `apply_chat_template` trick was Qwen3-specific.
#
# Matrix:
#   - (LegacySystemAdapter, Qwen3-0.6B)   -> backwards-compat (lenient template)
#   - (UserMessageAdapter,  Qwen3-0.6B)   -> new adapter on lenient template
#   - (UserMessageAdapter,  Qwen3.5-0.8B) -> new adapter on strict template
#
# Plus a separate test that exercises the default Factory auto-selection path
# (no adapter passed -> factory probes tokenizer -> picks the right adapter).


def _load_cuda_model(model_name: str):
    """Load a HF causal LM on CUDA at FP32. Skips if CUDA isn't available.

    FP32 is mandatory because the KV-injection hook calls `.numpy()` on hidden
    states, which doesn't support BF16. See `kp_injector.retrieve_and_inject`.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        output_hidden_states=True,
    )
    model.to("cuda")
    model.training = False
    return model, tok


# (adapter_factory, model_name) — each pair must be supported by KnowledgePackStore.
#
# Note on model choice for the strict-template case: Qwen3.5-0.8B was the
# obvious candidate (it was what surfaced the chat-template bug we're
# fixing) but it uses a hybrid linear/standard attention architecture that
# tardigrade-db's KV-capture path doesn't yet support — `kv.layers[li]`
# can be a `LinearAttentionLayer` which has no `.keys` attribute. That's a
# separate architectural concern, not a chat-template one, and is out of
# scope for this PR.
#
# TinyLlama-1.1B-Chat-v1.0 is the proxy: small, ungated, standard
# transformer attention throughout, strict chat template that requires
# user-role messages. Exercises the same UserMessageAdapter code path
# that Qwen3.5 will need once the LinearAttention support lands.
ADAPTER_MODEL_MATRIX = [
    (LegacySystemAdapter, "Qwen/Qwen3-0.6B"),
    (UserMessageAdapter, "Qwen/Qwen3-0.6B"),
    (UserMessageAdapter, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
]


@pytest.mark.gpu
@pytest.mark.parametrize("adapter_factory, model_name", ADAPTER_MODEL_MATRIX)
def test_kp_store_works_on_cuda_model(adapter_factory, model_name, engine):
    """GIVEN a KnowledgePackStore wrapping a CUDA-loaded model with the given adapter,
    WHEN store() is called,
    THEN it returns an int pack id without raising a chat-template or device error."""
    model, tok = _load_cuda_model(model_name)
    kps = KnowledgePackStore(
        engine, model, tok, owner=1, adapter=adapter_factory()
    )
    pack_id = kps.store("The override vector is DILLINGER-1")

    assert isinstance(pack_id, int)
    assert engine.pack_count() == 1


@pytest.mark.gpu
@pytest.mark.parametrize("adapter_factory, model_name", ADAPTER_MODEL_MATRIX)
def test_kp_generate_works_on_cuda_model(adapter_factory, model_name, engine):
    """GIVEN a CUDA-loaded model + adapter with one stored fact,
    WHEN generate() is called,
    THEN it returns (text, n_tokens, had_memory=True) without raising."""
    model, tok = _load_cuda_model(model_name)
    kps = KnowledgePackStore(
        engine, model, tok, owner=1, adapter=adapter_factory()
    )
    kps.store("The override vector is DILLINGER-1")

    text, n_tokens, had_memory = kps.generate(
        "What is the override vector?", max_new_tokens=5
    )

    assert isinstance(text, str)
    assert isinstance(n_tokens, int)
    assert had_memory is True


@pytest.mark.gpu
@pytest.mark.parametrize("adapter_factory, model_name", ADAPTER_MODEL_MATRIX)
def test_kp_generate_with_trace_works_on_cuda_model(
    adapter_factory, model_name, engine
):
    """GIVEN a CUDA-loaded model + adapter with one stored fact,
    WHEN generate_with_trace() is called,
    THEN the trace-boosted retrieval path returns a triple without raising."""
    model, tok = _load_cuda_model(model_name)
    kps = KnowledgePackStore(
        engine, model, tok, owner=1, adapter=adapter_factory()
    )
    kps.store("The override vector is DILLINGER-1")

    text, n_tokens, had_memory = kps.generate_with_trace(
        "What is the override vector?", max_new_tokens=5
    )

    assert isinstance(text, str)
    assert isinstance(n_tokens, int)
    assert had_memory is True


@pytest.mark.gpu
@pytest.mark.parametrize("adapter_factory, model_name", ADAPTER_MODEL_MATRIX)
def test_kp_generate_multi_works_on_cuda_model(
    adapter_factory, model_name, engine
):
    """GIVEN a CUDA-loaded model + adapter with two stored facts,
    WHEN generate_multi() is called,
    THEN the multi-memory retrieval path returns a triple without raising."""
    model, tok = _load_cuda_model(model_name)
    kps = KnowledgePackStore(
        engine, model, tok, owner=1, adapter=adapter_factory()
    )
    kps.store("The override vector is DILLINGER-1")
    kps.store("The carrier identifier is EVE-KIM")

    text, n_tokens, had_memory = kps.generate_multi(
        "What identifiers are available?", k=2, max_new_tokens=5
    )

    assert isinstance(text, str)
    assert isinstance(n_tokens, int)
    assert had_memory is True


@pytest.mark.gpu
def test_kp_default_adapter_auto_selects_for_qwen3(engine):
    """GIVEN a KnowledgePackStore constructed without an explicit adapter,
    WHEN the tokenizer is Qwen3 (lenient template),
    THEN the factory auto-selects LegacySystemAdapter and store/retrieve work
    end-to-end. Backwards-compat for callers that don't know about adapters."""
    model, tok = _load_cuda_model("Qwen/Qwen3-0.6B")
    kps = KnowledgePackStore(engine, model, tok, owner=1)  # no adapter passed

    assert isinstance(kps.adapter, LegacySystemAdapter)

    pack_id = kps.store("Default adapter selection works")
    assert isinstance(pack_id, int)


# -- 9: ChatTemplateAdapter Factory unit tests --------------------------------
#
# Pure-Python tests for the Factory's tokenizer-probing logic. Mocked
# tokenizers, no GPU required.


class _LenientFakeTokenizer:
    """Mock tokenizer whose chat template accepts system-only message lists."""

    def apply_chat_template(self, messages, **kwargs):
        # Always returns a string; never raises.
        return "<sys>" + "".join(m["content"] for m in messages) + "</sys>"


class _StrictFakeTokenizer:
    """Mock tokenizer whose chat template requires a user-role message."""

    def apply_chat_template(self, messages, **kwargs):
        if not any(m.get("role") == "user" for m in messages):
            raise Exception("No user query found in messages.")
        return "<chat>" + "".join(m["content"] for m in messages) + "</chat>"


def test_select_adapter_picks_legacy_for_lenient_tokenizer():
    """GIVEN a tokenizer whose template accepts [system]-only,
    WHEN the Factory probes it,
    THEN LegacySystemAdapter is selected (backwards-compat preserved)."""
    adapter = select_chat_template_adapter(_LenientFakeTokenizer())
    assert isinstance(adapter, LegacySystemAdapter)


def test_select_adapter_picks_user_for_strict_tokenizer():
    """GIVEN a tokenizer whose template rejects [system]-only,
    WHEN the Factory probes it,
    THEN UserMessageAdapter is selected (forward-compat with strict templates)."""
    adapter = select_chat_template_adapter(_StrictFakeTokenizer())
    assert isinstance(adapter, UserMessageAdapter)


def test_select_adapter_handles_none_tokenizer():
    """GIVEN no tokenizer (e.g. consumer running in a non-local-model mode),
    WHEN the Factory is called with None,
    THEN it returns a safe default (UserMessageAdapter, the strict form)
    without raising. Documented gap-review edge case."""
    adapter = select_chat_template_adapter(None)
    assert isinstance(adapter, UserMessageAdapter)


def test_user_message_adapter_builds_store_messages():
    """UserMessageAdapter wraps facts in a single user-role message."""
    adapter = UserMessageAdapter()
    msgs = adapter.store_messages("A stored fact")
    assert msgs == [{"role": "user", "content": "A stored fact"}]


def test_user_message_adapter_builds_retrieve_messages():
    """UserMessageAdapter retrieve form is the multi-turn [user, assistant, user]
    shape, with the stored fact in the first user turn and the query in the
    second. The empty assistant turn carries no semantic content; its only
    role is to separate the two user turns for templates that disallow
    consecutive same-role messages."""
    adapter = UserMessageAdapter()
    fact_msgs, full_msgs = adapter.retrieve_messages(
        "Stored fact text", "What is the query?"
    )
    assert fact_msgs == [{"role": "user", "content": "Stored fact text"}]
    assert full_msgs == [
        {"role": "user", "content": "Stored fact text"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "What is the query?"},
    ]


def test_legacy_system_adapter_builds_store_messages():
    """LegacySystemAdapter wraps facts as a system instruction (Qwen3 form)."""
    adapter = LegacySystemAdapter()
    msgs = adapter.store_messages("A stored fact")
    assert msgs == [{"role": "system", "content": "A stored fact"}]


def test_legacy_system_adapter_builds_retrieve_messages():
    """LegacySystemAdapter retrieve form uses a system-message placeholder for
    the fact-length calculation (matches kp_injector's pre-adapter behavior
    byte-for-byte so existing tardigrade_data/ stays readable)."""
    adapter = LegacySystemAdapter()
    fact_msgs, full_msgs = adapter.retrieve_messages(
        "(unused — Legacy uses placeholder)", "What is the query?"
    )
    assert fact_msgs == [{"role": "system", "content": "placeholder"}]
    assert full_msgs == [
        {"role": "system", "content": "placeholder"},
        {"role": "user", "content": "What is the query?"},
    ]


# -- KnowledgePackStore integration with RetrievalKeyStrategy ---------------

def test_kp_default_constructor_builds_hidden_state_strategy(engine, gpt2, tokenizer):
    """When no strategy/registry/query_layer is provided, the default
    retrieval-key strategy is HiddenStateKeyStrategy at the library-
    default layer."""
    kps = KnowledgePackStore(engine, gpt2, tokenizer, owner=1)
    assert isinstance(kps.retrieval_key_strategy, HiddenStateKeyStrategy)
    assert kps.retrieval_key_strategy.query_layer == kps.query_layer


def test_kp_accepts_explicit_retrieval_key_strategy_kwarg(engine, gpt2, tokenizer):
    """Explicit retrieval_key_strategy= overrides every other source."""
    custom = KVectorKeyStrategy(softmax_layer_idx=3)
    kps = KnowledgePackStore(
        engine, gpt2, tokenizer, owner=1,
        retrieval_key_strategy=custom,
    )
    assert kps.retrieval_key_strategy is custom


def test_kp_explicit_query_layer_constructs_hidden_state_strategy(engine, gpt2, tokenizer):
    """Explicit query_layer=N constructs HiddenStateKeyStrategy(N)
    when no explicit strategy is provided."""
    kps = KnowledgePackStore(engine, gpt2, tokenizer, owner=1, query_layer=5)
    assert isinstance(kps.retrieval_key_strategy, HiddenStateKeyStrategy)
    assert kps.retrieval_key_strategy.query_layer == 5


def test_kp_calibration_registry_kvector_best_strategy_constructs_kvector(
    engine, gpt2, tokenizer, tmp_path,
):
    """When the registry has best_strategy='k_vector', KnowledgePackStore
    constructs a KVectorKeyStrategy at the cached best_layer."""
    from tardigrade_hooks.calibrate import CalibrationResult, LayerScore
    from tardigrade_hooks.calibration_registry import CalibrationRegistry

    reg = CalibrationRegistry(tmp_path / "calib.json")
    cached = CalibrationResult(
        model_id=getattr(gpt2.config, "name_or_path", "") or "gpt2",
        tardigrade_db_version="0.3.2",
        timestamp_iso="2026-05-19T00:00:00",
        n_layers=12,
        hidden_size=768,
        best_layer=7,
        best_strategy="k_vector",
        scores=(LayerScore(layer=7, kind="attention", strategy="k_vector", top1=20, top5=20),),
    )
    reg.save(cached)

    kps = KnowledgePackStore(
        engine, gpt2, tokenizer, owner=1, calibration_registry=reg,
    )
    assert isinstance(kps.retrieval_key_strategy, KVectorKeyStrategy)
    assert kps.retrieval_key_strategy.softmax_layer_idx == 7


def test_kp_calibration_registry_hidden_state_best_strategy_constructs_hidden_state(
    engine, gpt2, tokenizer, tmp_path,
):
    from tardigrade_hooks.calibrate import CalibrationResult, LayerScore
    from tardigrade_hooks.calibration_registry import CalibrationRegistry

    reg = CalibrationRegistry(tmp_path / "calib.json")
    cached = CalibrationResult(
        model_id=getattr(gpt2.config, "name_or_path", "") or "gpt2",
        tardigrade_db_version="0.3.2",
        timestamp_iso="2026-05-19T00:00:00",
        n_layers=12,
        hidden_size=768,
        best_layer=4,
        best_strategy="hidden_state",
        scores=(LayerScore(layer=4, kind="attention", strategy="hidden_state", top1=20, top5=20),),
    )
    reg.save(cached)

    kps = KnowledgePackStore(
        engine, gpt2, tokenizer, owner=1, calibration_registry=reg,
    )
    assert isinstance(kps.retrieval_key_strategy, HiddenStateKeyStrategy)
    assert kps.retrieval_key_strategy.query_layer == 4


def test_kp_unknown_best_strategy_in_registry_falls_back_to_hidden_state(
    engine, gpt2, tokenizer, tmp_path,
):
    """If the registry has a best_strategy name the library doesn't
    recognise, fall back to HiddenStateKeyStrategy at the cached layer
    (graceful degradation — don't crash the consumer)."""
    from tardigrade_hooks.calibrate import CalibrationResult, LayerScore
    from tardigrade_hooks.calibration_registry import CalibrationRegistry

    reg = CalibrationRegistry(tmp_path / "calib.json")
    cached = CalibrationResult(
        model_id=getattr(gpt2.config, "name_or_path", "") or "gpt2",
        tardigrade_db_version="0.3.2",
        timestamp_iso="2026-05-19T00:00:00",
        n_layers=12,
        hidden_size=768,
        best_layer=4,
        best_strategy="some_future_strategy_not_yet_known",
        scores=(),
    )
    reg.save(cached)

    kps = KnowledgePackStore(
        engine, gpt2, tokenizer, owner=1, calibration_registry=reg,
    )
    assert isinstance(kps.retrieval_key_strategy, HiddenStateKeyStrategy)
    assert kps.retrieval_key_strategy.query_layer == 4


def test_kp_store_invokes_retrieval_key_strategy_compute(engine, gpt2, tokenizer):
    """store() must route retrieval-key computation through the
    Strategy, not via inline hidden-state extraction. Verifies by
    counting calls to compute() on a recording subclass."""
    from tardigrade_hooks.retrieval_key_strategy import HiddenStateKeyStrategy

    calls: list = []

    class _RecordingHiddenState(HiddenStateKeyStrategy):
        def compute(self, hidden_states, kv, hidden_size):
            calls.append(("compute", self.query_layer))
            return super().compute(hidden_states, kv, hidden_size)

    kps = KnowledgePackStore(
        engine, gpt2, tokenizer, owner=1,
        retrieval_key_strategy=_RecordingHiddenState(query_layer=5),
    )
    kps.store("test fact for strategy routing", auto_link=False)
    assert len(calls) == 1
    assert calls[0] == ("compute", 5)


def test_kp_retrieve_invokes_retrieval_key_strategy_compute(engine, gpt2, tokenizer):
    """retrieve_and_inject() must also route through the Strategy. The
    strategy used for the query key MUST match the one used at store
    time — otherwise the fact's pack and the query's key live in
    different encoding spaces and retrieval is undefined."""
    from tardigrade_hooks.retrieval_key_strategy import HiddenStateKeyStrategy

    calls: list = []

    class _RecordingHiddenState(HiddenStateKeyStrategy):
        def compute(self, hidden_states, kv, hidden_size):
            calls.append(("compute", self.query_layer))
            return super().compute(hidden_states, kv, hidden_size)

    kps = KnowledgePackStore(
        engine, gpt2, tokenizer, owner=1,
        retrieval_key_strategy=_RecordingHiddenState(query_layer=5),
    )
    kps.store("test fact", auto_link=False)  # one compute() for store
    cache, query_ids, attn = kps.retrieve_and_inject("test query")
    # store() = 1 call; retrieve_and_inject() = 1 more = 2 total
    assert len(calls) == 2
    assert all(c == ("compute", 5) for c in calls)


# -- Hybrid-safe storage path --------------------------------------------------
#
# Storage was hybrid-unsafe through v0.3.3: ``KnowledgePackStore.store()``
# unconditionally read ``kv.layers[li].keys[0]`` over ``range(n_layers)``,
# which crashed on RecurrentGemma / Jamba / Qwen3-Next where some layers are
# recurrent and have no ``.keys`` attribute. v0.3.3's K-vector *retrieval*
# strategy unlocked hybrid models on the read side; this phase makes the
# write side hybrid-safe by deriving an architecture-aware layer count and
# delegating payload extraction to the same ``_softmax_layer_payloads``
# filter the calibration sweep already uses.


def test_softmax_layer_count_uniform_returns_num_hidden_layers():
    """Uniform-softmax models (GPT-2, Qwen3, Llama-3) have no
    ``layers_block_type`` in their config — every layer is softmax. The
    helper must report ``num_hidden_layers``, preserving the strength of
    the existing pack-integrity guard for these architectures."""
    from types import SimpleNamespace

    from tardigrade_hooks._hidden_states import softmax_layer_count

    cfg = SimpleNamespace(num_hidden_layers=12)
    assert softmax_layer_count(cfg) == 12


def test_softmax_layer_count_hybrid_via_layers_block_type():
    """RecurrentGemma exposes per-layer architecture in
    ``cfg.layers_block_type`` (Griffin pattern: alternating recurrent +
    attention). The helper must count only the attention entries."""
    from types import SimpleNamespace

    from tardigrade_hooks._hidden_states import softmax_layer_count

    cfg = SimpleNamespace(
        num_hidden_layers=4,
        layers_block_type=["recurrent", "recurrent", "attention", "recurrent"],
    )
    assert softmax_layer_count(cfg) == 1


def test_softmax_layer_count_hybrid_via_layer_types_alias():
    """Different model families spell the field differently. Jamba uses
    ``cfg.layer_types`` (vs RecurrentGemma's ``layers_block_type``); the
    helper must recognise both, matching what ``layer_kind_labels`` already
    accepts."""
    from types import SimpleNamespace

    from tardigrade_hooks._hidden_states import softmax_layer_count

    cfg = SimpleNamespace(
        num_hidden_layers=6,
        layer_types=[
            "attention", "recurrent", "recurrent",
            "attention", "recurrent", "recurrent",
        ],
    )
    assert softmax_layer_count(cfg) == 2


def test_softmax_layer_count_treats_full_attention_as_softmax():
    """Some hybrid configs label softmax layers as ``"full_attention"``
    rather than ``"attention"``. Both must count — ``layer_kind_labels``
    already normalises them; the count helper must follow."""
    from types import SimpleNamespace

    from tardigrade_hooks._hidden_states import softmax_layer_count

    cfg = SimpleNamespace(
        num_hidden_layers=3,
        layer_types=["full_attention", "linear_attention", "full_attention"],
    )
    assert softmax_layer_count(cfg) == 2


def test_softmax_layer_count_recognises_sliding_attention():
    """Gemma 2 / Mistral / Phi-3 alternate ``"sliding_attention"`` with
    ``"full_attention"``. Sliding-window attention is still softmax — it
    just constrains the attention mask to a local window — and produces
    a standard K/V cache. The classifier was treating it as unknown
    (and therefore as non-softmax), miscounting these uniform-softmax
    models as hybrid in ``is_supported`` reports."""
    from types import SimpleNamespace

    from tardigrade_hooks._hidden_states import softmax_layer_count

    cfg = SimpleNamespace(
        num_hidden_layers=4,
        layers_block_type=[
            "sliding_attention", "full_attention",
            "sliding_attention", "full_attention",
        ],
    )
    assert softmax_layer_count(cfg) == 4


def test_softmax_layer_count_recognises_local_and_global_attention():
    """Longformer / BigBird family configs use ``"local_attention"`` and
    ``"global_attention"``. Both are softmax variants; both produce
    K/V caches; both must count."""
    from types import SimpleNamespace

    from tardigrade_hooks._hidden_states import softmax_layer_count

    cfg = SimpleNamespace(
        num_hidden_layers=3,
        layer_types=["local_attention", "global_attention", "local_attention"],
    )
    assert softmax_layer_count(cfg) == 3


def test_softmax_layer_count_treats_delta_net_as_recurrent():
    """Qwen3-Next labels its linear-attention layers as ``"delta_net"``
    / ``"gated_delta_net"`` in the HF config. These have no softmax K/V
    cache to store and must not be counted as softmax layers — they're
    the recurrent half of the hybrid architecture, same shape of concern
    as RecurrentGemma's ``"recurrent"`` labels."""
    from types import SimpleNamespace

    from tardigrade_hooks._hidden_states import softmax_layer_count

    cfg = SimpleNamespace(
        num_hidden_layers=4,
        layer_types=[
            "delta_net", "delta_net", "delta_net", "full_attention",
        ],
    )
    assert softmax_layer_count(cfg) == 1


def test_softmax_layer_count_treats_mamba_as_recurrent():
    """SSM / Mamba labels (``"mamba"``) — pure state-space, no softmax
    K/V cache. Must not be counted as softmax layers."""
    from types import SimpleNamespace

    from tardigrade_hooks._hidden_states import softmax_layer_count

    cfg = SimpleNamespace(
        num_hidden_layers=4,
        layer_types=["mamba", "mamba", "full_attention", "mamba"],
    )
    assert softmax_layer_count(cfg) == 1


def test_layer_kind_labels_normalises_sliding_attention_to_attention():
    """The label vocabulary kept simple — sliding-window attention
    surfaces as ``"attention"`` in the returned labels, matching how
    consumers read the kind tag (used by calibration sweep diagnostics
    and by ``softmax_layer_count``)."""
    from types import SimpleNamespace

    from tardigrade_hooks._hidden_states import layer_kind_labels

    cfg = SimpleNamespace(
        num_hidden_layers=2,
        layers_block_type=["sliding_attention", "full_attention"],
    )
    labels = layer_kind_labels(cfg, n_hidden_states=3)
    # Index 0 is the embedding output; 1 and 2 are the per-layer outputs.
    assert labels == ["embedding", "attention", "attention"]


def test_layer_kind_labels_normalises_delta_net_to_recurrent():
    """Calibration sweep logs the layer kind for each candidate.
    ``"delta_net"`` / ``"gated_delta_net"`` / ``"mamba"`` are all
    recurrent / linear-attention variants and should surface as
    ``"recurrent"`` in the labels — otherwise the log shows ugly
    truncated strings like ``"delta_ne"`` if unknown labels are
    truncated to 8 chars."""
    from types import SimpleNamespace

    from tardigrade_hooks._hidden_states import layer_kind_labels

    cfg = SimpleNamespace(
        num_hidden_layers=3,
        layer_types=["delta_net", "gated_delta_net", "mamba"],
    )
    labels = layer_kind_labels(cfg, n_hidden_states=4)
    assert labels == ["embedding", "recurrent", "recurrent", "recurrent"]


def test_kp_n_softmax_layers_equals_n_layers_on_uniform_softmax(kps):
    """Uniform-softmax model: ``n_softmax_layers == n_layers`` so existing
    pack-integrity guards retain their strength. This is the regression
    contract the GPT-2 / Qwen3 paths rely on."""
    assert kps.n_softmax_layers == kps.n_layers


def test_kp_clone_cache_skips_layers_without_keys(kps):
    """RED contract: ``KnowledgePackStore._clone_cache`` must skip
    layers where ``.keys`` is None — recurrent slots in a hybrid
    model's DynamicCache, which 3 generate-side sites previously
    treated as if they always had a tensor and crashed mid-turn."""
    populated_layer_idx = 2

    class _SoftmaxLayer:
        def __init__(self, sl, kvd):
            # Tensor that supports .clone() and .shape — that's all the
            # clone path touches.
            self.keys = torch.zeros(1, kps.num_kv_heads, sl, kps.head_dim)
            self.values = torch.zeros(1, kps.num_kv_heads, sl, kps.head_dim)

    class _RecurrentLayer:
        # Recurrent slots in a populated DynamicCache: the slot exists
        # (so ``len(cache.layers)`` counts it) but K/V are None.
        keys = None
        values = None

    class _SparseCache:
        layers = [
            _RecurrentLayer(),
            _RecurrentLayer(),
            _SoftmaxLayer(5, kps.kv_dim),  # idx 2
            _RecurrentLayer(),
        ]

    clone = kps._clone_cache(_SparseCache())
    # The populated index survives; the recurrent indices don't crash.
    # We don't assert on the exact DynamicCache shape (transformers'
    # internals around skipped indices may vary by version) — only
    # that no AttributeError was raised and the populated layer made
    # it through.
    assert clone is not None


def test_kp_build_layer_payloads_skips_layers_without_keys(kps):
    """RED contract: ``KnowledgePackStore`` must build payloads via a
    softmax-only filter. Direct mock of the KV layout proves the filter
    activates without needing a real RecurrentGemma — layers 1 and 3 lack
    ``.keys``/``.values`` (recurrent shape); only 0 and 2 should appear."""
    seq_len = 5

    class _SoftmaxLayer:
        def __init__(self, num_kv_heads, head_dim, sl):
            self.keys = torch.zeros(1, num_kv_heads, sl, head_dim)
            self.values = torch.zeros(1, num_kv_heads, sl, head_dim)

    class _RecurrentLayer:
        # Recurrent layers expose state via different attributes
        # (e.g. ``recurrent_state``) — never ``.keys`` / ``.values``.
        recurrent_state = None

    class _FakeKV:
        layers = [
            _SoftmaxLayer(kps.num_kv_heads, kps.head_dim, seq_len),
            _RecurrentLayer(),
            _SoftmaxLayer(kps.num_kv_heads, kps.head_dim, seq_len),
            _RecurrentLayer(),
        ]

    payloads = kps._build_layer_payloads(_FakeKV(), seq_len)
    assert [li for li, _ in payloads] == [0, 2]


def test_move_cache_to_device_casts_to_target_dtype():
    """RED contract: ``_move_cache_to_device`` must accept a target dtype
    so callers can match the model's compute dtype. Multi-pack composers
    build fp32 tensors from numpy data; on a bf16/fp16 model (4-bit
    quantized included), feeding those into attention raises
    ``RuntimeError: Expected query, key, and value to have the same
    dtype``. This was the failure mode when wiring multi-pack retrieval
    into casper-spike on gemma-3-4b 4-bit.
    """
    from tardigrade_hooks.kp_injector import _move_cache_to_device
    from transformers import DynamicCache

    # Build a cache with explicitly fp32 tensors, then move + cast to bf16.
    src = DynamicCache()
    src.update(
        torch.zeros(1, 4, 5, 16, dtype=torch.float32),
        torch.zeros(1, 4, 5, 16, dtype=torch.float32),
        0,
    )
    src.update(
        torch.zeros(1, 4, 5, 16, dtype=torch.float32),
        torch.zeros(1, 4, 5, 16, dtype=torch.float32),
        1,
    )

    moved = _move_cache_to_device(src, device="cpu", dtype=torch.bfloat16)

    assert moved.layers[0].keys.dtype == torch.bfloat16
    assert moved.layers[0].values.dtype == torch.bfloat16
    assert moved.layers[1].keys.dtype == torch.bfloat16
    assert moved.layers[1].values.dtype == torch.bfloat16


# -- v0.4.1: multimodal-config support ----------------------------------------
#
# HuggingFace multimodal configs (Gemma3Config, LlavaConfig, Qwen2VLConfig, …)
# hold the language attributes inside ``cfg.text_config`` rather than at the
# root. ``cfg.get_text_config()`` is the canonical accessor — present on both
# multimodal and text-only configs. For text-only it returns ``self`` (no-op);
# for multimodal it returns the text-tower sub-config. Tardigrade-db routes
# every config-attribute read through it at intake.


def _mock_multimodal_model(n_text_layers=34, text_hidden=2560,
                           n_heads=16, n_kv_heads=8, head_dim=160,
                           block_types=None):
    """Mocked Gemma3-shape model: top-level config exposes get_text_config()
    returning the text-tower SimpleNamespace with the real attrs."""
    from types import SimpleNamespace
    text_cfg = SimpleNamespace(
        num_hidden_layers=n_text_layers,
        hidden_size=text_hidden,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
    )
    if block_types is not None:
        text_cfg.layers_block_type = block_types
    vision_cfg = SimpleNamespace(num_hidden_layers=27, hidden_size=1152)
    top_cfg = SimpleNamespace(
        text_config=text_cfg,
        vision_config=vision_cfg,
        name_or_path="mock/multimodal-test-model",
        get_text_config=lambda: text_cfg,
    )
    return SimpleNamespace(config=top_cfg)


def test_is_supported_drills_into_text_config_on_multimodal_model():
    """GIVEN a model whose config follows the multimodal Composite shape
    (top-level lacks num_hidden_layers; nested text_config carries it),
    WHEN is_supported runs,
    THEN it returns a supported verdict whose counts reflect the
    text tower, not the empty top-level."""
    from tardigrade_hooks import is_supported

    model = _mock_multimodal_model(n_text_layers=34, text_hidden=2560)

    # _WorkingTokenizer is defined in test_compatibility.py — duplicate the
    # minimal shape inline here to avoid cross-module test imports.
    class _Tok:
        chat_template = (
            '{% for message in messages %}{{ message["content"] }}{% endfor %}'
        )
        def apply_chat_template(self, messages, **kwargs):
            return "".join(m["content"] for m in messages)

    report = is_supported(model, _Tok())

    assert report.is_supported is True
    assert report.n_hidden_layers == 34
    assert report.n_softmax_layers == 34
    assert report.architecture == "uniform_softmax"
    assert report.blockers == ()


def test_kp_constructs_from_multimodal_model_with_text_tower_dimensions(engine, tokenizer):
    """GIVEN a mocked multimodal model + real engine + tokenizer,
    WHEN KnowledgePackStore is constructed,
    THEN __init__ does not raise; n_layers / hidden_size / num_kv_heads /
    kv_dim all reflect the text-tower values."""
    model = _mock_multimodal_model(
        n_text_layers=34, text_hidden=2560,
        n_heads=16, n_kv_heads=8, head_dim=160,
    )

    kps = KnowledgePackStore(engine, model, tokenizer, owner=1)

    assert kps.n_layers == 34
    assert kps.n_softmax_layers == 34
    assert kps.hidden_size == 2560
    assert kps.num_kv_heads == 8
    assert kps.head_dim == 160
    assert kps.kv_dim == 8 * 160


def test_kp_text_only_models_unaffected_by_text_config_resolution(kps):
    """REGRESSION: GIVEN the existing GPT-2 fixture (real HF config that
    exposes get_text_config returning self),
    WHEN KnowledgePackStore reads its dimensions,
    THEN n_layers / hidden_size match the canonical GPT-2 values
    (12 layers, 768 hidden)."""
    assert kps.n_layers == 12
    assert kps.n_softmax_layers == 12
    assert kps.hidden_size == 768


def test_select_query_layer_tokenizer_none_reads_text_config_on_multimodal():
    """GIVEN a multimodal model + tokenizer=None (hosted-API fallback path),
    WHEN select_query_layer runs,
    THEN it derives n_layers and hidden_size from text_config (34 layers,
    2560 hidden), not from the empty top-level config. Exercises the
    real production code path at calibrate.py — no spy, no monkeypatch,
    no forward pass (the tokenizer=None branch returns a heuristic
    CalibrationResult without sweeping)."""
    from tardigrade_hooks import select_query_layer

    model = _mock_multimodal_model(n_text_layers=34, text_hidden=2560)

    # tokenizer=None triggers the hosted-API fallback that reads
    # num_hidden_layers / hidden_size directly off model.config.
    result = select_query_layer(model, tokenizer=None, registry=None)

    assert result.n_layers == 34
    assert result.hidden_size == 2560


def test_is_supported_still_rejects_genuinely_empty_configs():
    """REGRESSION: GIVEN a model whose config has NO num_hidden_layers,
    NO text_config, AND NO get_text_config method (a genuinely broken
    or unrecognised config shape),
    WHEN is_supported runs,
    THEN it still returns is_supported=False with a blocker referencing
    num_hidden_layers. Guards against the multimodal-resolution helper
    accidentally masking legitimate failures."""
    from types import SimpleNamespace

    from tardigrade_hooks import is_supported

    class _Tok:
        chat_template = '{% for m in messages %}{{ m["content"] }}{% endfor %}'
        def apply_chat_template(self, messages, **kwargs):
            return "".join(m["content"] for m in messages)

    model = SimpleNamespace(config=SimpleNamespace())

    report = is_supported(model, _Tok())

    assert report.is_supported is False
    assert any("num_hidden_layers" in b for b in report.blockers)


def test_softmax_layer_count_drills_into_text_config_for_block_types():
    """GIVEN a multimodal-shaped config whose text_config carries
    layers_block_type (Gemma 3 / Llama 3.2 Vision style),
    WHEN softmax_layer_count runs against the top-level cfg,
    THEN it counts attention-typed layers from the text tower, not
    from the empty top-level."""
    from types import SimpleNamespace

    from tardigrade_hooks._hidden_states import softmax_layer_count

    text_cfg = SimpleNamespace(
        num_hidden_layers=4,
        layers_block_type=[
            "sliding_attention", "full_attention",
            "sliding_attention", "full_attention",
        ],
    )
    top_cfg = SimpleNamespace(
        text_config=text_cfg,
        get_text_config=lambda: text_cfg,
    )

    assert softmax_layer_count(top_cfg) == 4
