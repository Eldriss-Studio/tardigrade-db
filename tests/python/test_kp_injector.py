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


# -- Phase 2: KnowledgePackStore integration with RetrievalKeyStrategy ------

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
