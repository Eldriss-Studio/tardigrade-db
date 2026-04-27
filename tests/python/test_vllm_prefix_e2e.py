# End-to-end proof: VLLMMemoryClient + vLLM generation with synthetic facts.
#
# Proves the full Path 2 loop: store facts → build governed prefix →
# prepend to prompt → vLLM generates → model recalls the stored fact.
#
# Requires: GPU + vLLM >= 0.9.0 + TardigradeDB built from source.
# Run with: pytest tests/python/test_vllm_prefix_e2e.py -v -m gpu

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db

vllm = pytest.importorskip("vllm", reason="vLLM not installed")
torch = pytest.importorskip("torch", reason="PyTorch not installed")

gpu = pytest.mark.gpu
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

MODEL_NAME = os.environ.get("TARDIGRADE_TEST_MODEL", "Qwen/Qwen3-0.6B")

# Fully synthetic — no model has ever seen these.
FACTS = [
    ("Agent Snibblex reported that the vault code is 9-Quornth-44",
     "What is the vault code?",
     "9-Quornth-44"),
    ("The capital of Vrenthar is Zyphlox-9",
     "What is the capital of Vrenthar?",
     "Zyphlox-9"),
    ("Dr. Molvax discovered the Krellian frequency at 8.31 plonks",
     "What is the Krellian frequency?",
     "8.31 plonks"),
]


@pytest.fixture(scope="module")
def db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="module")
def engine(db_path):
    return tardigrade_db.Engine(db_path)


@pytest.fixture(scope="module")
def llm():
    """Plain vLLM LLM — no TardigradeConnector needed.

    The prefix client works at the prompt level, not the connector level.
    vLLM's built-in prefix-cache handles KV reuse for repeated prefixes.
    """
    from vllm import LLM

    llm_instance = LLM(
        model=MODEL_NAME,
        max_model_len=512,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    yield llm_instance


def _store_fact(engine, text, owner=1):
    """Store a fact with a dummy retrieval key (prefix client uses text, not KV)."""
    key = np.random.randn(16).astype(np.float32)
    payload = np.zeros(16, dtype=np.float32)
    return engine.mem_write_pack(owner, key, [(0, payload)], 80.0, text)


@gpu
@requires_cuda
def test_cold_generation_does_not_know_synthetic_facts(llm):
    """Baseline: without memory prefix, the model cannot answer synthetic questions."""
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=50, temperature=0.0)

    for _, question, expected in FACTS:
        output = llm.generate([question], sp)[0].outputs[0].text
        assert expected.lower() not in output.lower(), (
            f"Cold generation already contains {expected!r} — "
            f"fact is not synthetic enough. Output: {output!r}"
        )


@gpu
@requires_cuda
def test_prefix_client_recalls_synthetic_facts(llm, engine, db_path):
    """THE PROOF: governed memory prefix makes the model recall synthetic facts.

    GIVEN synthetic facts stored in TardigradeDB,
    WHEN VLLMMemoryClient prepends them as a governed prefix,
    AND vLLM generates a response,
    THEN the output contains the synthetic answer.
    """
    from vllm import SamplingParams
    from tardigrade_vllm.prefix_client import VLLMMemoryClient

    sp = SamplingParams(max_tokens=100, temperature=0.0)

    for fact_text, _, _ in FACTS:
        _store_fact(engine, fact_text)

    client = VLLMMemoryClient(engine, owner=1)

    correct = 0
    results = []
    for fact_text, question, expected in FACTS:
        prompt = client.prepare_prompt(question)
        output = llm.generate([prompt], sp)[0].outputs[0].text

        # Strip thinking tags if present (Qwen3)
        if "</think>" in output:
            output = output.split("</think>")[-1].strip()

        hit = expected.lower() in output.lower()
        if hit:
            correct += 1
        results.append((expected, hit, output[:80]))

    for expected, hit, snippet in results:
        mark = "Y" if hit else "N"
        print(f"  {mark}  {expected:<20} → {snippet}")

    assert correct >= 2, (
        f"Only {correct}/{len(FACTS)} synthetic facts recalled via prefix. "
        f"Expected at least 2. Results: {results}"
    )


@gpu
@requires_cuda
def test_prefix_is_deterministic_across_requests(llm, engine, db_path):
    """Same prefix produces identical output on repeated requests.

    This validates that vLLM's prefix-cache can work: if the prefix
    text is deterministic, the token sequence is identical, and vLLM
    can cache the KV.
    """
    from vllm import SamplingParams
    from tardigrade_vllm.prefix_client import VLLMMemoryClient

    if engine.pack_count() == 0:
        _store_fact(engine, "The antidote requires 3 drops of Yombliquid-X")

    client = VLLMMemoryClient(engine, owner=1)
    sp = SamplingParams(max_tokens=30, temperature=0.0)
    question = "What is the antidote dosage?"

    prompt1 = client.prepare_prompt(question)
    prompt2 = client.prepare_prompt(question)
    assert prompt1 == prompt2, "Prefix must be deterministic"

    out1 = llm.generate([prompt1], sp)[0].outputs[0].text
    out2 = llm.generate([prompt2], sp)[0].outputs[0].text
    assert out1 == out2, (
        f"Same prefix + query should produce identical output.\n"
        f"Run 1: {out1!r}\nRun 2: {out2!r}"
    )


@gpu
@requires_cuda
def test_owner_isolation_in_generation(llm):
    """Owner 1's memories don't leak into Owner 2's prefix."""
    from vllm import SamplingParams
    from tardigrade_vllm.prefix_client import VLLMMemoryClient

    # Fresh engine so prior test facts don't confuse the model.
    with tempfile.TemporaryDirectory() as iso_dir:
        iso_engine = tardigrade_db.Engine(iso_dir)
        _store_fact(iso_engine, "Owner1 secret: the password is Glindavar-77", owner=1)
        _store_fact(iso_engine, "Owner2 secret: the password is Thraxium-99", owner=2)

        client1 = VLLMMemoryClient(iso_engine, owner=1)
        client2 = VLLMMemoryClient(iso_engine, owner=2)

        sp = SamplingParams(max_tokens=50, temperature=0.0)

        p1 = client1.prepare_prompt("What is the password? /no_think")
        p2 = client2.prepare_prompt("What is the password? /no_think")

        assert "Glindavar-77" in p1
        assert "Thraxium-99" not in p1
        assert "Thraxium-99" in p2
        assert "Glindavar-77" not in p2

        out1 = llm.generate([p1], sp)[0].outputs[0].text
        out2 = llm.generate([p2], sp)[0].outputs[0].text

        if "</think>" in out1:
            out1 = out1.split("</think>")[-1].strip()
        if "</think>" in out2:
            out2 = out2.split("</think>")[-1].strip()

        assert "Glindavar-77" in out1 or "glindavar" in out1.lower(), (
            f"Owner 1 should recall Glindavar-77, got: {out1!r}"
        )
        assert "Thraxium-99" in out2 or "thraxium" in out2.lower(), (
            f"Owner 2 should recall Thraxium-99, got: {out2!r}"
        )
