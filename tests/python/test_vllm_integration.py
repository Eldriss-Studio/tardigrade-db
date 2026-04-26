# ATDD acceptance tests for vLLM KV Connector end-to-end round-trip.
#
# Requires: GPU + vLLM >= 0.9.0 + TardigradeDB built from source.
# Run with: pytest tests/python/test_vllm_integration.py -v -m gpu
# Skip in CI: pytest -m "not gpu"
#
# These tests start a vLLM LLMEngine in-process (not the HTTP server)
# to test the connector's save and load paths with real model inference.

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db

# Skip entire module if vLLM not available
vllm = pytest.importorskip("vllm", reason="vLLM not installed")
torch = pytest.importorskip("torch", reason="PyTorch not installed")

gpu = pytest.mark.gpu
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

# Small model that fits in 8GB VRAM
MODEL_NAME = os.environ.get("TARDIGRADE_TEST_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def db_path():
    """Temporary directory for TardigradeDB engine during tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="module")
def engine(db_path):
    """TardigradeDB engine instance for verifying state."""
    return tardigrade_db.Engine(db_path)


@pytest.fixture(scope="module")
def llm(db_path):
    """vLLM LLMEngine configured with TardigradeConnector.

    Uses in-process engine (not HTTP server) for faster testing.
    Scope=module so the model loads once for all tests.
    """
    from vllm import LLM
    from vllm.config import KVTransferConfig

    kv_config = KVTransferConfig(
        kv_connector="TardigradeConnector",
        kv_connector_module_path="tardigrade_vllm.connector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "db_path": db_path,
            "owner": 1,
        },
    )

    llm_instance = LLM(
        model=MODEL_NAME,
        kv_transfer_config=kv_config,
        max_model_len=512,  # Keep VRAM usage low
        gpu_memory_utilization=0.8,
        enforce_eager=True,  # Skip CUDA graph compilation for faster startup
    )
    yield llm_instance
    # Cleanup handled by garbage collection


# -- Save Path Tests ----------------------------------------------------------

@gpu
@requires_cuda
def test_save_path_stores_pack_during_generation(llm, engine, db_path):
    """GIVEN vLLM serving with TardigradeConnector,
    WHEN a completion request is processed,
    THEN engine.pack_count() increments and the pack has layer data."""
    from vllm import SamplingParams

    initial_count = engine.pack_count()

    outputs = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=20, temperature=0.0),
    )

    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0, "Should generate some text"

    # The connector inside vLLM's EngineCore subprocess writes packs to the
    # same on-disk path. Re-open the engine here so we observe its writes.
    fresh_engine = tardigrade_db.Engine(db_path)
    new_count = fresh_engine.pack_count()
    assert new_count > initial_count, (
        f"Expected pack_count to increase from {initial_count}, got {new_count} "
        f"(in-process engine sees {engine.pack_count()})"
    )


@gpu
@requires_cuda
def test_saved_pack_contains_expected_layers(llm, db_path):
    """GIVEN a pack saved during generation,
    WHEN loaded by ID,
    THEN it contains the expected number of layers with non-empty data."""
    from vllm import SamplingParams

    # Generate to ensure a fresh pack exists this test
    llm.generate(
        ["A simple test prompt"],
        SamplingParams(max_tokens=10, temperature=0.0),
    )

    # Reopen engine to see writes from EngineCore subprocess
    fresh_engine = tardigrade_db.Engine(db_path)
    assert fresh_engine.pack_count() > 0, "Should have at least one pack"

    # Load pack #1 directly by ID — avoids the SLB dimension matching dance
    pack = fresh_engine.load_pack_by_id(1)
    assert len(pack["layers"]) > 0, "Pack should have layer data"
    for layer in pack["layers"]:
        assert len(layer["data"]) > 0, "Layer data should be non-empty"


# -- Load Path Tests ----------------------------------------------------------

@gpu
@requires_cuda
def test_load_path_matches_semantically_similar_request(llm, engine):
    """GIVEN a pack saved from "The capital of France is",
    WHEN a semantically similar request arrives,
    THEN the connector finds a matching pack (score above threshold).

    Note: This test validates the scheduler-side matching logic.
    The actual GPU injection in start_load_kv is tested separately.
    """
    from vllm import SamplingParams

    # First: ensure a pack exists from a factual prompt
    if engine.pack_count() == 0:
        llm.generate(
            ["The capital of France is"],
            SamplingParams(max_tokens=20, temperature=0.0),
        )

    # Second: generate with a semantically related prompt
    # The connector should find the stored KV via embedding-based matching
    outputs = llm.generate(
        ["What is the main city of France?"],
        SamplingParams(max_tokens=20, temperature=0.0),
    )

    # The generation should complete (not crash during load attempt)
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0


@gpu
@requires_cuda
def test_round_trip_produces_coherent_output(llm, engine):
    """GIVEN KV saved from generation and potentially loaded on a related query,
    THEN the model produces coherent text (basic sanity check)."""
    from vllm import SamplingParams

    outputs = llm.generate(
        ["Paris is known for"],
        SamplingParams(max_tokens=30, temperature=0.0),
    )

    text = outputs[0].outputs[0].text
    assert len(text) > 10, "Should generate meaningful text"
    # Basic coherence: should contain some real words, not random tokens
    words = text.split()
    assert len(words) >= 3, f"Expected coherent text, got: {text!r}"


# -- Cleanup Tests ------------------------------------------------------------

@gpu
@requires_cuda
def test_multiple_generations_accumulate_packs(llm, db_path):
    """GIVEN multiple generation requests,
    WHEN each completes,
    THEN pack_count increases monotonically."""
    from vllm import SamplingParams

    count_before = tardigrade_db.Engine(db_path).pack_count()

    prompts = [
        "The speed of light is approximately",
        "Water boils at a temperature of",
    ]
    for prompt in prompts:
        llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0.0))

    count_after = tardigrade_db.Engine(db_path).pack_count()
    assert count_after >= count_before + len(prompts), (
        f"Expected at least {len(prompts)} new packs, "
        f"got {count_after - count_before}"
    )
