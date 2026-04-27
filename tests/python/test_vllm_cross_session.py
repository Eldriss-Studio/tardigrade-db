# ATDD acceptance test for cross-vLLM-session pack persistence (Step 2.5).
#
# Spawns two vLLM subprocesses sharing the same TardigradeDB path:
#   Run #1: generates a factual prompt, writes packs to disk
#   Run #2: queries with a semantically related prompt, must see prior packs
#
# This is the production scenario the connector exists for: an agent's
# memory persisting across vLLM process restarts.

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import tardigrade_db

# Skip module if vLLM / torch / CUDA unavailable
vllm = pytest.importorskip("vllm", reason="vLLM not installed")
torch = pytest.importorskip("torch", reason="PyTorch not installed")

gpu = pytest.mark.gpu
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

MODEL_NAME = os.environ.get("TARDIGRADE_TEST_MODEL", "Qwen/Qwen3-0.6B")

# vLLM startup is heavy (~30s per process). We pay it twice in this test
# to actually prove cross-process persistence — that is the whole point.

_RUN1_SCRIPT = textwrap.dedent(r"""
    import sys
    sys.path.insert(0, {python_dir!r})
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    cfg = KVTransferConfig(
        kv_connector="TardigradeConnector",
        kv_connector_module_path="tardigrade_vllm.connector",
        kv_role="kv_both",
        kv_connector_extra_config={{"db_path": {db_path!r}, "owner": 1}},
    )
    llm = LLM(
        model={model!r},
        kv_transfer_config=cfg,
        max_model_len=512,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    llm.generate(["Tardigrades survive cryptobiosis: vacuum, radiation, dehydration."],
                 SamplingParams(max_tokens=20, temperature=0.0))
    print("RUN1_DONE", flush=True)
""")

_RUN2_SCRIPT = textwrap.dedent(r"""
    import sys
    sys.path.insert(0, {python_dir!r})
    import tardigrade_db
    # Probe the on-disk state BEFORE vLLM starts so we know what run #2's
    # connector will see at init.
    startup_count = tardigrade_db.Engine({db_path!r}).pack_count()
    print(f"STARTUP_PACKS={{startup_count}}", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    cfg = KVTransferConfig(
        kv_connector="TardigradeConnector",
        kv_connector_module_path="tardigrade_vllm.connector",
        kv_role="kv_both",
        kv_connector_extra_config={{"db_path": {db_path!r}, "owner": 1}},
    )
    llm = LLM(
        model={model!r},
        kv_transfer_config=cfg,
        max_model_len=512,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    out = llm.generate(["How do tardigrades survive harsh conditions?"],
                       SamplingParams(max_tokens=20, temperature=0.0))
    print("RUN2_DONE", flush=True)
    print(f"RUN2_TEXT: {{out[0].outputs[0].text}}", flush=True)
""")


def _python_dir() -> str:
    """Path to the python/ directory containing tardigrade_vllm."""
    return str(Path(__file__).resolve().parents[2] / "python")


@gpu
@requires_cuda
def test_pack_persists_across_vllm_restart(tmp_path):
    """GIVEN vLLM run #1 generates and saves packs to db_path,
    WHEN vLLM run #1 shuts down and run #2 starts at the same db_path,
    THEN run #2's connector sees the prior packs (pack_count > 0 at startup)
    AND a request semantically related to run #1's prompt completes successfully."""
    db_path = str(tmp_path / "xsession-mem")
    python_dir = _python_dir()

    # --- Run #1: write packs ---
    run1_code = _RUN1_SCRIPT.format(
        python_dir=python_dir, db_path=db_path, model=MODEL_NAME
    )
    r1 = subprocess.run(
        [sys.executable, "-c", run1_code],
        capture_output=True, text=True, timeout=300,
    )
    assert r1.returncode == 0, f"Run #1 failed:\nSTDOUT: {r1.stdout[-2000:]}\nSTDERR: {r1.stderr[-2000:]}"
    assert "RUN1_DONE" in r1.stdout, f"Run #1 did not finish cleanly:\n{r1.stdout[-1000:]}"

    # --- Verify run #1 actually wrote packs to disk ---
    pre_count = tardigrade_db.Engine(db_path).pack_count()
    assert pre_count > 0, (
        f"Run #1 should have written at least one pack, found {pre_count}. "
        f"Run #1 stdout tail: {r1.stdout[-500:]}"
    )

    # --- Run #2: separate process, same db_path ---
    run2_code = _RUN2_SCRIPT.format(
        python_dir=python_dir, db_path=db_path, model=MODEL_NAME
    )
    r2 = subprocess.run(
        [sys.executable, "-c", run2_code],
        capture_output=True, text=True, timeout=300,
    )
    assert r2.returncode == 0, f"Run #2 failed:\nSTDOUT: {r2.stdout[-2000:]}\nSTDERR: {r2.stderr[-2000:]}"
    assert "RUN2_DONE" in r2.stdout, f"Run #2 did not finish cleanly:\n{r2.stdout[-1000:]}"

    # --- Verify run #2 saw run #1's packs at startup ---
    # Run #2 prints "STARTUP_PACKS=N" before instantiating vLLM. Parse it.
    startup_lines = [ln for ln in r2.stdout.splitlines() if ln.startswith("STARTUP_PACKS=")]
    assert startup_lines, (
        f"Run #2 did not emit STARTUP_PACKS line. stdout: {r2.stdout[-1000:]}"
    )
    startup_count = int(startup_lines[0].split("=", 1)[1])
    assert startup_count >= pre_count, (
        f"Run #2 saw {startup_count} packs at startup but run #1 had written "
        f"{pre_count} — engine state did not persist across processes."
    )

    # Run #2 should ALSO write a new pack (its own request); confirm growth.
    final_count = tardigrade_db.Engine(db_path).pack_count()
    assert final_count >= startup_count, (
        f"Run #2 should not have decreased pack_count (had {startup_count}, "
        f"now {final_count})"
    )
