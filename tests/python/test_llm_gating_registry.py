"""ATDD: registry + end-to-end LLM-gated bench smoke (Slice L8).

Two test classes:

1. ``TestRegistryWiring`` — ``RegistryFactory.create_adapter
   ("tardigrade-llm-gated")`` returns a ``RetrieveThenReadAdapter``
   over a ``TardigradeAdapter`` with a generator from the env factory.

2. ``TestEndToEndSmoke`` — using ``TDB_BENCH_FORCE_FALLBACK=1`` to skip
   the GPU model load and ``TDB_LLM_GATE_PROVIDER=mock`` to skip API
   calls, run ingest → query → assert the mocked answer surfaces.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from tdb_bench.adapters import TardigradeAdapter
from tdb_bench.adapters.retrieve_then_read import RetrieveThenReadAdapter
from tdb_bench.models import BenchmarkItem
from tdb_bench.registry import RegistryFactory


_MOCK_RESPONSE = "I don't know"  # matches factory _MOCK_DEFAULT_RESPONSE


class TestRegistryWiring:
    @patch.dict(os.environ, {"TDB_LLM_GATE_PROVIDER": "mock"}, clear=False)
    def test_returns_retrieve_then_read_decorator(self):
        adapter = RegistryFactory.create_adapter(
            system="tardigrade-llm-gated", timeout_seconds=5
        )
        assert isinstance(adapter, RetrieveThenReadAdapter)

    @patch.dict(os.environ, {"TDB_LLM_GATE_PROVIDER": "mock"}, clear=False)
    def test_metadata_records_answerer_model_and_template_version(self):
        adapter = RegistryFactory.create_adapter(
            system="tardigrade-llm-gated", timeout_seconds=5
        )
        meta = adapter.metadata()
        assert "answerer_model" in meta
        assert "prompt_template_version" in meta
        assert meta["answerer_model"] == "mock"

    @patch.dict(os.environ, {"TDB_LLM_GATE_PROVIDER": "mock"}, clear=False)
    def test_inner_is_tardigrade_adapter(self):
        adapter = RegistryFactory.create_adapter(
            system="tardigrade-llm-gated", timeout_seconds=5
        )
        # Decorator's inner is the real TardigradeAdapter — proves the
        # registry didn't accidentally swap to a stub or in-memory adapter.
        assert isinstance(adapter._inner, TardigradeAdapter)  # noqa: SLF001

    def test_baseline_tardigrade_system_still_works(self):
        # Sanity: adding the new system didn't break the existing one.
        adapter = RegistryFactory.create_adapter(system="tardigrade", timeout_seconds=5)
        assert isinstance(adapter, TardigradeAdapter)
        assert not isinstance(adapter, RetrieveThenReadAdapter)


class TestEndToEndSmoke:
    @patch.dict(
        os.environ,
        {
            "TDB_BENCH_FORCE_FALLBACK": "1",  # in-memory path, no GPU/torch
            "TDB_LLM_GATE_PROVIDER": "mock",
        },
        clear=False,
    )
    def test_smoke_ingest_and_query_uses_mock_answer(self):
        adapter = RegistryFactory.create_adapter(
            system="tardigrade-llm-gated", timeout_seconds=5
        )
        items = [
            BenchmarkItem(
                item_id="x1",
                dataset="locomo",
                context="Alice moved to Berlin in 2021.",
                question="Where did Alice move?",
                ground_truth="Berlin",
            ),
            BenchmarkItem(
                item_id="x2",
                dataset="locomo",
                context="Bob's favorite language is Rust.",
                question="What is Bob's favorite language?",
                ground_truth="Rust",
            ),
        ]

        adapter.ingest(items)
        result = adapter.query(items[0], top_k=3)

        # Answer is from the mock — proves the LLM-gating seam is
        # exercising the generator, not falling back to the inner
        # adapter's mapped.ground_truth.
        assert result.status == "ok"
        assert result.answer == _MOCK_RESPONSE
        # Inner adapter populated evidence — proves retrieval ran.
        assert len(result.evidence) > 0
