"""ATDD tests for lexical RLS — Bridge pattern: same strategies, lexical retrieval substrate."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from tdb_bench.models import BenchmarkItem


def _item(item_id: str, context: str, question: str, ground_truth: str) -> BenchmarkItem:
    return BenchmarkItem(
        item_id=item_id,
        dataset="test",
        context=context,
        question=question,
        ground_truth=ground_truth,
    )


class TestScoredBestMatch:
    def test_returns_scored_tuples(self):
        from tdb_bench.adapters.tardigrade import _InMemoryStore

        store = _InMemoryStore()
        store.insert(_item("a", "Sonia ran a marathon in the highlands", "q", "highlands"))
        store.insert(_item("b", "Bob likes chess and reading", "q", "chess"))

        results = store.scored_best_match("marathon highlands", top_k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        score_a, item_a = results[0]
        assert isinstance(score_a, int)
        assert isinstance(item_a, BenchmarkItem)
        assert score_a > results[1][0]


class TestLexicalReformulationSearch:
    def test_finds_via_reformulation(self):
        from tdb_bench.adapters.tardigrade import (
            _InMemoryStore,
            _LexicalReformulationSearch,
        )

        store = _InMemoryStore()
        store.insert(_item("marathon", "Sonia completed her first ultramarathon in the Scottish Highlands", "q", "Scottish Highlands"))
        store.insert(_item("chess", "Bob plays chess every Tuesday at the library", "q", "Tuesday"))

        mock_strategy = MagicMock()
        mock_strategy.reformulate.return_value = ["ultramarathon Scottish Highlands"]

        rls = _LexicalReformulationSearch(store, [mock_strategy])
        answer, evidence = rls.query("What athletic achievements does Sonia have?", top_k=3)

        mock_strategy.reformulate.assert_called_once_with("What athletic achievements does Sonia have?")
        assert answer == "Scottish Highlands"

    def test_picks_highest_score(self):
        from tdb_bench.adapters.tardigrade import (
            _InMemoryStore,
            _LexicalReformulationSearch,
        )

        store = _InMemoryStore()
        store.insert(_item("a", "marathon running endurance race training", "q", "answer_a"))
        store.insert(_item("b", "ultramarathon Scottish Highlands trail race endurance", "q", "answer_b"))

        mock_strategy = MagicMock()
        mock_strategy.reformulate.return_value = [
            "running",
            "ultramarathon Scottish Highlands trail race endurance",
        ]

        rls = _LexicalReformulationSearch(store, [mock_strategy])
        answer, _ = rls.query("sports", top_k=3)
        assert answer == "answer_b"

    def test_empty_reformulations_falls_back(self):
        from tdb_bench.adapters.tardigrade import (
            _InMemoryStore,
            _LexicalReformulationSearch,
        )

        store = _InMemoryStore()
        store.insert(_item("a", "Sonia ran a marathon", "q", "marathon_answer"))

        mock_strategy = MagicMock()
        mock_strategy.reformulate.return_value = []

        rls = _LexicalReformulationSearch(store, [mock_strategy])
        answer, _ = rls.query("Sonia marathon", top_k=3)
        assert answer == "marathon_answer"


def _mock_deepseek_response(content: str):
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestAdapterIntegration:
    def test_in_memory_agent_mode_has_lexical_rls(self):
        from tdb_bench.adapters.tardigrade import _LexicalReformulationSearch

        with patch.dict(os.environ, {
            "TDB_BENCH_FORCE_FALLBACK": "1",
            "TDB_RLS_MODE": "agent",
            "DEEPSEEK_API_KEY": "test-key",
        }):
            from tdb_bench.adapters import tardigrade as mod
            orig_mode = mod._RLS_MODE
            mod._RLS_MODE = "agent"
            try:
                from tdb_bench.adapters.tardigrade import TardigradeAdapter
                adapter = TardigradeAdapter()
                assert adapter._mode == "in_memory"
                assert hasattr(adapter, "_lexical_rls")
                assert isinstance(adapter._lexical_rls, _LexicalReformulationSearch)
            finally:
                mod._RLS_MODE = orig_mode

    def test_in_memory_query_calls_deepseek(self):
        deepseek_response = (
            "What marathon races has Sonia completed?\n"
            "Tell me about Sonia's running events\n"
            "Sonia's ultramarathon results"
        )
        with patch.dict(os.environ, {
            "TDB_BENCH_FORCE_FALLBACK": "1",
            "TDB_RLS_MODE": "agent",
            "DEEPSEEK_API_KEY": "test-key",
        }):
            from tdb_bench.adapters import tardigrade as mod
            orig_mode = mod._RLS_MODE
            mod._RLS_MODE = "agent"
            try:
                from tdb_bench.adapters.tardigrade import TardigradeAdapter
                adapter = TardigradeAdapter()

                items = [
                    _item("m", "Sonia completed her first ultramarathon in Scotland", "q", "Scotland"),
                    _item("c", "Bob plays chess on Tuesdays", "q", "Tuesday"),
                ]
                adapter.ingest(items)

                with patch("urllib.request.urlopen", return_value=_mock_deepseek_response(deepseek_response)):
                    result = adapter.query(
                        _item("q", "", "What athletic achievements does Sonia have?", "Scotland"),
                        top_k=3,
                    )

                assert result.status == "ok"
                assert result.latency_ms > 0
            finally:
                mod._RLS_MODE = orig_mode

    def test_keyword_works_in_memory(self):
        from tdb_bench.adapters.tardigrade import _LexicalReformulationSearch

        with patch.dict(os.environ, {"TDB_BENCH_FORCE_FALLBACK": "1", "TDB_RLS_MODE": "keyword"}):
            from tdb_bench.adapters import tardigrade as mod
            orig_mode = mod._RLS_MODE
            mod._RLS_MODE = "keyword"
            try:
                from tdb_bench.adapters.tardigrade import TardigradeAdapter
                adapter = TardigradeAdapter()
                assert adapter._mode == "in_memory"
                assert isinstance(adapter._lexical_rls, _LexicalReformulationSearch)
            finally:
                mod._RLS_MODE = orig_mode
