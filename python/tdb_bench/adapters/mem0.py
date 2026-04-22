"""Mem0 OSS adapter (Adapter pattern)."""

from __future__ import annotations

import os
import time

from tdb_bench.models import AdapterQueryResult, BenchmarkItem

from .base_http import OptionalHttpAdapter


class Mem0Adapter(OptionalHttpAdapter):
    name = "mem0_oss"
    env_base_url = "MEM0_BASE_URL"
    env_api_key = "MEM0_API_KEY"
    require_api_key = False

    def __init__(self, timeout_seconds: int = 5) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self._user_id = os.getenv("MEM0_BENCH_USER_ID", "tdb-bench")

    def ingest(self, items: list[BenchmarkItem]) -> None:
        if not self._is_configured():
            return
        for item in items:
            context = item.context.strip()
            if not context:
                continue
            self._request_json(
                "POST",
                "/memories",
                payload={
                    "messages": [{"role": "user", "content": context}],
                    "user_id": self._user_id,
                    "metadata": {"item_id": item.item_id, "dataset": item.dataset},
                },
            )

    def query(self, item: BenchmarkItem, top_k: int) -> AdapterQueryResult:
        if not self._is_configured():
            return AdapterQueryResult(
                answer="",
                evidence=[],
                latency_ms=0.0,
                status="skipped",
                error="SERVICE_UNAVAILABLE: missing endpoint",
            )

        start = time.perf_counter()
        try:
            body = self._request_json(
                "POST",
                "/search",
                payload={
                    "query": item.question,
                    "user_id": self._user_id,
                },
            )
            rows = body if isinstance(body, list) else body.get("results", [])
            evidence = self._extract_evidence(rows, limit=top_k)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return AdapterQueryResult(
                answer=evidence[0] if evidence else "",
                evidence=evidence,
                latency_ms=latency_ms,
                status="ok",
                error=None,
            )
        except Exception as exc:  # pragma: no cover - network variability
            latency_ms = (time.perf_counter() - start) * 1000.0
            return AdapterQueryResult(
                answer="",
                evidence=[],
                latency_ms=latency_ms,
                status="failed",
                error=f"QUERY_FAILED: {exc}",
            )

    def reset(self) -> None:
        if not self._is_configured():
            return
        self._request_json("POST", "/reset", payload={})

    @staticmethod
    def _extract_evidence(rows: object, limit: int) -> list[str]:
        if not isinstance(rows, list):
            return []
        evidence: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = (
                row.get("memory")
                or row.get("text")
                or row.get("content")
                or row.get("value")
                or row.get("fact")
            )
            if text:
                evidence.append(str(text))
            if len(evidence) >= limit:
                break
        return evidence
