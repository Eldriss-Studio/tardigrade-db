"""Reusable HTTP adapter base for optional external systems."""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

from tdb_bench.contracts import BenchmarkAdapter
from tdb_bench.models import AdapterQueryResult, BenchmarkItem


class OptionalHttpAdapter(BenchmarkAdapter):
    """Adapter for real-optional external services.

    If endpoint is absent, operations produce explicit `skipped` outcomes.
    """

    name = "external"
    env_base_url = ""
    env_api_key = ""
    require_api_key = False

    def __init__(self, timeout_seconds: int = 5) -> None:
        self.timeout_seconds = timeout_seconds
        self._base_url = os.getenv(self.env_base_url, "").strip()
        self._api_key = os.getenv(self.env_api_key, "").strip()

    def ingest(self, items: list[BenchmarkItem]) -> None:
        if not self._is_configured():
            return
        payload = {
            "items": [
                {
                    "id": it.item_id,
                    "dataset": it.dataset,
                    "context": it.context,
                    "question": it.question,
                    "ground_truth": it.ground_truth,
                }
                for it in items
            ]
        }
        self._request_json("POST", "/ingest", payload=payload)

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
                "/query",
                payload={
                    "question": item.question,
                    "context": item.context,
                    "top_k": top_k,
                },
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            return AdapterQueryResult(
                answer=str(body.get("answer", "")),
                evidence=[str(x) for x in body.get("evidence", [])],
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

    def metadata(self) -> dict[str, str]:
        return {
            "adapter": self.name,
            "type": "optional_http",
            "configured": str(self._is_configured()).lower(),
            "base_url": self._base_url or "",
        }

    def _is_configured(self) -> bool:
        if not self._base_url:
            return False
        if self.require_api_key and not self._api_key:
            return False
        return True

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        params: dict[str, str | int | float | bool] | None = None,
    ) -> dict | list:
        query = ""
        if params:
            query = "?" + urllib.parse.urlencode(params)
        url = f"{self._base_url.rstrip('/')}{path}{query}"

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
        if not raw:
            return {}
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return parsed
        return {}
