"""Letta adapter (Adapter pattern)."""

from __future__ import annotations

import os
import re
import time

from tdb_bench.models import AdapterQueryResult, BenchmarkItem

from .base_http import OptionalHttpAdapter


class LettaAdapter(OptionalHttpAdapter):
    name = "letta"
    env_base_url = "LETTA_BASE_URL"
    env_api_key = "LETTA_API_KEY"
    require_api_key = False
    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    def __init__(self, timeout_seconds: int = 5) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self._agent_name = os.getenv("LETTA_BENCH_AGENT_NAME", "tdb-bench")
        self._agent_model = os.getenv("LETTA_BENCH_MODEL", "letta/letta-free")
        self._ingest_max_chars = max(1, int(os.getenv("LETTA_INGEST_MAX_CHARS", "4000")))
        self._agent_id: str | None = None

    def ingest(self, items: list[BenchmarkItem]) -> None:
        if not self._is_configured():
            return
        agent_id = self._ensure_agent()
        for item in items:
            context = item.context.strip()
            if not context:
                continue
            for chunk in self._iter_context_chunks(context):
                self._request_json(
                    "POST",
                    f"/v1/agents/{agent_id}/archival-memory",
                    payload={"text": chunk},
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
            agent_id = self._ensure_agent()
            evidence = self._query_evidence_chain(agent_id=agent_id, question=item.question, top_k=top_k)
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
        if self._agent_id:
            self._request_json("DELETE", f"/v1/agents/{self._agent_id}")
            self._agent_id = None

    def _ensure_agent(self) -> str:
        if self._agent_id:
            return self._agent_id

        maybe_existing = self._request_json("GET", "/v1/agents/", params={"name": self._agent_name, "limit": 1})
        if isinstance(maybe_existing, list) and maybe_existing:
            existing_id = maybe_existing[0].get("id")
            if existing_id:
                self._agent_id = str(existing_id)
                return self._agent_id

        created = self._request_json(
            "POST",
            "/v1/agents/",
            payload={"name": self._agent_name, "model": self._agent_model},
        )
        if not isinstance(created, dict) or "id" not in created:
            raise RuntimeError("Letta agent creation returned no id")
        self._agent_id = str(created["id"])
        return self._agent_id

    def _query_evidence_chain(self, agent_id: str, question: str, top_k: int) -> list[str]:
        # Chain-of-responsibility retrieval strategy:
        # semantic full question -> semantic keyword fallbacks -> lexical full question -> lexical keywords.
        for q in [question, *self._question_terms(question)]:
            semantic = self._request_json(
                "GET",
                f"/v1/agents/{agent_id}/archival-memory/search",
                params={"query": q, "top_k": top_k},
            )
            evidence = self._extract_semantic_evidence(semantic, limit=top_k)
            if evidence:
                return evidence

        lexical = self._request_json(
            "GET",
            f"/v1/agents/{agent_id}/archival-memory",
            params={"search": question, "limit": top_k},
        )
        evidence = self._extract_evidence(lexical, limit=top_k)
        if evidence:
            return evidence

        for q in self._question_terms(question):
            lexical = self._request_json(
                "GET",
                f"/v1/agents/{agent_id}/archival-memory",
                params={"search": q, "limit": top_k},
            )
            evidence = self._extract_evidence(lexical, limit=top_k)
            if evidence:
                return evidence
        return []

    @classmethod
    def _question_terms(cls, question: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[A-Za-z0-9#]+", question):
            normalized = token.lower().strip("#")
            if len(normalized) < 4 or normalized in cls._STOPWORDS:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            terms.append(normalized)
        return terms[:5]

    @staticmethod
    def _extract_evidence(rows: object, limit: int) -> list[str]:
        if not isinstance(rows, list):
            return []
        evidence: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = row.get("text") or row.get("memory") or row.get("content") or row.get("value")
            if text:
                evidence.append(str(text))
            if len(evidence) >= limit:
                break
        return evidence

    @staticmethod
    def _extract_semantic_evidence(payload: object, limit: int) -> list[str]:
        if not isinstance(payload, dict):
            return []
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            return []
        evidence: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = row.get("text") or row.get("memory") or row.get("content") or row.get("value")
            if text:
                evidence.append(str(text))
            if len(evidence) >= limit:
                break
        return evidence

    def _iter_context_chunks(self, context: str) -> list[str]:
        # Chunking strategy: prefer whitespace boundaries while honoring service payload caps.
        chunks: list[str] = []
        start = 0
        length = len(context)
        max_chars = self._ingest_max_chars
        while start < length:
            end = min(start + max_chars, length)
            if end < length:
                split = context.rfind(" ", start, end)
                if split > start:
                    end = split
            chunk = context[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
            while start < length and context[start].isspace():
                start += 1
        return chunks
