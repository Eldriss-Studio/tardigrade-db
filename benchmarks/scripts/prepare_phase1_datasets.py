#!/usr/bin/env python3
"""Prepare normalized Phase-1 benchmark JSONL files from official sources.

Output schema per line:
{
  "id": "...",
  "dataset": "locomo|longmemeval",
  "context": "...",
  "question": "...",
  "ground_truth": "..."
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase-1 benchmark datasets.")
    parser.add_argument(
        "--locomo-json",
        required=True,
        help="Path to LoCoMo JSON (e.g., data/locomo10.json).",
    )
    parser.add_argument(
        "--longmemeval-json",
        required=True,
        help="Path to LongMemEval JSON (e.g., longmemeval_oracle.json).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where converted JSONL files and manifest will be written.",
    )
    parser.add_argument(
        "--locomo-context",
        choices=["evidence", "full"],
        default="evidence",
        help="LoCoMo context construction mode (default: evidence).",
    )
    parser.add_argument(
        "--longmemeval-max-context-chars",
        type=int,
        default=0,
        help="Optional context char cap for LongMemEval rows (0 = no cap).",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_normalize_text(v) for v in value]
        return "; ".join([p for p in parts if p]).strip()
    return str(value).strip()


def _locomo_rows(path: Path, context_mode: str) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []

    for sample in payload:
        sample_id = str(sample.get("sample_id", "unknown"))
        conv = sample.get("conversation", {})

        dia_to_text: dict[int, str] = {}
        full_turns: list[str] = []
        session_keys = sorted(
            [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda x: int(x.split("_")[1]),
        )

        for session_key in session_keys:
            turns = conv.get(session_key, [])
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                speaker = _normalize_text(turn.get("speaker")) or "unknown"
                text = _normalize_text(turn.get("text"))
                if not text:
                    continue
                dia_id = turn.get("dia_id")
                if isinstance(dia_id, int):
                    dia_to_text[dia_id] = text
                full_turns.append(f"{speaker}: {text}")

        full_context = "\n".join(full_turns).strip()
        qa_rows = sample.get("qa", [])
        for idx, qa in enumerate(qa_rows):
            if not isinstance(qa, dict):
                continue
            question = _normalize_text(qa.get("question"))
            answer = _normalize_text(qa.get("answer"))
            evidence_ids = qa.get("evidence", [])
            evidence_lines: list[str] = []
            if isinstance(evidence_ids, list):
                for eid in evidence_ids:
                    if isinstance(eid, int) and eid in dia_to_text:
                        evidence_lines.append(dia_to_text[eid])

            if context_mode == "evidence" and evidence_lines:
                context = "\n".join(dict.fromkeys(evidence_lines)).strip()
            else:
                context = full_context

            if not question or not answer or not context:
                continue

            rows.append(
                {
                    "id": f"locomo-{sample_id}-q{idx:04d}",
                    "dataset": "locomo",
                    "context": context,
                    "question": question,
                    "ground_truth": answer,
                }
            )
    return rows


def _render_longmemeval_sessions(sessions: Any) -> str:
    if not isinstance(sessions, list):
        return ""
    chunks: list[str] = []
    for sidx, session in enumerate(sessions, start=1):
        if not isinstance(session, list):
            continue
        chunks.append(f"[session {sidx}]")
        for turn in session:
            if not isinstance(turn, dict):
                continue
            role = _normalize_text(turn.get("role")) or "unknown"
            content = _normalize_text(turn.get("content"))
            if not content:
                continue
            chunks.append(f"{role}: {content}")
    return "\n".join(chunks).strip()


def _longmemeval_rows(path: Path, max_context_chars: int) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        qid = _normalize_text(entry.get("question_id"))
        question = _normalize_text(entry.get("question"))
        answer = _normalize_text(entry.get("answer"))
        context = _render_longmemeval_sessions(entry.get("haystack_sessions"))
        if max_context_chars > 0 and len(context) > max_context_chars:
            context = context[:max_context_chars]
        if not qid or not question or not answer or not context:
            continue
        rows.append(
            {
                "id": f"longmemeval-{qid}",
                "dataset": "longmemeval",
                "context": context,
                "question": question,
                "ground_truth": answer,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    args = _parse_args()
    locomo_json = Path(args.locomo_json).resolve()
    longmemeval_json = Path(args.longmemeval_json).resolve()
    out_dir = Path(args.output_dir).resolve()

    if not locomo_json.exists():
        raise FileNotFoundError(f"LoCoMo file not found: {locomo_json}")
    if not longmemeval_json.exists():
        raise FileNotFoundError(f"LongMemEval file not found: {longmemeval_json}")

    locomo_rows = _locomo_rows(locomo_json, context_mode=args.locomo_context)
    longmemeval_rows = _longmemeval_rows(
        longmemeval_json,
        max_context_chars=max(0, int(args.longmemeval_max_context_chars)),
    )

    locomo_out = out_dir / "locomo_phase1.jsonl"
    longmemeval_out = out_dir / "longmemeval_phase1.jsonl"
    manifest_out = out_dir / "manifest.json"

    _write_jsonl(locomo_out, locomo_rows)
    _write_jsonl(longmemeval_out, longmemeval_rows)

    manifest = {
        "sources": {
            "locomo_json": str(locomo_json),
            "locomo_sha256": _sha256(locomo_json),
            "longmemeval_json": str(longmemeval_json),
            "longmemeval_sha256": _sha256(longmemeval_json),
        },
        "outputs": {
            "locomo_jsonl": str(locomo_out),
            "longmemeval_jsonl": str(longmemeval_out),
        },
        "counts": {
            "locomo_items": len(locomo_rows),
            "longmemeval_items": len(longmemeval_rows),
        },
        "options": {
            "locomo_context": args.locomo_context,
            "longmemeval_max_context_chars": int(args.longmemeval_max_context_chars),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {len(locomo_rows)} LoCoMo rows -> {locomo_out}")
    print(f"Wrote {len(longmemeval_rows)} LongMemEval rows -> {longmemeval_out}")
    print(f"Wrote manifest -> {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
