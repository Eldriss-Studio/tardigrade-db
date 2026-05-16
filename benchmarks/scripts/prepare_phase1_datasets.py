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
    parser.add_argument(
        "--locomo-exclude-categories",
        default=",".join(str(c) for c in sorted(DEFAULT_LOCOMO_EXCLUDE_CATEGORIES)),
        help=(
            "Comma-separated LoCoMo categories to exclude. Default '5' "
            "drops adversarial/unanswerable items — matches every "
            "leaderboard system. Pass empty string '' to include all."
        ),
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


def _session_index_from_dia_id(dia_id: Any) -> int | None:
    """Extract the session number from a LoCoMo dia_id ("D1:3" → 1).

    Returns ``None`` for any malformed/legacy form so callers can fall
    back to date-less evidence rather than crash.
    """
    if not isinstance(dia_id, str) or not dia_id.startswith("D"):
        return None
    head = dia_id[1:].split(":", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


# Default LoCoMo categories to exclude. Cat 5 is adversarial /
# unanswerable; every published leaderboard system (Mem0, Memobase,
# ByteRover, Letta, MemMachine) filters it because including it
# conflates retrieval quality with refusal calibration (different
# abilities). Per dial481/locomo-audit, the official eval scripts
# silently skip 446 Cat-5 items — matching that here keeps our
# numerator/denominator aligned with leaderboard reports. Override
# via the `--locomo-exclude-categories` CLI flag.
DEFAULT_LOCOMO_EXCLUDE_CATEGORIES: frozenset[int] = frozenset({5})


def _locomo_rows(
    path: Path,
    context_mode: str,
    exclude_categories: set[int] | frozenset[int] | None = None,
) -> list[dict[str, str]]:
    if exclude_categories is None:
        exclude_categories = DEFAULT_LOCOMO_EXCLUDE_CATEGORIES
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []

    for sample in payload:
        sample_id = str(sample.get("sample_id", "unknown"))
        conv = sample.get("conversation", {})

        dia_to_text: dict[Any, str] = {}
        dia_to_session: dict[Any, int] = {}
        full_turns: list[str] = []
        session_keys = sorted(
            [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda x: int(x.split("_")[1]),
        )
        # Session dates power the temporal-question fix (audit 2026-05-15
        # Phase 1B.6): evidence lines like "I went yesterday" can't resolve
        # against ground-truth dates like "7 May 2023" without the session
        # date prefixed. Missing keys map to ``None`` so date-less
        # fallback still ingests.
        session_dates: dict[int, str] = {}
        for sk in session_keys:
            try:
                idx = int(sk.split("_")[1])
            except (IndexError, ValueError):
                continue
            date = _normalize_text(conv.get(f"{sk}_date_time"))
            if date:
                session_dates[idx] = date

        for session_key in session_keys:
            session_idx = int(session_key.split("_")[1])
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
                # LoCoMo uses string dia_ids like "D1:3" (session 1, turn 3).
                # Older revisions used plain ints. Accept both.
                if isinstance(dia_id, (int, str)) and dia_id != "":
                    dia_to_text[dia_id] = text
                    # Trust dia_id parsing first, fall back to session_idx
                    # (matters for legacy int dia_ids that don't encode session).
                    parsed_session = _session_index_from_dia_id(dia_id)
                    dia_to_session[dia_id] = parsed_session if parsed_session is not None else session_idx
                full_turns.append(f"{speaker}: {text}")

        full_context = "\n".join(full_turns).strip()
        qa_rows = sample.get("qa", [])
        for idx, qa in enumerate(qa_rows):
            if not isinstance(qa, dict):
                continue
            # Category filter (default drops Cat 5 adversarial).
            cat = qa.get("category")
            if isinstance(cat, int) and cat in exclude_categories:
                continue
            question = _normalize_text(qa.get("question"))
            answer = _normalize_text(qa.get("answer"))
            evidence_ids = qa.get("evidence", [])
            evidence_lines: list[str] = []
            if isinstance(evidence_ids, list):
                for eid in evidence_ids:
                    # Accept both legacy int dia_ids and current string ids ("D1:3").
                    if isinstance(eid, (int, str)) and eid in dia_to_text:
                        text = dia_to_text[eid]
                        sidx = dia_to_session.get(eid)
                        date = session_dates.get(sidx) if sidx is not None else None
                        evidence_lines.append(
                            f"[{date}] {text}" if date else text
                        )

            if context_mode == "evidence":
                # Oracle mode: an item without evidence has no defined
                # ground-truth context, so we skip it instead of falling
                # back to the full conversation. Full-fallback items
                # produced ~128 chunks each at ingest, dominating the
                # retrieval space (4 items × 128 chunks = 512 cells in
                # a 1542-item corpus). Skipping keeps the corpus honest
                # at the cost of ~0.3% of items.
                if not evidence_lines:
                    continue
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

    exclude_cats: set[int] = set()
    if args.locomo_exclude_categories.strip():
        for piece in args.locomo_exclude_categories.split(","):
            piece = piece.strip()
            if piece:
                exclude_cats.add(int(piece))
    locomo_rows = _locomo_rows(
        locomo_json,
        context_mode=args.locomo_context,
        exclude_categories=exclude_cats,
    )
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
            "locomo_exclude_categories": sorted(exclude_cats),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {len(locomo_rows)} LoCoMo rows -> {locomo_out}")
    print(f"Wrote {len(longmemeval_rows)} LongMemEval rows -> {longmemeval_out}")
    print(f"Wrote manifest -> {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
