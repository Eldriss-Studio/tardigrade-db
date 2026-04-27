import hashlib
import struct
from dataclasses import dataclass, field

from tardigrade_hooks.prefix_format import BulletListFormat

TIER_DRAFT = 0
TIER_VALIDATED = 1
TIER_CORE = 2


@dataclass
class PrefixResult:
    text: str
    version: int
    pack_ids: list[int] = field(default_factory=list)
    token_estimate: int = 0


class MemoryPrefixBuilder:
    def __init__(
        self,
        engine,
        owner,
        format=None,
        include_validated=True,
        token_budget=None,
        tokenizer=None,
    ):
        self._engine = engine
        self._owner = owner
        self._format = format or BulletListFormat()
        self._include_validated = include_validated
        self._token_budget = token_budget
        self._tokenizer = tokenizer

    def build(self) -> PrefixResult:
        packs = self._engine.list_packs(owner=self._owner)

        eligible = []
        for p in packs:
            tier = p["tier"]
            if tier == TIER_CORE:
                eligible.append(p)
            elif tier == TIER_VALIDATED and self._include_validated:
                eligible.append(p)

        eligible.sort(key=lambda p: p["importance"], reverse=True)

        selected = []
        for p in eligible:
            if p["text"] is None:
                continue
            selected.append(p)

        if self._token_budget is not None:
            selected = self._apply_budget(selected)

        text = self._format.format(selected)
        pack_ids = [p["pack_id"] for p in selected]
        version = self._compute_version(selected)
        token_estimate = self._count_tokens(text)

        return PrefixResult(
            text=text,
            version=version,
            pack_ids=pack_ids,
            token_estimate=token_estimate,
        )

    def has_changed(self, previous_version: int) -> bool:
        result = self.build()
        return result.version != previous_version

    def _apply_budget(self, memories):
        if not self._token_budget:
            return memories

        kept = []
        total = 0
        header_cost = self._count_tokens("Memory context:\n")
        total += header_cost

        for m in memories:
            line = f"- {m['text'].replace(chr(10), ' ')}"
            cost = self._count_tokens(line + "\n")
            if total + cost > self._token_budget:
                break
            kept.append(m)
            total += cost
        return kept

    def _count_tokens(self, text):
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        return max(1, len(text) // 4)

    def _compute_version(self, memories):
        h = hashlib.sha256()
        for m in memories:
            h.update(struct.pack("<Q", m["pack_id"]))
            h.update(m["text"].encode("utf-8"))
        return int.from_bytes(h.digest()[:8], "little")
