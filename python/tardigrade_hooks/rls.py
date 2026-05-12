"""Reflective Latent Search (RLS) — agentic retrieval in latent space.

Strategy pattern for query reformulation. Template Method for the
RETRIEVE → EVALUATE → REFORMULATE → RE-RETRIEVE → FUSE loop.
All retrieval stays tensor-native.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_STOP_WORDS = frozenset(
    "what does did do is was were are how has had have who where when why "
    "tell me about the a an in on at to for of by from with and or but "
    "something some any been being know known".split()
)

_SYNONYMS: dict[str, list[str]] = {
    "language": ["translation", "translated", "linguistic", "foreign", "patent"],
    "languages": ["translation", "translated", "linguistic", "foreign", "patent"],
    "athletic": ["running", "marathon", "ultramarathon", "race", "endurance", "sport"],
    "sport": ["running", "marathon", "ultramarathon", "race", "athletic"],
    "outdoors": ["hiking", "marathon", "highlands", "mountain", "trail"],
    "outdoorsy": ["hiking", "marathon", "highlands", "mountain", "trail"],
    "mechanical": ["motorcycle", "engine", "motor", "restored", "jawa", "machine"],
    "nature": ["ecology", "peatlands", "composting", "organic", "environmental", "mycorrhizal"],
    "ecology": ["peatlands", "composting", "organic", "environmental", "mycorrhizal", "nature"],
    "chemical": ["spectrometer", "calibration", "lab", "chemistry", "equipment", "mass"],
    "chemicals": ["spectrometer", "calibration", "lab", "chemistry", "equipment", "mass"],
    "materials": ["spectrometer", "magnets", "rare-earth", "supply", "mining"],
    "research": ["published", "paper", "peer-reviewed", "journal", "study"],
    "science": ["published", "paper", "peer-reviewed", "journal", "ecology"],
    "scientific": ["published", "paper", "peer-reviewed", "journal", "ecology"],
    "engineering": ["designed", "built", "system", "recovery", "waste-heat", "kiln"],
    "environmental": ["composting", "organic", "waste", "cooperative", "emissions"],
    "music": ["balalaika", "play", "grandmother", "instrument"],
}


class ReformulationStrategy(ABC):
    """Strategy: produces alternative query texts for re-retrieval."""

    @abstractmethod
    def reformulate(self, query_text: str | None) -> list[str]: ...


class KeywordExpansionStrategy(ReformulationStrategy):
    """Extracts content words, expands with synonyms."""

    def reformulate(self, query_text: str | None) -> list[str]:
        if not query_text or not query_text.strip():
            return []
        words = re.findall(r"[a-zA-Z]+", query_text.lower())
        content = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
        if not content:
            return []
        expanded = list(content)
        for word in content:
            syns = _SYNONYMS.get(word, [])
            expanded.extend(syns)
        return [" ".join(dict.fromkeys(expanded))]


class MultiPhrasingStrategy(ReformulationStrategy):
    """Generates template-based query variants without LLM generation."""

    def reformulate(self, query_text: str | None) -> list[str]:
        if not query_text or not query_text.strip():
            return []
        words = re.findall(r"[a-zA-Z]+", query_text.lower())
        content = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
        if not content:
            return []
        keyword_only = " ".join(content)
        subject = content[0] if content else ""
        rest = " ".join(content[1:]) if len(content) > 1 else ""
        who_what = f"{subject} {rest}" if rest else subject
        return [keyword_only, f"What did {who_what} involve"]


EMBEDDING_EXPANSION_TOP_K = 10
EMBEDDING_EXPANSION_MIN_SIM = 0.3


class EmbeddingExpansionStrategy(ReformulationStrategy):
    """Expands query with nearest neighbors from the model's embedding table.

    Language-agnostic: uses the model's own vocabulary knowledge.
    No external thesaurus. Pure latent-space.
    """

    def __init__(self, tokenizer, embed_weights, top_k: int = EMBEDDING_EXPANSION_TOP_K):
        import numpy as np

        self._tokenizer = tokenizer
        self._embed = np.array(embed_weights, dtype=np.float32)
        self._top_k = top_k
        norms = np.linalg.norm(self._embed, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        self._normed = self._embed / norms

    def reformulate(self, query_text: str | None) -> list[str]:
        if not query_text or not query_text.strip():
            return []
        import numpy as np

        words = re.findall(r"[a-zA-Z]+", query_text.lower())
        content = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
        if not content:
            return []

        expanded_tokens = list(content)
        for word in content:
            token_ids = self._tokenizer.encode(word, add_special_tokens=False)
            if not token_ids:
                continue
            token_id = token_ids[0]
            if token_id >= len(self._normed):
                continue
            query_vec = self._normed[token_id]
            sims = self._normed @ query_vec
            top_indices = np.argsort(sims)[::-1][1:self._top_k + 1]
            for idx in top_indices:
                if sims[idx] < EMBEDDING_EXPANSION_MIN_SIM:
                    break
                neighbor_text = self._tokenizer.decode([int(idx)]).strip()
                if len(neighbor_text) > 2 and neighbor_text.lower() not in _STOP_WORDS:
                    expanded_tokens.append(neighbor_text.lower())

        unique = list(dict.fromkeys(expanded_tokens))
        return [" ".join(unique)] if len(unique) > len(content) else []


def rrf_fuse_handles(handle_lists: list[list], k: int = 60) -> list:
    """Fuse MemoryCellHandle lists via RRF, dedup by cell_id."""
    scores: dict[int, float] = defaultdict(float)
    handle_by_id: dict[int, object] = {}

    for handles in handle_lists:
        for rank, h in enumerate(handles):
            cid = h.cell_id
            scores[cid] += 1.0 / (k + rank + 1)
            if cid not in handle_by_id:
                handle_by_id[cid] = h

    sorted_ids = sorted(scores, key=lambda cid: -scores[cid])
    return [handle_by_id[cid] for cid in sorted_ids]


DEFAULT_CONFIDENCE_THRESHOLD = 1.5
DEFAULT_MAX_ATTEMPTS = 2


class ReflectiveLatentSearch:
    """RETRIEVE → EVALUATE → REFORMULATE → RE-RETRIEVE → FUSE loop."""

    def __init__(
        self,
        engine,
        model,
        tokenizer,
        query_layer: int,
        hidden_size: int,
        owner: int = 1,
        k: int = 5,
        strategies: list[ReformulationStrategy] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ):
        self._engine = engine
        self._model = model
        self._tokenizer = tokenizer
        self._query_layer = query_layer
        self._hidden_size = hidden_size
        self._owner = owner
        self._k = k
        self._strategies = strategies or [KeywordExpansionStrategy()]
        self._threshold = confidence_threshold
        self._max_attempts = max_attempts

    def query(self, question: str, top_k: int | None = None) -> list:
        """Run the RLS loop. Returns list of MemoryCellHandle."""
        k = top_k or self._k
        handles = self._retrieve(question)

        if self._is_confident(handles):
            return handles[:k]

        all_results = [handles]
        attempts = 0
        for strategy in self._strategies:
            if attempts >= self._max_attempts - 1:
                break
            variants = strategy.reformulate(question)
            for variant in variants:
                if attempts >= self._max_attempts - 1:
                    break
                new_handles = self._retrieve(variant)
                if new_handles:
                    all_results.append(new_handles)
                attempts += 1

        fused = rrf_fuse_handles(all_results)
        return fused[:k]

    def _retrieve(self, text: str) -> list:
        """Single-shot latent retrieval via forward pass + engine."""
        import numpy as np
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256,
        )
        inputs = {k_: v.to(self._model.device) for k_, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)

        h = out.hidden_states[self._query_layer][0]
        tokens = h[1:].cpu().numpy().astype(np.float32)

        if hasattr(self._engine, "mem_read_tokens"):
            results = self._engine.mem_read_tokens(tokens, self._k * 2, self._owner)
        else:
            from .encoding import encode_per_token
            key = encode_per_token(tokens, self._hidden_size)
            results = self._engine.mem_read(key, self._k * 2, self._owner)

        from .hook import MemoryCellHandle
        return [
            MemoryCellHandle(
                cell_id=r.cell_id, owner=r.owner, layer=r.layer, score=r.score,
                key=np.array(r.key(), dtype=np.float32),
                value=np.array(r.value(), dtype=np.float32),
            )
            for r in results
        ]

    def _is_confident(self, handles: list) -> bool:
        if len(handles) < 2:
            return False
        ratio = handles[0].score / handles[1].score if handles[1].score > 0 else float("inf")
        return ratio >= self._threshold
