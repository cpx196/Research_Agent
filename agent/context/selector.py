"""Evidence deduplication and lightweight relevance selection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence


_TERM = re.compile(r"[a-zA-Z0-9_]+|[\u3400-\u9fff]")


def _terms(text: str) -> set[str]:
    return {token.lower() for token in _TERM.findall(text or "")}


@dataclass
class SelectionResult:
    evidence: list[dict[str, Any]]
    available_count: int
    deduplicated_count: int
    dropped_count: int
    top_k: int
    scores: list[float] = field(default_factory=list)


class EvidenceSelector:
    """Select relevant evidence without building another persisted index.

    Existing local-search records already carry BGE/FAISS relevance scores.
    This selector combines those scores with lexical overlap so evidence from
    web/paper tools (which has no vector score) is handled too.  An embedding
    model can be injected later through ``embedding_model`` for cosine-based
    ranking without changing the manager interface.
    """

    def __init__(self, dedup_similarity_threshold: float = 0.92, embedding_model: Any | None = None) -> None:
        self.dedup_similarity_threshold = dedup_similarity_threshold
        self.embedding_model = embedding_model

    @staticmethod
    def _key(item: Mapping[str, Any]) -> tuple[str, str]:
        return (str(item.get("source", "")), str(item.get("page", "")))

    @staticmethod
    def _content(item: Mapping[str, Any]) -> str:
        return str(item.get("content") or item.get("evidence_text") or "").strip()

    def _deduplicate(self, evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        by_location: dict[tuple[str, str], dict[str, Any]] = {}
        seen_contents: list[str] = []
        for raw in evidence:
            item = dict(raw)
            content = self._content(item)
            normalized = " ".join(content.lower().split())
            key = self._key(item)
            if normalized and any(
                SequenceMatcher(None, normalized, previous).ratio() >= self.dedup_similarity_threshold
                for previous in seen_contents
            ):
                continue
            if key in by_location and key != ("", ""):
                existing = by_location[key]
                if content and content not in str(existing.get("content", "")):
                    existing["content"] = (str(existing.get("content", "")).rstrip() + " " + content).strip()
                if float(item.get("score", 0.0) or 0.0) > float(existing.get("score", 0.0) or 0.0):
                    existing["score"] = item.get("score")
                continue
            if normalized:
                seen_contents.append(normalized)
            by_location[key] = item
        return list(by_location.values())

    def select_with_stats(self, evidence: Sequence[Mapping[str, Any]], query: str, top_k: int) -> SelectionResult:
        available_count = len(evidence)
        unique = self._deduplicate(evidence)
        query_terms = _terms(query)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, item in enumerate(unique):
            content_terms = _terms(self._content(item) + " " + str(item.get("source", "")))
            lexical = len(query_terms & content_terms) / max(len(query_terms), 1)
            raw_score = item.get("score")
            semantic = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
            # Local BGE score is useful, but lexical overlap makes the policy
            # transparent for evidence collected from other tools.
            combined = (0.65 * semantic + 0.35 * lexical) if raw_score is not None else lexical
            scored.append((combined, -index, item))
        scored.sort(reverse=True, key=lambda row: (row[0], row[1]))
        limit = max(1, int(top_k))
        selected = [item for _, _, item in scored[:limit]]
        return SelectionResult(
            evidence=selected,
            available_count=available_count,
            deduplicated_count=max(0, available_count - len(unique)),
            dropped_count=max(0, len(unique) - len(selected)),
            top_k=limit,
            scores=[round(score, 4) for score, _, _ in scored[:limit]],
        )

    def select(self, evidence: Sequence[Mapping[str, Any]], query: str, top_k: int) -> list[dict[str, Any]]:
        return self.select_with_stats(evidence, query, top_k).evidence
