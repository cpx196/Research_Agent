"""Token estimation and priority-aware context budgets for V4."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any


def estimate_tokens(value: Any) -> int:
    """Estimate input tokens without loading a model tokenizer.

    English-like text is approximated as four characters per token, while
    CJK characters are approximated as 1.8 characters per token.  This is a
    budget signal rather than billing-accurate token accounting.
    """

    text = str(value or "")
    if not text:
        return 0
    cjk = sum(1 for char in text if "\u3400" <= char <= "\u9fff")
    non_cjk = len(text) - cjk
    return max(1, math.ceil(cjk / 1.8 + non_cjk / 4.0))


@dataclass(frozen=True)
class ContextConfig:
    """All V4 context controls in one configurable object."""

    enabled: bool = True
    researcher_top_k: int = 6
    writer_top_k: int = 10
    verifier_top_k: int = 10
    planner_budget: int = 3000
    researcher_budget: int = 6000
    writer_budget: int = 8000
    verifier_budget: int = 8000
    tool_result_compress_threshold: int = 2500
    dedup_similarity_threshold: float = 0.92
    max_tool_result_chars: int = 12000
    max_evidence_chars: int = 1800
    llm_extractor_enabled: bool = False

    @classmethod
    def from_env(cls) -> "ContextConfig":
        """Read optional V4 overrides while keeping safe defaults."""

        def integer(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        def decimal(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError:
                return default

        return cls(
            enabled=os.getenv("CONTEXT_ENABLED", "true").strip().lower() not in {"0", "false", "no"},
            researcher_top_k=integer("CONTEXT_RESEARCHER_TOP_K", cls.researcher_top_k),
            writer_top_k=integer("CONTEXT_WRITER_TOP_K", cls.writer_top_k),
            verifier_top_k=integer("CONTEXT_VERIFIER_TOP_K", cls.verifier_top_k),
            planner_budget=integer("CONTEXT_PLANNER_BUDGET", cls.planner_budget),
            researcher_budget=integer("CONTEXT_RESEARCHER_BUDGET", cls.researcher_budget),
            writer_budget=integer("CONTEXT_WRITER_BUDGET", cls.writer_budget),
            verifier_budget=integer("CONTEXT_VERIFIER_BUDGET", cls.verifier_budget),
            tool_result_compress_threshold=integer(
                "CONTEXT_TOOL_RESULT_THRESHOLD", cls.tool_result_compress_threshold
            ),
            dedup_similarity_threshold=decimal(
                "CONTEXT_DEDUP_SIMILARITY_THRESHOLD", cls.dedup_similarity_threshold
            ),
            max_tool_result_chars=integer("CONTEXT_MAX_TOOL_RESULT_CHARS", cls.max_tool_result_chars),
            max_evidence_chars=integer("CONTEXT_MAX_EVIDENCE_CHARS", cls.max_evidence_chars),
            llm_extractor_enabled=os.getenv("CONTEXT_LLM_EXTRACTOR", "false").strip().lower()
            not in {"0", "false", "no"},
        )


@dataclass(frozen=True)
class ContextSection:
    """A context block with V4 priority: P0 is highest priority."""

    name: str
    text: str
    priority: int = 1


@dataclass
class BudgetResult:
    text: str
    raw_tokens: int
    final_tokens: int
    budget: int
    dropped_sections: list[str] = field(default_factory=list)
    truncated_sections: list[str] = field(default_factory=list)
    sections_kept: list[str] = field(default_factory=list)


def fit_to_budget(sections: list[ContextSection], budget: int) -> BudgetResult:
    """Keep high-priority sections and truncate only the lowest priorities."""

    budget = max(1, int(budget))
    raw_rendered = "\n\n".join(f"## {section.name}\n{section.text}" for section in sections)
    raw_tokens = estimate_tokens(raw_rendered)
    ordered = sorted(enumerate(sections), key=lambda item: (item[1].priority, item[0]))
    kept: list[tuple[int, ContextSection, str]] = []
    dropped: list[str] = []
    truncated: list[str] = []
    remaining = budget

    for _, section in ordered:
        section_tokens = estimate_tokens(section.text)
        if section_tokens <= remaining:
            kept.append((section.priority, section, section.text))
            remaining -= section_tokens
            continue

        # P0/P1 information is retained as far as possible.  A P0 block can
        # exceed the total budget only when it is intrinsically larger than
        # the budget; in that case it is kept and the metric records overflow.
        if section.priority <= 1 and remaining > 0:
            target_chars = max(32, int(len(section.text) * remaining / max(section_tokens, 1)))
            shortened = section.text[:target_chars].rstrip() + "\n[context truncated]"
            kept.append((section.priority, section, shortened))
            truncated.append(section.name)
            remaining = 0
        else:
            dropped.append(section.name)

    kept.sort(key=lambda item: (item[0], sections.index(item[1])))

    def render(items: list[tuple[int, ContextSection, str]]) -> str:
        return "\n\n".join(f"## {section.name}\n{text}" for _, section, text in items)

    # The first pass budgets section bodies.  Headers and separators also
    # consume tokens, so enforce the final rendered budget once more.  P0
    # sections are never removed; lower-priority sections are shortened or
    # dropped until the final context fits whenever that is possible.
    rendered = render(kept)
    while estimate_tokens(rendered) > budget:
        candidates = [index for index, (priority, _, _) in enumerate(kept) if priority > 0]
        if not candidates:
            break
        index = candidates[-1]
        priority, section, current_text = kept[index]
        low, high = 0, len(current_text)
        best: str | None = None
        while low <= high:
            middle = (low + high) // 2
            candidate = current_text[:middle].rstrip()
            if middle < len(current_text):
                candidate = (candidate + "\n[context truncated]").strip()
            trial = list(kept)
            trial[index] = (priority, section, candidate)
            if estimate_tokens(render(trial)) <= budget:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        if best is None:
            dropped.append(section.name)
            kept.pop(index)
        else:
            kept[index] = (priority, section, best)
            if section.name not in truncated:
                truncated.append(section.name)
        rendered = render(kept)
    return BudgetResult(
        text=rendered,
        raw_tokens=raw_tokens,
        final_tokens=estimate_tokens(rendered),
        budget=budget,
        dropped_sections=dropped,
        truncated_sections=truncated,
        sections_kept=[section.name for _, section, _ in kept],
    )
