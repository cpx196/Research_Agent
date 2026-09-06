"""Context observability helpers used by V4 traces and evaluations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


class ContextMetrics:
    """Build stable, JSON-friendly metrics for a node context."""

    @staticmethod
    def node_stats(
        *,
        node: str,
        raw_tokens: int,
        final_tokens: int,
        budget: int,
        selected_evidence: int = 0,
        available_evidence: int = 0,
        deduplicated_evidence: int = 0,
        dropped_evidence: int = 0,
        compressed_tool_results: int = 0,
        dropped_sections: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "node": node,
            "raw_context_tokens": raw_tokens,
            "final_context_tokens": final_tokens,
            "context_budget": budget,
            "selected_evidence": selected_evidence,
            "available_evidence": available_evidence,
            "deduplicated_evidence": deduplicated_evidence,
            "dropped_evidence": dropped_evidence,
            "compressed_tool_results": compressed_tool_results,
            "compression_ratio": max(0.0, round(1.0 - final_tokens / raw_tokens, 4)) if raw_tokens else 0.0,
            "dropped_sections": dropped_sections or [],
        }


def merge_context_stats(state: Mapping[str, Any], node: str, stats: Mapping[str, Any]) -> dict[str, Any]:
    current = deepcopy(dict(state.get("context_stats", {})))
    nodes = dict(current.get("nodes", {}))
    nodes[node] = dict(stats)
    current["nodes"] = nodes
    history = list(current.get("history", []))
    entry = dict(stats)
    entry["sequence"] = len(history) + 1
    history.append(entry)
    current["history"] = history
    current["last_node"] = node
    current["max_context_tokens"] = max(
        [int(item.get("final_context_tokens", 0)) for item in history] or [0]
    )
    current["total_context_tokens"] = sum(
        int(item.get("final_context_tokens", 0)) for item in history
    )
    return current


def add_context_event(state: Mapping[str, Any], category: str, event: Mapping[str, Any]) -> dict[str, Any]:
    current = deepcopy(dict(state.get("context_stats", {})))
    values = list(current.get(category, []))
    values.append(dict(event))
    current[category] = values
    return current
