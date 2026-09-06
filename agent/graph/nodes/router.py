"""Router graph node."""

from __future__ import annotations

from typing import Any

from agent.graph.router import HybridRouter

from ..context import WorkflowContext
from ..state import ResearchState
from .common import trace


def make_router_node(context: WorkflowContext):
    router = HybridRouter(
        context.llm,
        [
            str(schema.get("function", {}).get("name", ""))
            for schema in context.tool_schemas
            if schema.get("function", {}).get("name")
        ],
    )

    def router_node(state: ResearchState) -> dict[str, Any]:
        decision = router.classify(
            str(state.get("query", "")),
            conversation_history=state.get("conversation_history", []),
            session_preferences=state.get("session_preferences", {}),
        )
        lines = trace(
            "[Node] Router",
            f"Original Query: {decision.original_query}",
            f"Standalone Query: {decision.standalone_query}",
            f"Resolved Topic: {decision.topic or '(none)'}",
            f"Intent: {decision.intent}",
            f"Route: {decision.route}",
            f"Complexity: {decision.complexity}",
            (
                "Flags: "
                f"planning={decision.need_planning}, "
                f"web={decision.need_web}, "
                f"local_rag={decision.need_local_rag}, "
                f"verification={decision.need_verification}"
            ),
            f"Suggested Workers: {decision.suggested_workers}",
            f"Allowed Tools: {', '.join(decision.allowed_tools) or '(none)'}",
            f"Primary Tool: {decision.primary_tool or '(none)'}",
            f"Fallback Tools: {', '.join(decision.fallback_tools) or '(none)'}",
            f"Confidence: {decision.confidence:.2f}",
            f"Source: {decision.source}",
            f"[Conditional Edge] Router -> {decision.route}",
        )
        return {
            "route_decision": decision.as_dict(),
            "original_query": decision.original_query,
            "standalone_query": decision.standalone_query,
            "resolved_topic": decision.topic,
            "resolved_intent": decision.intent,
            "search_queries": dict(decision.search_queries),
            "graph_trace": lines,
        }

    return router_node
