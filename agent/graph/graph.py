"""LangGraph assembly and public runner for Research Agent V3."""

from __future__ import annotations

import uuid
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.agent import TraceLogger
from agent.context import ContextConfig, ContextManager
from tools import TOOL_REGISTRY, TOOL_SCHEMAS

from .context import VerifierOverride, WorkflowContext
from .nodes import (
    make_direct_node,
    make_claim_aligner_node,
    make_claim_extractor_node,
    make_claim_planner_node,
    make_evidence_node,
    make_evidence_quality_node,
    make_error_detector_node,
    make_planner_node,
    make_react_node,
    make_relevance_node,
    make_repair_node,
    make_researcher_node,
    make_router_node,
    make_verifier_node,
    make_verification_judge_node,
    make_verification_question_node,
    make_verification_researcher_node,
    make_writer_node,
)
from .routing import (
    route_after_react,
    route_after_relevance,
    route_after_researcher,
    route_after_router,
    route_after_verifier,
)
from .state import ResearchState
from .tools import make_guarded_tool_node


def _planned_allowed_tools(state: Mapping[str, Any]) -> list[str]:
    plan = list(state.get("plan", []))
    index = int(state.get("current_step", 0))
    if not plan or index >= len(plan):
        return []
    planned = str(plan[index].get("tool", ""))
    route_allowed = {
        str(name)
        for name in state.get("route_decision", {}).get("allowed_tools", [])
    }
    return [planned] if planned and planned in route_allowed else []


def build_research_graph(
    llm: Any,
    *,
    tools: Mapping[str, Any] | None = None,
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    max_iterations: int = 2,
    verifier_mode: str = "simple",
    max_verification_tool_calls: int = 2,
    checkpointer: Any | None = None,
    context_manager: ContextManager | None = None,
    verifier_override: VerifierOverride | None = None,
    forced_verifier_failures: int = 0,
    stream_callback: Any | None = None,
):
    """Compile the V3 StateGraph while retaining the V0/V2 tool functions."""

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if verifier_mode not in {"none", "simple", "active"}:
        raise ValueError("verifier_mode must be 'none', 'simple', or 'active'")
    if not 1 <= max_verification_tool_calls <= 4:
        raise ValueError("max_verification_tool_calls must be between 1 and 4")
    active_tools = dict(tools if tools is not None else TOOL_REGISTRY)
    active_schemas = list(tool_schemas if tool_schemas is not None else TOOL_SCHEMAS)
    context = WorkflowContext(
        llm=llm,
        tools=active_tools,
        tool_schemas=active_schemas,
        max_iterations=max_iterations,
        verifier_mode=verifier_mode,
        max_verification_tool_calls=max_verification_tool_calls,
        context_manager=context_manager,
        verifier_override=verifier_override,
        forced_verifier_failures=max(0, forced_verifier_failures),
        stream_callback=stream_callback,
    )

    builder = StateGraph(ResearchState)
    builder.add_node("router", make_router_node(context))
    builder.add_node("direct", make_direct_node(context))
    builder.add_node("react", make_react_node(context))
    builder.add_node(
        "react_tools",
        make_guarded_tool_node(
            active_tools,
            active_schemas,
            lambda state: state.get("route_decision", {}).get("allowed_tools", []),
        ),
    )
    builder.add_node("react_evidence", make_evidence_node(context))
    builder.add_node("react_relevance", make_relevance_node(context, react_path=True))
    builder.add_node("planner", make_planner_node(context))
    builder.add_node("researcher", make_researcher_node(context))
    builder.add_node(
        "tools",
        make_guarded_tool_node(
            active_tools,
            active_schemas,
            _planned_allowed_tools,
        ),
    )
    builder.add_node("evidence", make_evidence_node(context))
    builder.add_node("relevance", make_relevance_node(context))
    builder.add_node("writer", make_writer_node(context))
    builder.add_node("evidence_quality", make_evidence_quality_node(context))
    builder.add_node("claim_planner", make_claim_planner_node(context))
    builder.add_node("repair", make_repair_node(context))
    if verifier_mode == "active":
        builder.add_node("claim_extractor", make_claim_extractor_node(context))
        builder.add_node("claim_evidence_aligner", make_claim_aligner_node(context))
        builder.add_node("verification_error_detector", make_error_detector_node(context))
        builder.add_node("verification_question_generator", make_verification_question_node(context))
        builder.add_node("verification_researcher", make_verification_researcher_node(context))
        builder.add_node("verification_judge", make_verification_judge_node(context))
    elif verifier_mode == "simple":
        builder.add_node("verifier", make_verifier_node(context))

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"direct": "direct", "react": "react", "research": "planner"},
    )
    builder.add_edge("direct", END)
    builder.add_conditional_edges(
        "react",
        route_after_react,
        {"tools": "react_tools", "end": END},
    )
    builder.add_edge("react_tools", "react_evidence")
    builder.add_edge("react_evidence", "react_relevance")
    builder.add_edge("react_relevance", "react")
    builder.add_edge("planner", "researcher")
    builder.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {"tools": "tools", "writer": "evidence_quality"},
    )
    builder.add_edge("tools", "evidence")
    builder.add_edge("evidence", "relevance")
    builder.add_conditional_edges(
        "relevance",
        route_after_relevance,
        {"researcher": "researcher", "writer": "evidence_quality"},
    )
    builder.add_edge("evidence_quality", "claim_planner")
    builder.add_edge("claim_planner", "writer")
    if verifier_mode == "active":
        builder.add_edge("writer", "claim_extractor")
        builder.add_edge("claim_extractor", "claim_evidence_aligner")
        builder.add_edge("claim_evidence_aligner", "verification_error_detector")
        builder.add_edge("verification_error_detector", "verification_question_generator")
        builder.add_edge("verification_question_generator", "verification_researcher")
        builder.add_edge("verification_researcher", "verification_judge")
        builder.add_conditional_edges(
            "verification_judge",
            route_after_verifier,
            {"end": END, "repair": "repair"},
        )
        builder.add_edge("repair", "claim_extractor")
    elif verifier_mode == "simple":
        builder.add_edge("writer", "verifier")
        builder.add_conditional_edges(
            "verifier",
            route_after_verifier,
            {"end": END, "repair": "repair"},
        )
        builder.add_edge("repair", "verifier")
    else:
        builder.add_edge("writer", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


class LangGraphResearchAgent:
    """User-facing V3 runner with in-memory checkpointing and graph traces."""

    def __init__(
        self,
        llm: Any,
        tools: Mapping[str, Any] | None = None,
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
        max_iterations: int = 2,
        verifier_mode: str = "simple",
        max_verification_tool_calls: int = 2,
        verbose: bool = True,
        trace_file: str | Path | None = None,
        checkpointer: Any | None = None,
        context_enabled: bool = False,
        context_config: ContextConfig | None = None,
        context_manager: ContextManager | None = None,
        verifier_override: VerifierOverride | None = None,
        forced_verifier_failures: int = 0,
        token_callback: Any | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.verifier_mode = verifier_mode
        self.max_verification_tool_calls = max_verification_tool_calls
        self.stream_callback = token_callback
        self.live_trace_callback: Any | None = None
        self.live_claim_callback: Any | None = None
        self.context_manager = context_manager
        if self.context_manager is None and context_enabled:
            self.context_manager = ContextManager(config=context_config, extractor_llm=llm)
        if self.context_manager is not None and not self.context_manager.config.enabled:
            self.context_manager = None
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_research_graph(
            llm,
            tools=tools,
            tool_schemas=tool_schemas,
            max_iterations=max_iterations,
            verifier_mode=verifier_mode,
            max_verification_tool_calls=max_verification_tool_calls,
            checkpointer=self.checkpointer,
            context_manager=self.context_manager,
            verifier_override=verifier_override,
            forced_verifier_failures=forced_verifier_failures,
            stream_callback=lambda token: self.stream_callback(token) if self.stream_callback else None,
        )
        self.trace = TraceLogger(verbose=verbose, file_path=trace_file)
        self.last_state: ResearchState = {}
        self.last_run_stats: dict[str, Any] = {}

    def run(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        conversation_history: Sequence[Mapping[str, str]] | None = None,
        session_preferences: Mapping[str, str] | None = None,
    ) -> str:
        query = query.strip()
        if not query:
            return "AgentError: Query cannot be empty."
        thread_id = thread_id or f"graph-{uuid.uuid4()}"
        started_at = time.perf_counter()
        history = [
            {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
            for item in (conversation_history or [])[-12:]
            if str(item.get("content", "")).strip()
        ]
        history_messages = [
            AIMessage(content=item["content"])
            if item["role"] == "assistant"
            else HumanMessage(content=item["content"])
            for item in history
        ]
        initial: ResearchState = {
            "messages": history_messages + [HumanMessage(content=query)],
            "query": query,
            "original_query": query,
            "standalone_query": query,
            "resolved_topic": "",
            "resolved_intent": "general",
            "search_queries": {},
            "conversation_history": history,
            "session_preferences": dict(session_preferences or {}),
            "route_decision": {},
            "react_steps": 0,
            "react_pending": False,
            "plan": [],
            "current_step": 0,
            "evidence": [],
            "accepted_evidence": [],
            "qualified_evidence": [],
            "evidence_quality_checks": [],
            "relevance_checks": [],
            "relevance_passed": False,
            "relevance_feedback": "",
            "retrieval_retry_count": 0,
            "max_retrieval_retries": 1,
            "retry_query": "",
            "retry_tool": "",
            "open_questions": [],
            "draft": "",
            "original_draft": "",
            "planned_claims": [],
            "claims": [],
            "claim_evidence": [],
            "claim_audits": [],
            "verification_passed": self.verifier_mode == "none",
            "verification_feedback": "",
            "verification_details": {},
            "potential_errors": [],
            "verification_questions": [],
            "verification_evidence": [],
            "verifier_feedback": [],
            "verification_iteration": 0,
            "max_verification_iterations": self.max_iterations,
            "verification_tool_calls": 0,
            "max_verification_tool_calls": self.max_verification_tool_calls,
            "verifier_mode": self.verifier_mode,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "context_stats": {},
            "graph_trace": ["[Graph] START"],
        }
        run_config = {"configurable": {"thread_id": thread_id}}
        if self.live_trace_callback is not None:
            self.live_trace_callback(f"[Query] {query}")
            self.live_trace_callback("[Graph] START")
            live_claim_state: dict[str, list[Any]] = {
                "planned_claims": [],
                "evidence_quality_checks": [],
                "claims": [],
                "relations": [],
                "repairs": [],
            }
            for update in self.graph.stream(initial, config=run_config, stream_mode="updates"):
                if not isinstance(update, Mapping):
                    continue
                for node_update in update.values():
                    if not isinstance(node_update, Mapping):
                        continue
                    for line in node_update.get("graph_trace", []) or []:
                        self.live_trace_callback(str(line))
                    if self.live_claim_callback is not None and any(
                        key in node_update for key in (
                            "planned_claims", "evidence_quality_checks", "claims",
                            "claim_evidence", "claim_audits",
                        )
                    ):
                        key_map = {
                            "planned_claims": "planned_claims",
                            "evidence_quality_checks": "evidence_quality_checks",
                            "claims": "claims",
                            "claim_evidence": "relations",
                            "claim_audits": "repairs",
                        }
                        for state_key, payload_key in key_map.items():
                            if state_key in node_update:
                                live_claim_state[payload_key] = list(node_update.get(state_key, []) or [])
                        self.live_claim_callback({
                            **live_claim_state,
                            "stage": str(next(iter(update.keys()), "claim_update")),
                        })
            result = dict(self.graph.get_state(run_config).values)
            self.live_trace_callback("[Graph] END")
            self.live_trace_callback("[Final Answer]")
        else:
            result = self.graph.invoke(initial, config=run_config)
        self.last_state = result
        answer = str(result.get("draft", "")).strip() or "AgentError: Graph returned an empty draft."
        trace_lines = list(result.get("graph_trace", []))
        self.trace.separator()
        self.trace.emit(f"[Query] {query}")
        for line in trace_lines:
            self.trace.emit(line)
        self.trace.emit("[Graph] END")
        self.trace.emit("[Final Answer]")
        self.trace.emit(answer)
        self.last_run_stats = {
            "query": query,
            "iteration": result.get("iteration", 0),
            "plan_steps": len(result.get("plan", [])),
            "evidence": len(result.get("evidence", [])),
            "tool_calls": sum(
                1
                for message in result.get("messages", [])
                if getattr(message, "tool_calls", None)
            ),
            "llm_calls": sum(
                1
                for line in trace_lines
                if line in {
                    "[Node] DirectAnswer", "[Node] ReAct", "[Node] Researcher",
                    "[Node] Writer", "[Node] AnswerRepair", "[Node] VerificationErrorDetector",
                    "[Node] VerificationQuestionGenerator", "[Node] VerificationJudge",
                }
            ),
            "graph_steps": sum(1 for line in trace_lines if line.startswith("[Node] ")),
            "latency_seconds": round(time.perf_counter() - started_at, 3),
            "verification_passed": result.get("verification_passed", False),
            "context_enabled": self.context_manager is not None,
            "context_stats": result.get("context_stats", {}),
            "thread_id": thread_id,
            "conversation_turns": len(history) // 2,
            "session_preferences": dict(session_preferences or {}),
            "route_decision": result.get("route_decision", {}),
            "original_query": result.get("original_query", query),
            "standalone_query": result.get("standalone_query", query),
            "resolved_topic": result.get("resolved_topic", ""),
            "resolved_intent": result.get("resolved_intent", "general"),
            "accepted_evidence": len(result.get("accepted_evidence", [])),
            "relevance_checks": result.get("relevance_checks", []),
            "retrieval_retries": result.get("retrieval_retry_count", 0),
            "verification_details": result.get("verification_details", {}),
            "claims": len(result.get("claims", [])),
            "claim_status_counts": {
                status: sum(1 for claim in result.get("claims", []) if claim.get("status") == status)
                for status in ("SUPPORTED", "PARTIALLY_SUPPORTED", "INFERRED", "CONTRADICTED", "UNSUPPORTED")
            },
            "claim_repairs": len(result.get("claim_audits", [])),
            "verifier_mode": result.get("verifier_mode", self.verifier_mode),
            "verification_iterations": result.get("verification_iteration", 0),
            "verification_tool_calls": result.get("verification_tool_calls", 0),
            "router_source": result.get("route_decision", {}).get("source", "unknown"),
            "router_llm_calls": int(result.get("route_decision", {}).get("source") == "llm"),
        }
        return answer
