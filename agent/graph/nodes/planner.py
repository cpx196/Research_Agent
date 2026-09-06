"""Deterministic Planner driven by the semantic Router output."""

from __future__ import annotations

import re
from typing import Any

from ..context import WorkflowContext
from ..state import PlanStep, ResearchState
from .common import trace


def _step(index: int, task: str, tool: str, *, retry: bool = False) -> PlanStep:
    return {
        "id": index,
        "task": task,
        "tool": tool,
        "status": "pending",
        "retry": retry,
    }


def build_research_plan(query: str, router_decision: dict[str, Any] | None = None) -> list[PlanStep]:
    """Build an auditable plan without another LLM call.

    When Router metadata is available, its standalone query and tool-specific
    queries are authoritative.  The older keyword behavior is retained for
    direct callers and compatibility tests.
    """

    decision = dict(router_decision or {})
    standalone = str(decision.get("standalone_query") or query).strip()
    normalized = standalone.lower()
    calculation = bool(re.search(r"计算(?!机)", normalized)) or any(
        token in normalized for token in ("算一下", "calculate")
    )
    if calculation:
        return [_step(1, standalone, "calculator")]

    if decision.get("route") == "research":
        allowed = [str(name) for name in decision.get("allowed_tools", []) if str(name)]
        primary = str(decision.get("primary_tool", ""))
        search_queries = dict(decision.get("search_queries", {}))
        intent = str(decision.get("intent", "general"))

        selected: list[str] = []
        local_paper_pair = "local_search" in allowed and "paper_search" in allowed
        # For local-library topics, local_search and paper_search are both
        # planned sources. Passing one source's relevance gate cannot suppress
        # execution of the other source.
        if local_paper_pair and intent in {"paper_search", "general", "research"}:
            selected.extend(("local_search", "paper_search"))
        elif primary in allowed:
            selected.append(primary)
        # Comparative/current research intentionally gathers local and web
        # evidence as independent planned sources.
        if intent != "paper_search" and decision.get("need_local_rag") and decision.get("need_web"):
            for tool in ("local_search", "web_search"):
                if tool in allowed and tool not in selected:
                    selected.append(tool)
        if not selected and allowed:
            selected.append(allowed[0])
        return [
            _step(index, str(search_queries.get(tool) or standalone), tool)
            for index, tool in enumerate(selected, start=1)
        ]

    # Legacy deterministic planning for callers without Router metadata.
    capability_question = (
        any(token in normalized for token in ("联网", "网络信息", "网络搜索"))
        and any(token in normalized for token in ("能", "不能", "可以", "无法", "为什么", "能力"))
    ) or any(token in normalized for token in ("can you search", "web access", "internet access"))
    if capability_question:
        return []

    wants_web = any(
        token in normalized
        for token in (
            "最新", "开源", "官方", "代码", "联网搜索", "网络搜索", "搜索", "查一下",
            "查询", "实时", "当前", "新闻", "是谁", "谁是", "who is", "latest",
            "official", "current", "news",
        )
    )
    local_topic = any(
        token in normalized
        for token in ("jepa", "dino", "机器人", "robot", "本地", "ust", "world model", "world modeling")
    )
    wants_paper = any(token in normalized for token in ("论文", "文献", "paper", "papers"))

    plan: list[PlanStep] = []
    if local_topic:
        plan.append(_step(len(plan) + 1, standalone, "local_search"))
    elif wants_paper:
        plan.append(_step(len(plan) + 1, standalone, "paper_search"))
    if wants_web:
        plan.append(_step(len(plan) + 1, standalone, "web_search"))
    return plan


def make_planner_node(context: WorkflowContext):
    def planner_node(state: ResearchState) -> dict[str, Any]:
        query = str(state.get("standalone_query") or state.get("query", "")).strip()
        decision = dict(state.get("route_decision", {}))
        plan = build_research_plan(query, decision)
        open_questions = [str(step.get("task", "")) for step in plan]
        lines = trace(
            "[Node] Planner",
            f"Planning Query: {query}",
            "Plan:",
        )
        if plan:
            lines.extend(f"{step['id']}. {step['task']} [{step['tool']}]" for step in plan)
        else:
            lines.append("(No external research required; Writer may answer directly.)")
        updates: dict[str, Any] = {
            "plan": plan,
            "current_step": 0,
            "iteration": 0,
            "open_questions": open_questions,
            "graph_trace": lines,
        }
        if context.context_manager is not None:
            built = context.context_manager.build_planner_context(state)
            updates["context_stats"] = context.context_manager.record_node(state, built)
            lines[2:2] = [
                "[Context Manager] Planner context built from resolved Query + session constraints.",
                f"Raw Context Tokens: {built.budget.raw_tokens}",
                f"Final Context Tokens: {built.budget.final_tokens}/{built.budget.budget}",
            ]
        lines.extend(("[State Update] plan=current structured plan", "[Edge] Planner -> Researcher"))
        return updates

    return planner_node
