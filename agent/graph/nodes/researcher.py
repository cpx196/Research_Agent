"""Researcher node: choose one planned tool call for the ToolNode."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from ..context import WorkflowContext
from ..state import PlanStep, ResearchState
from ..tools import filter_tool_schemas
from .common import response_to_ai_message, trace


def _fallback_tool_call(step: PlanStep, query: str, step_index: int) -> AIMessage:
    tool = str(step.get("tool", "local_search"))
    task = str(step.get("task", query))
    if tool == "calculator":
        arguments = {"expression": query.replace("计算", "").replace("算一下", "").strip()}
    elif tool == "paper_search":
        arguments = {"query": task, "max_results": 5}
    elif tool == "web_search":
        arguments = {"query": task, "max_results": 5}
    else:
        tool = "local_search"
        arguments = {"query": task, "top_k": 5}
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool,
                "args": arguments,
                "id": f"graph-research-{step_index}",
                "type": "tool_call",
            }
        ],
    )


def _normalize_planned_call(ai_message: AIMessage, step: PlanStep, query: str, step_index: int) -> AIMessage:
    """Keep provider tool selection but bind arguments to the plan task.

    Some lightweight/demo adapters echo the whole researcher prompt as the
    search query.  The workflow should never index that control prompt, so
    the structured plan remains the source of truth for tool arguments.
    """

    preferred_tool = str(step.get("tool", "local_search"))
    task = str(step.get("task", query))
    proposed = ai_message.tool_calls[0]
    proposed_args = proposed.get("args", {})
    if not isinstance(proposed_args, dict):
        proposed_args = {}
    if preferred_tool == "local_search":
        args = {"query": task, "top_k": int(proposed_args.get("top_k", 5))}
    elif preferred_tool == "paper_search":
        args = {"query": task, "max_results": int(proposed_args.get("max_results", 5))}
    elif preferred_tool == "web_search":
        args = {"query": task, "max_results": int(proposed_args.get("max_results", 5))}
    else:
        args = proposed_args
    return AIMessage(
        content=ai_message.content,
        tool_calls=[
            {
                "name": preferred_tool,
                "args": args,
                # A stable per-plan-step id keeps Evidence records distinct
                # even when a demo/provider reuses the same call id.
                "id": f"graph-research-{step_index}",
                "type": "tool_call",
            }
        ],
    )


def make_researcher_node(context: WorkflowContext):
    def researcher_node(state: ResearchState) -> dict[str, Any]:
        plan = list(state.get("plan", []))
        current_step = int(state.get("current_step", 0))
        query = str(state.get("standalone_query") or state.get("query", ""))
        if current_step >= len(plan):
            updates: dict[str, Any] = {
                "graph_trace": trace(
                    "[Node] Researcher",
                    "No pending plan step.",
                    "[Edge] Researcher -> Writer",
                )
            }
            if context.context_manager is not None:
                built = context.context_manager.build_researcher_context(state)
                updates["context_stats"] = context.context_manager.record_node(state, built)
                updates["graph_trace"][1:1] = [
                    "[Context Manager] Researcher context includes compact session history without raw tool history.",
                    f"Raw Context Tokens: {built.budget.raw_tokens}",
                    f"Final Context Tokens: {built.budget.final_tokens}/{built.budget.budget}",
                ]
            return updates

        step = plan[current_step]
        built = context.context_manager.build_researcher_context(state) if context.context_manager else None
        prompt = (
            "You are the Researcher node in a controlled workflow. Execute exactly one "
            "pending plan step by calling the most appropriate available tool. Return one "
            f"tool call only. Original query: {query}\n"
            f"Plan step {current_step + 1}: {step.get('task', query)}\n"
            f"Preferred tool: {step.get('tool', 'local_search')}"
        )
        if built is not None:
            prompt = (
                "You are the Researcher node. Use this node-specific context and execute "
                "exactly one tool call for the current step.\n\n" + built.text
            )
        ai_message = None
        preferred_tool = str(step.get("tool", "local_search"))
        planned_schemas = filter_tool_schemas(context.tool_schemas, [preferred_tool])
        try:
            response = context.llm.chat(
                messages=[
                    {"role": "system", "content": "You are a tool-selection researcher."},
                    {"role": "user", "content": prompt},
                ],
                tools=planned_schemas,
            )
            ai_message = response_to_ai_message(response)
        except Exception:
            ai_message = None

        # If a provider returns text instead of a tool call, preserve the
        # workflow contract with a deterministic call for the planned step.
        preferred_tool = str(step.get("tool", "local_search"))
        if preferred_tool == "calculator":
            # Calculator arguments must preserve the user's exact expression;
            # the deterministic fallback avoids a provider substituting a
            # canned/demo expression.
            ai_message = _fallback_tool_call(step, query, current_step + 1)
        elif ai_message is None or not ai_message.tool_calls:
            ai_message = _fallback_tool_call(step, query, current_step + 1)
        else:
            # One ToolNode turn corresponds to one plan step.  Ignore extra
            # provider calls to keep state/evidence alignment deterministic.
            # The plan owns the intended tool; this prevents a provider/demo
            # adapter from silently executing a different task.
            proposed = ai_message.tool_calls[0]
            if str(proposed.get("name", "")) != preferred_tool:
                ai_message = _fallback_tool_call(step, query, current_step + 1)
            else:
                ai_message = _normalize_planned_call(ai_message, step, query, current_step + 1)

        call = ai_message.tool_calls[0]
        lines = trace(
            "[Node] Researcher",
            f"Iteration: {state.get('iteration', 0)}",
            f"Plan step: {current_step + 1}/{len(plan)}",
            f"Tool Call: {call.get('name', '')}",
            f"Arguments: {call.get('args', {})}",
        )
        if built is not None:
            lines[0:0] = [
                "[Context Manager] Researcher context excludes full messages/raw results and keeps compact session context.",
                f"Raw Context Tokens: {built.budget.raw_tokens}",
                f"Selected Evidence: {len(built.selected_evidence)}/{built.selection.available_count if built.selection else 0}",
                f"Final Context Tokens: {built.budget.final_tokens}/{built.budget.budget}",
            ]
        lines.append("[Edge] Researcher -> ToolNode")
        updates: dict[str, Any] = {
            "messages": [ai_message],
            "graph_trace": lines,
        }
        if built is not None:
            updates["context_stats"] = context.context_manager.record_node(state, built)
        return updates

    return researcher_node
