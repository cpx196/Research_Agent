"""Bounded ReAct node for short tool-calling tasks."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage

from ..context import WorkflowContext
from ..state import ResearchState
from .common import message_to_openai, response_to_ai_message, trace
from ..tools import filter_tool_schemas


def _explicit_calculator_expression(query: str) -> str:
    expression = re.sub(r"^\s*(?:计算|算一下|calculate)\s*", "", query, flags=re.IGNORECASE).strip()
    if re.fullmatch(r"[\d\s()+\-*/%.]+", expression):
        return expression
    return ""


def _bind_explicit_calculator_call(ai_message: AIMessage, query: str) -> AIMessage:
    expression = _explicit_calculator_expression(query)
    if not expression or not ai_message.tool_calls:
        return ai_message
    first = ai_message.tool_calls[0]
    if str(first.get("name", "")) != "calculator":
        return ai_message
    call = dict(first)
    call["args"] = {"expression": expression}
    return AIMessage(content=ai_message.content, tool_calls=[call])


def _bind_routed_search_call(ai_message: AIMessage, state: ResearchState) -> AIMessage:
    """Bind search arguments to Router-produced standalone queries."""

    if not ai_message.tool_calls:
        return ai_message
    first = ai_message.tool_calls[0]
    name = str(first.get("name", ""))
    if name not in {"web_search", "local_search", "paper_search"}:
        return ai_message
    decision = dict(state.get("route_decision", {}))
    search_queries = dict(decision.get("search_queries", {}))
    routed_query = str(
        state.get("retry_query")
        or search_queries.get(name)
        or state.get("standalone_query")
        or state.get("query", "")
    ).strip()
    call = dict(first)
    proposed_args = call.get("args", {})
    args = dict(proposed_args) if isinstance(proposed_args, dict) else {}
    args["query"] = routed_query
    if name == "local_search":
        args["top_k"] = int(args.get("top_k", 5))
        args.pop("max_results", None)
    else:
        args["max_results"] = int(args.get("max_results", 5))
        args.pop("top_k", None)
    call["args"] = args
    return AIMessage(content=ai_message.content, tool_calls=[call])


def _force_controlled_retry(ai_message: AIMessage, state: ResearchState, finalization: bool) -> AIMessage:
    checks = list(state.get("relevance_checks", []))
    retry_query = str(state.get("retry_query", "")).strip()
    if finalization or not checks or bool(checks[-1].get("passed")) or not retry_query:
        return ai_message
    if ai_message.tool_calls:
        return ai_message
    decision = dict(state.get("route_decision", {}))
    allowed = [str(name) for name in decision.get("allowed_tools", [])]
    tool = str(state.get("retry_tool") or decision.get("primary_tool", ""))
    if tool not in allowed:
        tool = allowed[0] if allowed else ""
    if tool not in {"web_search", "local_search", "paper_search"}:
        return ai_message
    args: dict[str, Any] = {"query": retry_query}
    args["top_k" if tool == "local_search" else "max_results"] = 5
    return AIMessage(
        content="",
        tool_calls=[{
            "name": tool,
            "args": args,
            "id": f"react-retry-{int(state.get('retrieval_retry_count', 0))}",
            "type": "tool_call",
        }],
    )


def make_react_node(context: WorkflowContext):
    def react_node(state: ResearchState) -> dict[str, Any]:
        steps = int(state.get("react_steps", 0))
        max_steps = max(1, int(state.get("max_iterations", context.max_iterations)))
        finalization = steps >= max_steps
        instruction = (
            "You are a bounded ReAct agent. Use an available tool only when it helps answer the "
            "current query. After observing a tool result, either call another necessary tool or "
            "give the final answer. Follow conversation constraints and do not invent facts."
        )
        if finalization:
            instruction += " The tool-call budget is exhausted; now provide the final answer using the observations."
        standalone_query = str(state.get("standalone_query") or state.get("query", ""))
        instruction += f" The resolved standalone query is: {standalone_query}"
        checks = list(state.get("relevance_checks", []))
        if checks and not bool(checks[-1].get("passed")):
            retry_query = str(state.get("retry_query", ""))
            if retry_query and not finalization:
                instruction += (
                    " The previous tool result was irrelevant. Retry with the permitted tool using this exact "
                    f"query: {retry_query}"
                )
            else:
                instruction += " Ignore the irrelevant observation and state that reliable evidence was not found."
        language = str(state.get("session_preferences", {}).get("response_language", ""))
        if language == "zh-CN":
            instruction += " Respond in Simplified Chinese."
        elif language == "en":
            instruction += " Respond in English."
        allowed_tools = [
            str(name)
            for name in state.get("route_decision", {}).get("allowed_tools", [])
            if str(name).strip()
        ]
        allowed_schemas = filter_tool_schemas(context.tool_schemas, allowed_tools)
        if allowed_schemas:
            instruction += (
                " Only use the following tools for this route: "
                + ", ".join(allowed_tools)
                + ". Do not request any other tool."
            )
        else:
            instruction += " No tools are permitted for this route; provide the answer directly."
        messages = [
            {"role": "system", "content": instruction},
            *[message_to_openai(message) for message in state.get("messages", [])],
        ]
        ai_message: AIMessage | None = None
        try:
            response = context.llm.chat(
                messages=messages,
                tools=None if finalization else allowed_schemas,
            )
            ai_message = response_to_ai_message(response)
        except Exception:
            ai_message = None

        if ai_message is None:
            ai_message = AIMessage(content="AgentError: ReAct returned an empty response.")
        ai_message = _bind_explicit_calculator_call(ai_message, str(state.get("query", "")))
        ai_message = _force_controlled_retry(ai_message, state, finalization)
        ai_message = _bind_routed_search_call(ai_message, state)
        has_tool_call = bool(ai_message.tool_calls) and not finalization
        next_steps = steps + 1 if has_tool_call else steps
        draft = "" if has_tool_call else str(ai_message.content or "").strip()
        if not has_tool_call and not draft:
            draft = "AgentError: ReAct returned an empty final answer."
            ai_message = AIMessage(content=draft)
        lines = trace(
            "[Node] ReAct",
            f"Tool-call iteration: {steps}/{max_steps}",
            f"Tool call requested: {has_tool_call}",
            "[Edge] ReAct -> ToolNode" if has_tool_call else "[Edge] ReAct -> END",
        )
        updates: dict[str, Any] = {
            "messages": [ai_message],
            "react_steps": next_steps,
            "react_pending": has_tool_call,
            "verification_passed": not has_tool_call,
            "verification_feedback": "Skipped by Router: bounded ReAct path." if not has_tool_call else "",
            "graph_trace": lines,
        }
        if draft:
            updates["draft"] = draft
        return updates

    return react_node
