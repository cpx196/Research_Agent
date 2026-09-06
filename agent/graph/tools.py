"""LangGraph ToolNode adapters for the existing V0/V2 tool functions.

The functions in ``tools/`` remain the source of truth.  This module only
adapts their existing OpenAI-style schemas to LangChain ``StructuredTool``
objects so the official LangGraph ``ToolNode`` can execute them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from pydantic import Field, create_model

from tools import TOOL_REGISTRY, TOOL_SCHEMAS


_JSON_TYPES: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _args_model(name: str, parameters: Mapping[str, Any]) -> type[Any]:
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_schema in properties.items():
        field_schema = field_schema if isinstance(field_schema, Mapping) else {}
        annotation = _JSON_TYPES.get(str(field_schema.get("type", "string")), Any)
        description = str(field_schema.get("description", ""))
        if field_name in required:
            default: Any = ...
        else:
            default = field_schema.get("default", None)
            annotation = annotation | None
        fields[field_name] = (annotation, Field(default, description=description))
    return create_model(name, **fields)


def build_langgraph_tools(
    registry: Mapping[str, Any] | None = None,
    schemas: Sequence[Mapping[str, Any]] | None = None,
) -> list[StructuredTool]:
    """Return ToolNode-compatible adapters over the central tool registry."""

    active_registry = registry if registry is not None else TOOL_REGISTRY
    active_schemas = schemas if schemas is not None else TOOL_SCHEMAS
    result: list[StructuredTool] = []
    for schema in active_schemas:
        function_schema = schema.get("function", {})
        name = str(function_schema.get("name", ""))
        if not name or name not in active_registry:
            continue
        parameters = function_schema.get("parameters", {})
        args_schema = _args_model(f"{name.title().replace('_', '')}Input", parameters)
        result.append(
            StructuredTool.from_function(
                func=active_registry[name],
                name=name,
                description=str(function_schema.get("description", name)),
                args_schema=args_schema,
                infer_schema=False,
            )
        )
    return result


def filter_tool_schemas(
    schemas: Sequence[Mapping[str, Any]],
    allowed_names: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Keep only schemas explicitly permitted by the current route."""

    allowed = {str(name) for name in allowed_names}
    return [
        schema
        for schema in schemas
        if str(schema.get("function", {}).get("name", "")) in allowed
    ]


def make_guarded_tool_node(
    registry: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
    allowed_names: Callable[[Mapping[str, Any]], Sequence[str]],
):
    """Build a ToolNode that enforces a runtime tool allowlist.

    ReAct receives a filtered schema list, but providers can still return a
    malformed or stale tool call.  This wrapper is the execution-side guard:
    allowed calls are delegated to LangGraph's official ToolNode, while
    disallowed calls become non-executing ToolError observations.
    """

    def guarded_node(state: Mapping[str, Any]) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        last_message = messages[-1] if messages else None
        if not isinstance(last_message, AIMessage):
            return {"messages": []}

        requested_calls = list(last_message.tool_calls or [])
        allowed = {
            str(name)
            for name in allowed_names(state)
            if str(name).strip()
        }
        available = {
            str(schema.get("function", {}).get("name", ""))
            for schema in schemas
            if schema.get("function", {}).get("name")
        } & set(registry)
        executable_calls = [
            call
            for call in requested_calls
            if str(call.get("name", "")) in allowed
            and str(call.get("name", "")) in available
        ]
        blocked_calls = [call for call in requested_calls if call not in executable_calls]

        tool_messages: list[ToolMessage] = []
        if executable_calls:
            executable_names = {str(call.get("name", "")) for call in executable_calls}
            executable_schemas = filter_tool_schemas(schemas, executable_names)
            # Constructing the ToolNode from the current allowlist makes the
            # runtime boundary match the schemas shown to the model.
            tool_node = ToolNode(build_langgraph_tools(registry, executable_schemas))
            if blocked_calls:
                filtered_message = AIMessage(
                    content=last_message.content,
                    tool_calls=executable_calls,
                )
                filtered_state = dict(state)
                filtered_state["messages"] = messages[:-1] + [filtered_message]
                result = tool_node.invoke(filtered_state)
            else:
                result = tool_node.invoke(state)
            tool_messages.extend(
                message
                for message in result.get("messages", [])
                if isinstance(message, ToolMessage)
            )

        blocked_trace: list[str] = []
        for call in blocked_calls:
            name = str(call.get("name", "unknown"))
            allowed_text = ", ".join(sorted(allowed)) or "(none)"
            tool_messages.append(
                ToolMessage(
                    content=(
                        f"ToolError: Tool '{name}' is not allowed by Router for this route. "
                        f"Allowed tools: {allowed_text}."
                    ),
                    tool_call_id=str(call.get("id", "")),
                    name=name,
                )
            )
            blocked_trace.append(f"[Tool Guard] Blocked disallowed tool: {name}")

        updates: dict[str, Any] = {"messages": tool_messages}
        if blocked_trace:
            updates["graph_trace"] = blocked_trace
        return updates

    return guarded_node
