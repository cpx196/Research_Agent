"""Tool functions, schemas, and the central registry."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Callable

from .calculator import calculator, calculator_schema
from .local_search import local_search, local_search_schema, warmup_local_rag
from .paper_search import paper_search, paper_search_schema
from .web_search import web_search, web_search_schema

ToolFunction = Callable[..., str]

TOOL_REGISTRY: dict[str, ToolFunction] = {
    "calculator": calculator,
    "web_search": web_search,
    "paper_search": paper_search,
    "local_search": local_search,
}

TOOL_SCHEMAS = [calculator_schema, web_search_schema, paper_search_schema, local_search_schema]


def execute_tool(
    name: str,
    arguments: Mapping[str, Any],
    registry: Mapping[str, ToolFunction] | None = None,
) -> str:
    """Execute a registered tool and convert every failure to an observation."""

    active_registry = registry or TOOL_REGISTRY
    if name not in active_registry:
        return f"ToolError: Unknown tool {name}"
    if not isinstance(arguments, Mapping):
        return "ToolError: Invalid tool arguments"

    function = active_registry[name]
    try:
        signature = inspect.signature(function)
        missing = [
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and parameter.name not in arguments
        ]
        if missing:
            return f"ToolError: Missing required argument(s): {', '.join(missing)}"
        return str(function(**dict(arguments)))
    except Exception as exc:  # a tool must never crash the agent loop
        return f"ToolError: {type(exc).__name__}: {exc}"


__all__ = [
    "TOOL_REGISTRY",
    "TOOL_SCHEMAS",
    "calculator",
    "execute_tool",
    "local_search",
    "warmup_local_rag",
    "paper_search",
    "web_search",
]
