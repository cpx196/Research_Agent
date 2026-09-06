"""Small message and tracing helpers shared by graph nodes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage


def message_to_openai(message: AnyMessage) -> dict[str, Any]:
    """Convert a LangChain message to the raw mapping used by ``LLMClient``."""

    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, ToolMessage):
        role = "tool"
    else:
        role = "assistant"

    result: dict[str, Any] = {
        "role": role,
        "content": message.content,
    }
    if isinstance(message, ToolMessage):
        result["tool_call_id"] = message.tool_call_id
    if isinstance(message, AIMessage) and message.tool_calls:
        result["tool_calls"] = [
            {
                "id": str(call.get("id", "")),
                "type": "function",
                "function": {
                    "name": str(call.get("name", "")),
                    "arguments": json.dumps(call.get("args", {}), ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return result


def extract_response_message(response: Any) -> Mapping[str, Any]:
    """Extract a message mapping from an OpenAI-compatible response."""

    if isinstance(response, Mapping) and response.get("message") is not None:
        message = response["message"]
    else:
        choices = response.get("choices", []) if isinstance(response, Mapping) else []
        if not choices:
            return {}
        choice = choices[0]
        message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
    return message if isinstance(message, Mapping) else {}


def response_to_ai_message(response: Any) -> AIMessage | None:
    """Convert a raw LLM response into a LangChain AIMessage."""

    message = extract_response_message(response)
    content = message.get("content") or ""
    calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(message.get("tool_calls", []) or [], start=1):
        if not isinstance(raw_call, Mapping):
            continue
        function = raw_call.get("function", {})
        if not isinstance(function, Mapping):
            continue
        raw_args = function.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args
        if not isinstance(args, dict):
            args = {}
        name = str(function.get("name", ""))
        if name:
            calls.append(
                {
                    "name": name,
                    "args": args,
                    "id": str(raw_call.get("id", f"research-call-{index}")),
                    "type": "tool_call",
                }
            )
    if not content and not calls:
        return None
    return AIMessage(content=str(content), tool_calls=calls)


def trace(*lines: object) -> list[str]:
    return [str(line) for line in lines]

