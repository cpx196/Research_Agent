"""Native Python tool-calling agent loop."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from prompts.system_prompt import SYSTEM_PROMPT
from tools import TOOL_REGISTRY, TOOL_SCHEMAS, execute_tool


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


class TraceLogger:
    """Print and optionally persist human-readable loop traces."""

    def __init__(
        self,
        verbose: bool = True,
        stream: TextIO | None = None,
        file_path: str | Path | None = None,
        on_emit: Any | None = None,
    ) -> None:
        self.verbose = verbose
        self.stream = stream or sys.stdout
        self.file_path = Path(file_path) if file_path else None
        self.on_emit = on_emit
        self.lines: list[str] = []

    def emit(self, line: str = "") -> None:
        self.lines.append(line)
        if self.verbose:
            print(line, file=self.stream)
        if self.file_path:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self.on_emit is not None:
            self.on_emit(line)

    def separator(self) -> None:
        self.emit("-" * 72)

    def text(self) -> str:
        return "\n".join(self.lines)


class ResearchAgent:
    """A minimal LLM + tools + state + loop implementation."""

    def __init__(
        self,
        llm: Any,
        tools: Mapping[str, Any] | None = None,
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
        max_steps: int = 10,
        verbose: bool = True,
        trace_file: str | Path | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.llm = llm
        self.tools = dict(tools or TOOL_REGISTRY)
        self.tool_schemas = list(tool_schemas or TOOL_SCHEMAS)
        self.max_steps = max_steps
        self.trace = TraceLogger(verbose=verbose, file_path=trace_file)
        self.last_messages: list[dict[str, Any]] = []
        self.last_run_stats: dict[str, Any] = {}

    def run(
        self,
        query: str,
        *,
        conversation_history: Sequence[Mapping[str, str]] | None = None,
        session_preferences: Mapping[str, str] | None = None,
    ) -> str:
        """Run one independent query until a final answer or max_steps."""

        query = query.strip()
        if not query:
            return "AgentError: Query cannot be empty."

        preference_text = ""
        if session_preferences:
            preference_text = "\nSession preferences: " + json.dumps(
                dict(session_preferences), ensure_ascii=False
            )
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT + preference_text}]
        messages.extend(
            {
                "role": "assistant" if item.get("role") == "assistant" else "user",
                "content": str(item.get("content", "")),
            }
            for item in (conversation_history or [])[-12:]
            if str(item.get("content", "")).strip()
        )
        messages.append({"role": "user", "content": query})
        started_at = time.perf_counter()
        tool_call_count = 0
        tool_error_count = 0
        final_answer: str | None = None

        self.trace.separator()
        self.trace.emit(f"[Query] {query}")

        for step in range(1, self.max_steps + 1):
            self.trace.emit("")
            self.trace.emit(f"[Agent] Step {step}")
            try:
                response = self.llm.chat(messages=messages, tools=self.tool_schemas)
                assistant_message = self._extract_message(response)
            except Exception as exc:  # keep a provider failure from killing the CLI
                final_answer = f"AgentError: {type(exc).__name__}: {exc}"
                self.trace.emit(f"[LLM Error] {final_answer}")
                break

            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                content = assistant_message.get("content")
                final_answer = str(content).strip() if content is not None else ""
                if not final_answer:
                    final_answer = "AgentError: LLM returned an empty final answer."
                self.trace.emit("[Final Answer]")
                self.trace.emit(final_answer)
                break

            for index, tool_call in enumerate(tool_calls, start=1):
                tool_call_count += 1
                name, raw_arguments, call_id = self._tool_call_parts(tool_call, step, index)
                self.trace.emit("[LLM]")
                self.trace.emit(f"Tool Call: {name}")
                self.trace.emit("Arguments:")
                self.trace.emit(self._pretty_json_or_text(raw_arguments))

                arguments, parse_error = self._parse_arguments(raw_arguments)
                if parse_error:
                    observation = parse_error
                elif not name:
                    observation = "ToolError: Invalid tool call: missing tool name"
                else:
                    self.trace.emit("[Tool]")
                    self.trace.emit(name)
                    observation = execute_tool(name, arguments, self.tools)

                if observation.startswith("ToolError:"):
                    tool_error_count += 1
                self.trace.emit("[Observation]")
                self.trace.emit(observation)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": observation,
                    }
                )

        if final_answer is None:
            final_answer = "Agent stopped because max_steps was reached."
            self.trace.emit("[Agent Stop]")
            self.trace.emit(final_answer)

        self.last_messages = messages
        self.last_run_stats = {
            "query": query,
            "steps": step if "step" in locals() else 0,
            "tool_calls": tool_call_count,
            "tool_errors": tool_error_count,
            "latency_seconds": round(time.perf_counter() - started_at, 3),
            "final_answer": final_answer,
        }
        self.trace.emit(
            "[Stats] "
            f"steps={self.last_run_stats['steps']} "
            f"tool_calls={tool_call_count} "
            f"tool_errors={tool_error_count} "
            f"latency_seconds={self.last_run_stats['latency_seconds']}"
        )
        return final_answer

    @staticmethod
    def _extract_message(response: Any) -> dict[str, Any]:
        message = _read(response, "message")
        if message is None:
            choices = _read(response, "choices", [])
            if not choices:
                raise ValueError("LLM response contains no choices")
            message = _read(choices[0], "message")
        if message is None:
            raise ValueError("LLM response contains no message")

        normalized: dict[str, Any] = {
            "role": _read(message, "role", "assistant"),
            "content": _read(message, "content"),
        }
        tool_calls = _read(message, "tool_calls")
        if tool_calls:
            normalized["tool_calls"] = tool_calls
        return normalized

    @staticmethod
    def _tool_call_parts(tool_call: Any, step: int, index: int) -> tuple[str, Any, str]:
        function = _read(tool_call, "function", {})
        name = str(_read(function, "name", "") or "")
        raw_arguments = _read(function, "arguments", "")
        call_id = str(_read(tool_call, "id", f"call_{step}_{index}") or f"call_{step}_{index}")
        return name, raw_arguments, call_id

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> tuple[dict[str, Any], str | None]:
        if isinstance(raw_arguments, Mapping):
            return dict(raw_arguments), None
        if not isinstance(raw_arguments, str) or not raw_arguments.strip():
            return {}, "ToolError: Invalid JSON arguments"
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            return {}, "ToolError: Invalid JSON arguments"
        if not isinstance(arguments, dict):
            return {}, "ToolError: Invalid JSON arguments"
        return arguments, None

    @staticmethod
    def _pretty_json_or_text(value: Any) -> str:
        if isinstance(value, str):
            try:
                return json.dumps(json.loads(value), ensure_ascii=False, indent=2)
            except (TypeError, json.JSONDecodeError):
                return value
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            return str(value)
