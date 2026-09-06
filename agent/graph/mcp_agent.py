"""V5 LangGraph runner using MCP-discovered tools with native fallback."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.context import ContextConfig
from mcp_layer import MCPClientManager, UnifiedToolRegistry, default_research_server_config

from .graph import LangGraphResearchAgent


class MCPResearchAgent:
    """V5 Host: discover MCP tools, then execute them inside the V3/V4 graph."""

    def __init__(
        self,
        llm: Any,
        *,
        max_iterations: int = 2,
        verifier_mode: str = "simple",
        max_verification_tool_calls: int = 2,
        verbose: bool = True,
        trace_file: str | Path | None = None,
        context_config: ContextConfig | None = None,
        mcp_config: Any | None = None,
    ) -> None:
        self.mcp_config = mcp_config or default_research_server_config()
        self.manager = MCPClientManager([self.mcp_config])
        self.registry = UnifiedToolRegistry(manager=self.manager, mcp_config=self.mcp_config)
        self.discovery_error = ""
        try:
            schemas = asyncio.run(self.manager.discover())
            self.registry.add_mcp_tools(schemas, self.mcp_config)
            if self.manager.last_error:
                self.discovery_error = self.manager.last_error
        except Exception as exc:
            self.discovery_error = f"MCP discovery failed: {type(exc).__name__}: {exc}"
        self.agent = LangGraphResearchAgent(
            llm,
            tools=self.registry.callables,
            tool_schemas=self.registry.schemas,
            max_iterations=max_iterations,
            verifier_mode=verifier_mode,
            max_verification_tool_calls=max_verification_tool_calls,
            verbose=verbose,
            trace_file=trace_file,
            context_enabled=True,
            context_config=context_config,
        )
        self.trace = self.agent.trace
        self.last_state = self.agent.last_state
        self.last_run_stats: dict[str, Any] = {}

    @property
    def stream_callback(self):
        return self.agent.stream_callback

    @stream_callback.setter
    def stream_callback(self, callback):
        self.agent.stream_callback = callback

    @property
    def live_trace_callback(self):
        return self.agent.live_trace_callback

    @live_trace_callback.setter
    def live_trace_callback(self, callback):
        self.agent.live_trace_callback = callback

    def run(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        conversation_history: Sequence[Mapping[str, str]] | None = None,
        session_preferences: Mapping[str, str] | None = None,
    ) -> str:
        history_start = len(self.registry.call_history)
        answer = self.agent.run(
            query,
            thread_id=thread_id,
            conversation_history=conversation_history,
            session_preferences=session_preferences,
        )
        self.last_state = self.agent.last_state
        self.last_run_stats = dict(self.agent.last_run_stats)
        run_history = self.registry.call_history[history_start:]
        self.last_run_stats.update({
            "mcp_discovery_error": self.discovery_error,
            "mcp_tools": [
                name for name, adapter in self.registry.adapters.items()
                if getattr(getattr(adapter, "spec", None), "source", "") == "MCP"
            ],
            "tool_sources": [result.source for result in run_history],
        })
        # Append V5 protocol fields to the same trace stream used by V3/V4.
        for result in run_history:
            for trace_line in result.observation().splitlines():
                if self.live_trace_callback is not None:
                    self.live_trace_callback(trace_line)
                self.trace.emit(trace_line)
        return answer


__all__ = ["MCPResearchAgent"]
