"""Unified Native/MCP tool adapters and trace metadata."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from tools import execute_tool

from .config import MCPServerConfig
from .manager import MCPCallResult, MCPClientManager


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    source: str = "Native"
    server_name: str = ""
    transport: str = ""

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters
        }}


@dataclass
class ToolCallResult:
    content: str
    source: str
    tool_name: str
    server_name: str = ""
    transport: str = ""
    latency_ms: float = 0.0
    fallback: bool = False
    is_error: bool = False

    def observation(self) -> str:
        return "\n".join([
            f"[Tool Source] {self.source}",
            f"[MCP Server] {self.server_name or '-'}",
            f"[MCP Tool] {self.tool_name}",
            f"[Transport] {self.transport or '-'}",
            f"[Latency] {self.latency_ms:.1f}ms",
            f"[Fallback] {'Native' if self.fallback else 'None'}",
            f"[Observation]\n{self.content}",
        ])


class ToolAdapter(Protocol):
    spec: ToolSpec

    def call_sync(self, arguments: Mapping[str, Any]) -> ToolCallResult:
        ...


class NativeToolAdapter:
    def __init__(self, name: str, function: Callable[..., str], schema: Mapping[str, Any]) -> None:
        function_schema = schema.get("function", {})
        self.spec = ToolSpec(
            name=name,
            description=str(function_schema.get("description", name)),
            parameters=dict(function_schema.get("parameters", {})),
        )
        self.function = function

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    def call_sync(self, arguments: Mapping[str, Any]) -> ToolCallResult:
        started = time.perf_counter()
        content = execute_tool(self.spec.name, arguments, {self.spec.name: self.function})
        return ToolCallResult(content, "Native", self.spec.name, latency_ms=(time.perf_counter() - started) * 1000, is_error=content.startswith("ToolError:"))

    async def call(self, arguments: Mapping[str, Any]) -> ToolCallResult:
        return await asyncio.to_thread(self.call_sync, arguments)


class MCPToolAdapter:
    def __init__(
        self,
        manager: MCPClientManager,
        config: MCPServerConfig,
        schema: Mapping[str, Any],
        fallback: NativeToolAdapter | None = None,
    ) -> None:
        function_schema = schema.get("function", {})
        self.manager = manager
        self.config = config
        self.fallback = fallback
        self.spec = ToolSpec(
            name=str(function_schema.get("name", "")),
            description=str(function_schema.get("description", "MCP tool")),
            parameters=dict(function_schema.get("parameters", {})),
            source="MCP",
            server_name=config.name,
            transport=config.transport,
        )

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    def call_sync(self, arguments: Mapping[str, Any]) -> ToolCallResult:
        started = time.perf_counter()
        result: MCPCallResult = self.manager.call_tool_sync(self.config, self.spec.name, arguments)
        latency = (time.perf_counter() - started) * 1000
        if result.is_error and self.fallback is not None:
            native_result = self.fallback.call_sync(arguments)
            return ToolCallResult(
                native_result.content,
                "MCP",
                self.spec.name,
                self.config.name,
                self.config.transport,
                latency,
                fallback=True,
                is_error=native_result.is_error,
            )
        return ToolCallResult(
            result.content, "MCP", self.spec.name, result.server_name, result.transport,
            latency, fallback=False, is_error=result.is_error,
        )

    async def call(self, arguments: Mapping[str, Any]) -> ToolCallResult:
        return await asyncio.to_thread(self.call_sync, arguments)
