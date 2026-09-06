"""MCP client manager: lifecycle, discovery, timeout and safe error boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

from .client import MCPClient, MCPClientError, content_to_text
from .config import MCPServerConfig


@dataclass
class MCPCallResult:
    content: str
    is_error: bool = False
    server_name: str = ""
    transport: str = "stdio"


def _tool_schema(tool: Any) -> dict[str, Any]:
    """Convert an MCP Tool model to the project's OpenAI-compatible schema."""

    data = tool.model_dump(mode="json") if hasattr(tool, "model_dump") else dict(tool)
    return {
        "type": "function",
        "function": {
            "name": str(data.get("name", "")),
            "description": str(data.get("description") or data.get("title") or "MCP tool"),
            "parameters": data.get("inputSchema") or {
                "type": "object", "properties": {}, "additionalProperties": False
            },
        },
    }


class MCPClientManager:
    """Manage one or more MCP servers from the Research Agent Host."""

    def __init__(self, configs: list[MCPServerConfig] | None = None) -> None:
        self.configs = configs or []
        self.clients: dict[str, MCPClient] = {}
        self.discovered_tools: dict[str, dict[str, Any]] = {}
        self.last_error: str = ""

    async def connect(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for config in self.configs:
            try:
                client = self.clients.setdefault(config.name, MCPClient(config))
                await client.connect()
                statuses[config.name] = "connected"
            except MCPClientError as exc:
                self.last_error = str(exc)
                statuses[config.name] = f"error: {exc}"
        return statuses

    async def list_tools(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for config in self.configs:
            client = self.clients.get(config.name)
            if client is None or not client.connected:
                continue
            try:
                response = await client.list_tools()
                for tool in getattr(response, "tools", []):
                    schema = _tool_schema(tool)
                    name = schema["function"]["name"]
                    self.discovered_tools[name] = {
                        "schema": schema,
                        "server_name": config.name,
                        "transport": config.transport,
                    }
                    results.append(schema)
            except MCPClientError as exc:
                self.last_error = str(exc)
        return results

    async def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> MCPCallResult:
        info = self.discovered_tools.get(name)
        if info is None:
            return MCPCallResult(f"MCPError: Unknown tool {name}", True)
        client = self.clients.get(str(info["server_name"]))
        if client is None or not client.connected:
            return MCPCallResult("MCPError: Server is not connected", True, str(info["server_name"]))
        try:
            response = await client.call_tool(name, dict(arguments or {}))
            return MCPCallResult(
                content_to_text(getattr(response, "content", response)),
                bool(getattr(response, "isError", False)),
                str(info["server_name"]),
                str(info["transport"]),
            )
        except MCPClientError as exc:
            self.last_error = str(exc)
            return MCPCallResult(f"MCPError: {exc}", True, str(info["server_name"]), str(info["transport"]))

    async def list_resources(self) -> list[Any]:
        resources: list[Any] = []
        for client in self.clients.values():
            if client.connected:
                try:
                    resources.extend(getattr(await client.list_resources(), "resources", []))
                except MCPClientError as exc:
                    self.last_error = str(exc)
        return resources

    async def read_resource(self, server_name_or_uri: str, uri: str | None = None) -> str:
        """Read a resource by ``(server_name, uri)`` or by URI when unambiguous."""

        if uri is None:
            uri = server_name_or_uri
            client = next(iter(self.clients.values()), None)
        else:
            client = self.clients.get(server_name_or_uri)
        if client is None or not client.connected:
            return "MCPError: Server is not connected"
        try:
            response = await client.read_resource(uri)
            return content_to_text(getattr(response, "contents", response))
        except MCPClientError as exc:
            self.last_error = str(exc)
            return f"MCPError: {exc}"

    async def list_prompts(self) -> list[Any]:
        prompts: list[Any] = []
        for client in self.clients.values():
            if client.connected:
                try:
                    prompts.extend(getattr(await client.list_prompts(), "prompts", []))
                except MCPClientError as exc:
                    self.last_error = str(exc)
        return prompts

    async def get_prompt(self, server_name: str, name: str, arguments: dict[str, str] | None = None) -> str:
        client = self.clients.get(server_name)
        if client is None or not client.connected:
            return "MCPError: Server is not connected"
        try:
            response = await client.get_prompt(name, arguments)
            return content_to_text(getattr(response, "messages", response))
        except MCPClientError as exc:
            self.last_error = str(exc)
            return f"MCPError: {exc}"

    async def close(self) -> None:
        errors: list[str] = []
        for client in list(self.clients.values()):
            try:
                await client.close()
            except MCPClientError as exc:
                errors.append(str(exc))
        self.clients.clear()
        if errors:
            self.last_error = "; ".join(errors)

    async def discover(self) -> list[dict[str, Any]]:
        await self.connect()
        try:
            return await self.list_tools()
        finally:
            await self.close()

    async def call_tool_once(self, config: MCPServerConfig, name: str, arguments: Mapping[str, Any]) -> MCPCallResult:
        """Connect, call, and close in one loop-safe operation for sync graphs."""

        one_shot = MCPClientManager([config])
        await one_shot.connect()
        try:
            await one_shot.list_tools()
            return await one_shot.call_tool(name, arguments)
        finally:
            await one_shot.close()

    def call_tool_sync(self, config: MCPServerConfig, name: str, arguments: Mapping[str, Any]) -> MCPCallResult:
        try:
            return asyncio.run(self.call_tool_once(config, name, arguments))
        except Exception as exc:
            return MCPCallResult(f"MCPError: {type(exc).__name__}: {exc}", True, config.name, config.transport)
