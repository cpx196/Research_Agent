"""Small lifecycle wrapper around the official Python MCP SDK."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
import json
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MCPServerConfig


class MCPClientError(RuntimeError):
    """An MCP transport, protocol, or server-side failure."""


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def content_to_text(content: Any) -> str:
    """Convert MCP content blocks and ordinary values into an observation."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text_value = getattr(block, "text", None)
            parts.append(str(text_value if text_value is not None else _jsonable(block)))
        return "\n".join(parts)
    return json.dumps(_jsonable(content), ensure_ascii=False, default=str)


class MCPClient:
    """One connected stdio MCP client; callers own its async lifecycle."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> Any:
        if self.config.transport != "stdio":
            raise MCPClientError(f"Unsupported transport: {self.config.transport}")
        if self._session is not None:
            return None
        stack = AsyncExitStack()
        try:
            server = StdioServerParameters(
                command=self.config.command,
                args=list(self.config.args),
                cwd=self.config.cwd,
                env=self.config.env or None,
            )
            read_stream, write_stream = await stack.enter_async_context(stdio_client(server))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await asyncio.wait_for(session.initialize(), timeout=self.config.timeout_seconds)
            self._stack = stack
            self._session = session
            return None
        except Exception as exc:
            await stack.aclose()
            raise MCPClientError(f"MCP server connect failed: {exc}") from exc

    async def _request(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._session is None:
            raise MCPClientError("MCP client is not connected")
        operation = getattr(self._session, method)
        try:
            return await asyncio.wait_for(
                operation(*args, **kwargs), timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise MCPClientError(f"MCP timeout during {method}") from exc
        except Exception as exc:
            raise MCPClientError(f"MCP {method} failed: {exc}") from exc

    async def list_tools(self) -> Any:
        return await self._request("list_tools")

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return await self._request("call_tool", name, arguments or {})

    async def list_resources(self) -> Any:
        return await self._request("list_resources")

    async def read_resource(self, uri: str) -> Any:
        return await self._request("read_resource", uri)

    async def list_prompts(self) -> Any:
        return await self._request("list_prompts")

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> Any:
        return await self._request("get_prompt", name, arguments or {})

    async def close(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:
                raise MCPClientError(f"MCP client shutdown failed: {exc}") from exc

