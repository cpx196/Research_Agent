"""Unified registry built on top of, but separate from, the legacy registry."""

from __future__ import annotations

from typing import Any, Mapping

from tools import TOOL_REGISTRY, TOOL_SCHEMAS

from .adapters import MCPToolAdapter, NativeToolAdapter, ToolAdapter, ToolCallResult
from .config import MCPServerConfig
from .manager import MCPClientManager


class UnifiedToolRegistry:
    """Expose native and MCP tools through one schema/call surface."""

    def __init__(
        self,
        manager: MCPClientManager | None = None,
        mcp_config: MCPServerConfig | None = None,
        native_registry: Mapping[str, Any] | None = None,
        native_schemas: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self.manager = manager or MCPClientManager()
        self.mcp_config = mcp_config
        registry = dict(native_registry or TOOL_REGISTRY)
        schemas = list(native_schemas or TOOL_SCHEMAS)
        schema_by_name = {str(s.get("function", {}).get("name", "")): s for s in schemas}
        self.adapters: dict[str, ToolAdapter] = {
            name: NativeToolAdapter(name, function, schema_by_name.get(name, {"function": {"name": name}}))
            for name, function in registry.items()
        }
        self._schemas: dict[str, dict[str, Any]] = {
            name: adapter.spec.schema() for name, adapter in self.adapters.items()
        }
        self.call_history: list[ToolCallResult] = []

    def add_mcp_tools(self, schemas: list[Mapping[str, Any]], config: MCPServerConfig | None = None) -> None:
        config = config or self.mcp_config
        if config is None:
            raise ValueError("An MCP server config is required")
        for schema in schemas:
            name = str(schema.get("function", {}).get("name", ""))
            if not name:
                continue
            fallback = self.adapters.get(name)
            fallback = fallback if isinstance(fallback, NativeToolAdapter) else None
            self.adapters[name] = MCPToolAdapter(self.manager, config, schema, fallback=fallback)
            self._schemas[name] = dict(schema)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return list(self._schemas.values())

    @property
    def callables(self) -> dict[str, Any]:
        return {name: self._make_callable(adapter) for name, adapter in self.adapters.items()}

    def _make_callable(self, adapter: ToolAdapter):
        def call(**arguments: Any) -> str:
            result = adapter.call_sync(arguments)
            self.call_history.append(result)
            return result.observation()
        call.__name__ = adapter.spec.name
        return call

    async def call(self, name: str, arguments: Mapping[str, Any]) -> ToolCallResult:
        adapter = self.adapters.get(name)
        if adapter is None:
            result = ToolCallResult(f"ToolError: Unknown tool {name}", "Native", name, is_error=True)
        else:
            result = await adapter.call(arguments)  # type: ignore[attr-defined]
        self.call_history.append(result)
        return result
