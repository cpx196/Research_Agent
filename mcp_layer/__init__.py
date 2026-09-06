"""V5 MCP integration layer.

The project keeps the original native registry intact and uses this package to
connect MCP servers, normalize their schemas, and expose a common tool API.
"""

from .adapters import MCPToolAdapter, NativeToolAdapter, ToolCallResult, ToolSpec
from .client import MCPClient, MCPClientError
from .config import MCPServerConfig, default_research_server_config
from .manager import MCPClientManager
from .registry import UnifiedToolRegistry

__all__ = [
    "MCPClient",
    "MCPClientError",
    "MCPClientManager",
    "MCPServerConfig",
    "MCPToolAdapter",
    "NativeToolAdapter",
    "ToolCallResult",
    "ToolSpec",
    "UnifiedToolRegistry",
    "default_research_server_config",
]
