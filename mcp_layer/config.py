"""Configuration for local and external MCP servers."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MCPServerConfig:
    """How the Host starts one MCP server.

    V5 intentionally starts with stdio.  Keeping command/args separate makes
    the same manager usable for a future HTTP transport without changing the
    graph or tool adapters.
    """

    name: str
    command: str
    args: tuple[str, ...] = field(default_factory=tuple)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"
    timeout_seconds: float = 20.0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "command": self.command,
            "args": list(self.args),
            "cwd": self.cwd,
            "transport": self.transport,
            "timeout_seconds": self.timeout_seconds,
        }


def default_research_server_config(project_root: Path | None = None) -> MCPServerConfig:
    """Return a portable config that uses the currently running Python."""

    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    timeout = float(os.getenv("MCP_TIMEOUT_SECONDS", "20"))
    return MCPServerConfig(
        name="research-mcp",
        command=sys.executable,
        args=(str(root / "mcp_layer" / "servers" / "research_server.py"),),
        cwd=str(root),
        timeout_seconds=max(1.0, timeout),
    )
