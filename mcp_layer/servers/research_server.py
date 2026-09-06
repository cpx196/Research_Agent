"""Research MCP Server used by V5's stdio integration.

Only the explicitly exposed research tools/resources are available.  The
server never writes diagnostics to stdout because stdout is the MCP protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.calculator import calculator  # noqa: E402
from tools.paper_search import paper_search  # noqa: E402


mcp = FastMCP(
    "research-mcp",
    instructions="Controlled research tools for the Research Agent V5 Host.",
    log_level="ERROR",
)


@mcp.tool(name="calculator")
def calculator_tool(expression: str) -> str:
    """Evaluate a basic mathematical expression safely."""

    return calculator(expression)


@mcp.tool(name="paper_search")
def paper_search_tool(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers matching a research query."""

    return paper_search(query, max_results=max_results)


@mcp.resource("research://manifest", name="paper_manifest", mime_type="application/json")
def paper_manifest() -> str:
    """Expose only the local paper manifest, never arbitrary filesystem paths."""

    manifest = PROJECT_ROOT / "papers" / "manifest.json"
    if not manifest.is_file():
        return json.dumps({"status": "missing", "path": "papers/manifest.json"})
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return json.dumps({"status": "error", "message": str(exc)})
    return json.dumps(data, ensure_ascii=False)


@mcp.prompt(name="paper_research")
def paper_research_prompt(topic: str) -> str:
    """Return a reusable fixed workflow prompt for paper research."""

    return (
        f"研究主题：{topic}\n"
        "先调用 paper_search 获取候选论文，再提炼问题、方法、证据和局限；"
        "所有结论都要标注来源。"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
