"""Run only the V5 smoke checks and produce two complete MCP traces."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent.graph import MCPResearchAgent
from agent.llm import DemoLLM
from mcp_layer.config import default_research_server_config
from mcp_layer.manager import MCPClientManager
from mcp_layer.registry import UnifiedToolRegistry
from mcp_layer.skills import PaperResearchSkill
from tools.local_search import local_search_schema


ROOT = Path(__file__).resolve().parents[1]


def fake_local_search(query: str, top_k: int = 5) -> str:
    return f"[Result 1]\nSource: v5-smoke.pdf\nPage: {top_k}\nText: local evidence for {query}"


async def protocol_smoke() -> dict[str, object]:
    manager = MCPClientManager([default_research_server_config()])
    statuses = await manager.connect()
    tools = await manager.list_tools()
    calculation = await manager.call_tool("calculator", {"expression": "6 * 7"})
    resources = await manager.list_resources()
    prompts = await manager.list_prompts()
    await manager.close()
    return {
        "status": statuses,
        "tools": [item["function"]["name"] for item in tools],
        "calculator": calculation.content,
        "resources": len(resources),
        "prompts": len(prompts),
    }


def main() -> None:
    traces_path = ROOT / "evaluation" / "traces_v5.md"
    results_path = ROOT / "evaluation" / "results_v5.md"
    traces_path.write_text("# V5 Trace\n\n", encoding="utf-8")
    smoke: dict[str, object] = {"protocol": asyncio.run(protocol_smoke())}

    graph_runs = []
    for index, query in enumerate(("计算 12 * 8", "计算 7 * 6"), start=1):
        trace_file = ROOT / "evaluation" / f".v5_trace_{index}.log"
        agent = MCPResearchAgent(DemoLLM(), verbose=False, trace_file=trace_file)
        answer = agent.run(query)
        trace_text = trace_file.read_text(encoding="utf-8")
        with traces_path.open("a", encoding="utf-8") as handle:
            handle.write(f"## Trace {index}: {query}\n\n```text\n{trace_text}```\n\n")
        graph_runs.append({
            "query": query,
            "answer": answer,
            "mcp_tools": agent.last_run_stats.get("mcp_tools", []),
            "tool_sources": agent.last_run_stats.get("tool_sources", []),
            "discovery_error": agent.discovery_error,
        })

    skill_registry = UnifiedToolRegistry(
        native_registry={"local_search": fake_local_search},
        native_schemas=[local_search_schema],
    )
    skill_result = asyncio.run(PaperResearchSkill(skill_registry).run("JEPA"))
    smoke["skill"] = {
        "name": skill_result["skill"],
        "status": skill_result["status"],
        "tools": [step["tool"] for step in skill_result["steps"]],
    }
    smoke["graph_runs"] = graph_runs
    results_path.write_text(
        "# V5 Smoke Results\n\n"
        "本次只做最小验收，不做大规模 benchmark。\n\n"
        "```json\n" + json.dumps(smoke, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    for index in range(1, 3):
        (ROOT / "evaluation" / f".v5_trace_{index}.log").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
