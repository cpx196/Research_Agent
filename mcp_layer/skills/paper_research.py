"""A small paper-research Skill composed from local and optional external tools."""

from __future__ import annotations

from typing import Any

from ..registry import UnifiedToolRegistry


class PaperResearchSkill:
    """Run a fixed, auditable paper-research workflow.

    V5 defines a Skill as multiple tools plus a fixed workflow. The external
    search step is opt-in so CPU/offline demos and tests remain deterministic.
    """

    name = "paper_research"

    def __init__(self, registry: UnifiedToolRegistry) -> None:
        self.registry = registry

    async def run(self, topic: str, *, include_external: bool = False, top_k: int = 5) -> dict[str, Any]:
        topic = topic.strip()
        if not topic:
            return {"skill": self.name, "status": "error", "message": "topic is empty"}
        steps: list[dict[str, Any]] = []
        local = await self.registry.call("local_search", {"query": topic, "top_k": top_k})
        steps.append({"tool": "local_search", "source": local.source, "content": local.content})
        if include_external:
            external = await self.registry.call("paper_search", {"query": topic, "max_results": 5})
            steps.append({"tool": "paper_search", "source": external.source, "content": external.content})
        good = any(not step["content"].startswith(("ToolError:", "MCPError:")) for step in steps)
        return {
            "skill": self.name,
            "status": "ok" if good else "error",
            "topic": topic,
            "steps": steps,
            "evidence": [step["content"] for step in steps if step["content"]],
        }
