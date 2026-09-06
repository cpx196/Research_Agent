import asyncio
import unittest

from mcp_layer.registry import UnifiedToolRegistry
from mcp_layer.skills import PaperResearchSkill
from tools.local_search import local_search_schema


def fake_local_search(query: str, top_k: int = 5) -> str:
    return f"[Result 1]\nSource: fake.pdf\nPage: {top_k}\nText: evidence for {query}"


class SkillTests(unittest.TestCase):
    def test_paper_research_skill_runs_fixed_local_workflow(self) -> None:
        registry = UnifiedToolRegistry(
            native_registry={"local_search": fake_local_search},
            native_schemas=[local_search_schema],
        )
        result = asyncio.run(PaperResearchSkill(registry).run("JEPA", top_k=3))
        self.assertEqual(result["skill"], "paper_research")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["steps"][0]["tool"], "local_search")
        self.assertIn("fake.pdf", result["evidence"][0])


if __name__ == "__main__":
    unittest.main()
