import unittest

from agent.graph.nodes.planner import build_research_plan


class PlannerTests(unittest.TestCase):
    def test_person_lookup_uses_web_search(self) -> None:
        plan = build_research_plan("danking是谁")
        self.assertEqual(plan[0]["tool"], "web_search")

    def test_capability_question_is_answered_directly(self) -> None:
        self.assertEqual(build_research_plan("你不能联网搜索么？"), [])

    def test_current_local_topic_uses_local_and_web_sources(self) -> None:
        tools = [step["tool"] for step in build_research_plan("DINO 最新官方代码")]
        self.assertEqual(tools, ["local_search", "web_search"])

    def test_router_metadata_keeps_local_paper_and_web_as_independent_steps(self) -> None:
        decision = {
            "route": "research",
            "standalone_query": "DINO 最新官方代码",
            "topic": "DINO",
            "intent": "general",
            "primary_tool": "local_search",
            "allowed_tools": ["local_search", "paper_search", "web_search"],
            "need_local_rag": True,
            "need_web": True,
            "search_queries": {},
        }
        tools = [step["tool"] for step in build_research_plan("DINO 最新官方代码", decision)]
        self.assertEqual(tools, ["local_search", "paper_search", "web_search"])


if __name__ == "__main__":
    unittest.main()
