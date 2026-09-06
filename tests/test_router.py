import json
import unittest

from agent.graph.router import HybridRouter, validate_router_output


def response(content: str):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class RouterLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        return response(self.content)


class RouterTests(unittest.TestCase):
    def test_comparison_query_preserves_both_model_topics(self) -> None:
        decision = HybridRouter(
            RouterLLM("not JSON"), ["local_search", "paper_search"]
        ).classify("DINOv3 和 V-JEPA 2 哪个更好？")
        self.assertEqual(decision.topic, "DINOv3 vs V-JEPA 2")
        self.assertIn("DINOv3", decision.search_queries["local_search"])
        self.assertIn("V-JEPA 2", decision.search_queries["local_search"])

    def test_obvious_direct_route_does_not_call_llm(self) -> None:
        llm = RouterLLM("not JSON")
        decision = HybridRouter(llm, ["web_search"]).classify("你好")
        self.assertEqual(decision.route, "direct")
        self.assertEqual(decision.source, "rules")
        self.assertFalse(llm.calls)

    def test_llm_route_is_structured_and_sees_follow_up_history(self) -> None:
        payload = {
            "route": "react",
            "complexity": "tool",
            "need_planning": False,
            "need_web": True,
            "need_local_rag": False,
            "need_verification": False,
            "suggested_workers": 1,
            "confidence": 0.83,
        }
        llm = RouterLLM(json.dumps(payload))
        decision = HybridRouter(llm, ["web_search"]).classify(
            "danking",
            conversation_history=[{"role": "user", "content": "你知道danking是谁吗"}],
        )
        self.assertEqual(decision.route, "react")
        self.assertEqual(decision.source, "llm")
        self.assertTrue(decision.need_web)
        self.assertEqual(decision.allowed_tools, ("web_search",))
        self.assertIn("你知道danking是谁吗", llm.calls[0][0][1]["content"])

    def test_invalid_llm_output_falls_back_to_react(self) -> None:
        decision = HybridRouter(RouterLLM("I cannot classify this"), ["web_search"]).classify("unknown entity")
        self.assertEqual(decision.route, "react")
        self.assertEqual(decision.source, "fallback")
        self.assertEqual(decision.confidence, 0.2)

    def test_validation_repairs_direct_route_with_external_requirement(self) -> None:
        decision = validate_router_output({
            "route": "direct",
            "complexity": "simple",
            "need_web": True,
            "confidence": "0.7",
        })
        self.assertIsNotNone(decision)
        self.assertEqual(decision.route, "react")
        self.assertEqual(decision.suggested_workers, 1)

    def test_simple_numeric_comparison_stays_direct(self) -> None:
        decision = HybridRouter(RouterLLM("not JSON"), ["calculator"]).classify("比较一下 2 和 3 哪个大")
        self.assertEqual(decision.route, "direct")

    def test_research_route_always_keeps_planning_and_verification(self) -> None:
        decision = validate_router_output({
            "route": "research",
            "complexity": "simple",
            "need_planning": False,
            "need_verification": False,
            "suggested_workers": 0,
        })
        self.assertIsNotNone(decision)
        self.assertTrue(decision.need_planning)
        self.assertTrue(decision.need_verification)
        self.assertGreaterEqual(decision.suggested_workers, 1)

    def test_allowlist_follows_router_flags(self) -> None:
        decision = HybridRouter(
            RouterLLM("not JSON"),
            ["web_search", "local_search", "calculator"],
        ).classify("danking是谁")
        self.assertEqual(decision.allowed_tools, ("web_search",))

        decision = HybridRouter(
            RouterLLM("not JSON"),
            ["web_search", "local_search", "calculator"],
        ).classify("根据本地论文解释 JEPA")
        self.assertEqual(decision.allowed_tools, ("local_search",))

    def test_follow_up_paper_request_is_resolved_from_history(self) -> None:
        decision = HybridRouter(
            RouterLLM("not used"),
            ["paper_search", "local_search", "web_search"],
        ).classify(
            "找点论文看看呢",
            conversation_history=[
                {"role": "user", "content": "我想学习Jepa相关的信息"},
                {"role": "assistant", "content": "下面给你一份 JEPA 学习路线。"},
            ],
        )
        self.assertEqual(decision.standalone_query, "查找并推荐 JEPA 相关核心论文")
        self.assertEqual(decision.topic, "JEPA")
        self.assertEqual(decision.intent, "paper_search")
        self.assertEqual(decision.primary_tool, "local_search")
        self.assertEqual(decision.allowed_tools, ("local_search", "paper_search"))
        self.assertEqual(decision.search_queries["paper_search"], "JEPA Joint Embedding Predictive Architecture")

    def test_known_local_topic_definition_enables_local_and_paper_sources(self) -> None:
        decision = HybridRouter(
            RouterLLM("not used"),
            ["paper_search", "local_search", "web_search"],
        ).classify("JEPA是什么")
        self.assertEqual(decision.route, "research")
        self.assertEqual(decision.primary_tool, "local_search")
        self.assertEqual(decision.allowed_tools, ("local_search", "paper_search"))

    def test_current_known_topic_adds_web_without_dropping_local_sources(self) -> None:
        decision = HybridRouter(
            RouterLLM("not used"),
            ["paper_search", "local_search", "web_search"],
        ).classify("DINO 最新官方代码")
        self.assertEqual(decision.route, "research")
        self.assertEqual(
            decision.allowed_tools,
            ("local_search", "paper_search", "web_search"),
        )


if __name__ == "__main__":
    unittest.main()
