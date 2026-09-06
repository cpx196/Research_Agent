import unittest

from agent.agent import ResearchAgent
from agent.context import ContextConfig
from agent.graph import LangGraphResearchAgent
from agent.llm import DemoLLM
from tools.calculator import calculator, calculator_schema
from tools.local_search import local_search_schema
from tools.paper_search import paper_search_schema
from tools.web_search import web_search_schema


def fake_local_search(query: str, top_k: int = 5) -> str:
    del top_k
    return (
        "[RAG] Query embedding generated\n"
        "[RAG] Top-K retrieval complete\n\n"
        "[Result 1]\n"
        "Source: fake-jepa.pdf\n"
        "Page: 7\n"
        "Score: 0.9200\n"
        f"Text: Evidence for {query}."
    )


def fake_long_local_search(query: str, top_k: int = 5) -> str:
    del top_k
    return (
        "[RAG] Query embedding generated\n[RAG] Top-K retrieval complete\n\n"
        "[Result 1]\nSource: long-jepa.pdf\nPage: 9\nScore: 0.9100\n"
        f"Text: {query}; " + "long supporting fact " * 1200
    )


def fake_web_search(query: str, max_results: int = 5) -> str:
    del max_results
    return f"Title: Example\nURL: https://example.test/result\nSnippet: Web evidence for {query}."


def fake_paper_search(query: str, max_results: int = 5) -> str:
    del max_results
    return (
        "Title: A Path Towards Autonomous Machine Intelligence\n"
        "Authors: Yann LeCun\n"
        "Published: 2022-06-27\n"
        "arXiv: https://arxiv.org/abs/2202.05861\n"
        f"Abstract: JEPA Joint Embedding Predictive Architecture evidence for {query}."
    )


class RouteAwareLLM:
    """Small model double that records the schemas visible to ReAct."""

    def __init__(self, *, request_disallowed_tool: bool = False) -> None:
        self.request_disallowed_tool = request_disallowed_tool
        self.tools_seen: list[list[str]] = []

    def chat(self, messages, tools=None):
        names = [schema["function"]["name"] for schema in (tools or [])]
        self.tools_seen.append(names)
        if self.request_disallowed_tool and len(self.tools_seen) == 1:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "unauthorized-local",
                            "type": "function",
                            "function": {
                                "name": "local_search",
                                "arguments": '{"query": "must not run"}',
                            },
                        }],
                    }
                }]
            }
        if self.request_disallowed_tool and len(self.tools_seen) > 1:
            return {"choices": [{"message": {"role": "assistant", "content": "已安全停止。"}}]}
        if not any(message.get("role") == "tool" for message in messages):
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "web-call",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query": "danking"}',
                            },
                        }],
                    }
                }]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "联网结果已处理。"}}]}


class GraphTests(unittest.TestCase):
    def test_graph_compiles_with_explicit_nodes(self) -> None:
        agent = LangGraphResearchAgent(DemoLLM(), verbose=False)
        node_names = set(agent.graph.get_graph().nodes)
        self.assertTrue({"planner", "researcher", "tools", "evidence", "writer", "verifier"} <= node_names)

    def test_graph_trace_contains_final_answer(self) -> None:
        agent = LangGraphResearchAgent(DemoLLM(), verbose=False)
        answer = agent.run("你好")
        self.assertIn("[Final Answer]", agent.trace.lines)
        self.assertEqual(agent.trace.lines[-1], answer)
        self.assertIn("Research Agent", answer)
        self.assertIn("[Node] DirectAnswer", agent.last_state["graph_trace"])

    def test_inmemory_checkpoint_contains_final_state(self) -> None:
        agent = LangGraphResearchAgent(DemoLLM(), verbose=False)
        agent.run("什么是 Transformer？", thread_id="checkpoint-test")
        snapshot = agent.graph.get_state({"configurable": {"thread_id": "checkpoint-test"}})
        self.assertEqual(snapshot.values["query"], "什么是 Transformer？")
        self.assertEqual(snapshot.values["draft"], agent.last_state["draft"])

    def test_calculator_runs_through_langgraph_toolnode(self) -> None:
        agent = LangGraphResearchAgent(
            DemoLLM(),
            tools={"calculator": calculator},
            tool_schemas=[calculator_schema],
            verbose=False,
        )
        answer = agent.run("计算 12 * 8")
        self.assertIn("96", answer)
        self.assertEqual(agent.last_state["evidence"][0]["content"], "96")
        self.assertTrue(agent.last_state["verification_passed"])

    def test_react_only_sees_tools_allowed_by_router(self) -> None:
        llm = RouteAwareLLM()
        local_calls: list[str] = []

        def should_not_run_local(query: str, top_k: int = 5) -> str:
            del top_k
            local_calls.append(query)
            return "This local tool must not be called."

        agent = LangGraphResearchAgent(
            llm,
            tools={"web_search": fake_web_search, "local_search": should_not_run_local},
            tool_schemas=[web_search_schema, local_search_schema],
            verbose=False,
        )
        agent.run("danking是谁")

        self.assertEqual(agent.last_state["route_decision"]["allowed_tools"], ["web_search"])
        self.assertEqual(llm.tools_seen, [["web_search"], ["web_search"]])
        self.assertEqual(local_calls, [])

    def test_execution_guard_blocks_a_stale_disallowed_tool_call(self) -> None:
        llm = RouteAwareLLM(request_disallowed_tool=True)
        local_calls: list[str] = []

        def should_not_run_local(query: str, top_k: int = 5) -> str:
            del top_k
            local_calls.append(query)
            return "This local tool must not be called."

        agent = LangGraphResearchAgent(
            llm,
            tools={"web_search": fake_web_search, "local_search": should_not_run_local},
            tool_schemas=[web_search_schema, local_search_schema],
            verbose=False,
        )
        answer = agent.run("danking是谁")

        self.assertEqual(answer, "已安全停止。")
        self.assertEqual(local_calls, [])
        self.assertIn("[Tool Guard] Blocked disallowed tool: local_search", agent.last_state["graph_trace"])

    def test_research_collects_explicit_evidence(self) -> None:
        agent = LangGraphResearchAgent(
            DemoLLM(),
            tools={"local_search": fake_local_search},
            tool_schemas=[local_search_schema],
            verbose=False,
        )
        answer = agent.run("根据本地论文解释 JEPA。")
        self.assertIn("fake-jepa.pdf", answer)
        self.assertEqual(agent.last_state["evidence"][0]["source"], "fake-jepa.pdf")
        self.assertEqual(agent.last_state["evidence"][0]["page"], 7)

    def test_context_graph_records_node_specific_metrics(self) -> None:
        agent = LangGraphResearchAgent(
            DemoLLM(),
            tools={"local_search": fake_local_search},
            tool_schemas=[local_search_schema],
            context_enabled=True,
            verbose=False,
        )
        agent.run("根据本地论文解释 JEPA。")
        stats = agent.last_state["context_stats"]
        self.assertTrue(stats["history"])
        self.assertEqual(stats["nodes"]["writer"]["context_budget"], 8000)
        self.assertIn("[Context Manager] Writer context", "\n".join(agent.last_state["graph_trace"]))

    def test_context_graph_compresses_long_tool_result(self) -> None:
        agent = LangGraphResearchAgent(
            DemoLLM(),
            tools={"local_search": fake_long_local_search},
            tool_schemas=[local_search_schema],
            context_enabled=True,
            context_config=ContextConfig(tool_result_compress_threshold=20, max_evidence_chars=300),
            verbose=False,
        )
        answer = agent.run("根据本地论文解释 JEPA。")
        self.assertIn("long-jepa.pdf", answer)
        self.assertTrue(agent.last_state["context_stats"]["tool_results"][0]["compressed"])

    def test_verifier_fail_routes_to_answer_repair(self) -> None:
        agent = LangGraphResearchAgent(
            DemoLLM(),
            tools={"local_search": fake_local_search},
            tool_schemas=[local_search_schema],
            max_iterations=2,
            forced_verifier_failures=1,
            verbose=False,
        )
        agent.run("根据本地论文解释 JEPA。")
        trace = "\n".join(agent.last_state["graph_trace"])
        self.assertIn("[Conditional Edge] Verifier -> AnswerRepair", trace)
        self.assertIn("[Node] AnswerRepair", trace)
        self.assertIn("[Conditional Edge] Verifier -> END", trace)
        self.assertEqual(agent.last_state["iteration"], 2)

    def test_max_iterations_stops_a_failing_workflow(self) -> None:
        agent = LangGraphResearchAgent(
            DemoLLM(),
            tools={"local_search": fake_local_search},
            tool_schemas=[local_search_schema],
            max_iterations=2,
            forced_verifier_failures=10,
            verbose=False,
        )
        answer = agent.run("根据本地论文解释 JEPA。")
        self.assertEqual(agent.last_state["iteration"], 2)
        self.assertFalse(agent.last_state["verification_passed"])
        self.assertIn("最大修复轮次", answer)

    def test_contextual_paper_request_uses_resolved_topic(self) -> None:
        agent = LangGraphResearchAgent(
            DemoLLM(),
            tools={"paper_search": fake_paper_search, "local_search": fake_local_search},
            tool_schemas=[paper_search_schema, local_search_schema],
            context_enabled=True,
            verbose=False,
        )
        answer = agent.run(
            "找点论文看看呢",
            conversation_history=[
                {"role": "user", "content": "我想学习Jepa相关的信息"},
                {"role": "assistant", "content": "下面给你一份 JEPA 学习路线。"},
            ],
            session_preferences={"response_language": "zh-CN"},
        )

        self.assertEqual(agent.last_state["standalone_query"], "查找并推荐 JEPA 相关核心论文")
        self.assertEqual(agent.last_state["resolved_topic"], "JEPA")
        self.assertEqual(
            [step["tool"] for step in agent.last_state["plan"]],
            ["local_search", "paper_search"],
        )
        self.assertTrue(agent.last_state["relevance_checks"][0]["passed"])
        self.assertEqual(len(agent.last_state["accepted_evidence"]), 2)
        self.assertIn("https://arxiv.org/abs/2202.05861", answer)

    def test_irrelevant_paper_result_does_not_suppress_planned_local_rag(self) -> None:
        calls: list[tuple[str, str]] = []

        def irrelevant_paper_search(query: str, max_results: int = 5) -> str:
            del max_results
            calls.append(("paper_search", query))
            return (
                "Title: 找字的笔顺\nAuthors: Example\nPublished: 2020-01-01\n"
                "arXiv: https://example.test/hanzi\nAbstract: 找字共有七画，部首是扌。"
            )

        def relevant_local_search(query: str, top_k: int = 5) -> str:
            calls.append(("local_search", query))
            return fake_local_search(query, top_k)

        agent = LangGraphResearchAgent(
            DemoLLM(),
            tools={"paper_search": irrelevant_paper_search, "local_search": relevant_local_search},
            tool_schemas=[paper_search_schema, local_search_schema],
            verbose=False,
        )
        agent.run(
            "找点论文看看呢",
            conversation_history=[{"role": "user", "content": "我想学习 JEPA 相关的信息"}],
        )

        self.assertEqual([name for name, _ in calls], ["local_search", "paper_search"])
        self.assertEqual(agent.last_state["retrieval_retry_count"], 0)
        self.assertEqual([item["passed"] for item in agent.last_state["relevance_checks"]], [True, False])
        self.assertEqual(len(agent.last_state["evidence"]), 2)
        self.assertEqual(len(agent.last_state["accepted_evidence"]), 1)
        self.assertEqual(agent.last_state["accepted_evidence"][0]["source_type"], "local_search")

    def test_legacy_agent_remains_available(self) -> None:
        agent = ResearchAgent(DemoLLM(), verbose=False)
        self.assertEqual(agent.run("计算 12 * 8"), "计算结果是：32061481")


if __name__ == "__main__":
    unittest.main()
