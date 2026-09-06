import json
import unittest

from agent.graph import LangGraphResearchAgent
from agent.llm import DemoLLM
from tools.local_search import local_search_schema


def response(content: str):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class ActiveVerifierLLM(DemoLLM):
    def __init__(self, *, inject_error: bool = False) -> None:
        self.inject_error = inject_error

    def chat(self, messages, tools=None):
        system = str(messages[0].get("content", "")) if messages else ""
        if self.inject_error and "Writer node" in system:
            return response(
                "根据 vjepa2.pdf 第 4 页，V-JEPA 2 使用视频预测表征。\n"
                "V-JEPA 2 已完全开源机器人控制训练与推理代码。"
            )
        if "evidence error detector" in system:
            errors = []
            if self.inject_error:
                errors = [{
                    "claim": "V-JEPA 2 已完全开源机器人控制训练与推理代码。",
                    "reason": "当前 Evidence 没有支持完整开源这一结论。",
                }]
            return response(json.dumps({"potential_errors": errors}, ensure_ascii=False))
        if "targeted verification questions" in system:
            return response(json.dumps({
                "verification_questions": ["在本地资料中核查 V-JEPA 2 是否完整开源机器人控制代码？"]
            }, ensure_ascii=False))
        if "final verification judge" in system:
            return response(json.dumps({
                "passed": False,
                "feedback": [{
                    "claim": "V-JEPA 2 已完全开源机器人控制训练与推理代码。",
                    "problem": "补充检索仍未证明训练与推理代码均已开源。",
                    "suggested_action": "targeted_rewrite",
                }],
            }, ensure_ascii=False))
        return super().chat(messages, tools=tools)


def local_result(query: str) -> str:
    return (
        "[Result 1]\nSource: vjepa2.pdf\nPage: 4\nScore: 0.9500\n"
        f"Text: V-JEPA 2 evidence for {query}."
    )


class ActiveVerifierTests(unittest.TestCase):
    def test_supported_answer_passes_without_secondary_research(self) -> None:
        calls: list[str] = []

        def local_search(query: str, top_k: int = 5) -> str:
            del top_k
            calls.append(query)
            return local_result(query)

        agent = LangGraphResearchAgent(
            ActiveVerifierLLM(),
            tools={"local_search": local_search},
            tool_schemas=[local_search_schema],
            verifier_mode="active",
            verbose=False,
        )
        agent.run("根据本地论文解释 V-JEPA 2。")

        self.assertTrue(agent.last_state["verification_passed"])
        self.assertEqual(agent.last_state["potential_errors"], [])
        self.assertEqual(agent.last_state["verification_questions"], [])
        self.assertEqual(agent.last_state["verification_tool_calls"], 0)
        self.assertEqual(len(calls), 1)
        trace = "\n".join(agent.last_state["graph_trace"])
        self.assertIn("[Node] VerificationErrorDetector", trace)
        self.assertIn("[Node] VerificationJudge", trace)
        self.assertNotIn("[Node] AnswerRepair", trace)

    def test_missing_claim_evidence_runs_targeted_research_and_fails(self) -> None:
        calls: list[str] = []

        def local_search(query: str, top_k: int = 5) -> str:
            del top_k
            calls.append(query)
            return local_result(query)

        agent = LangGraphResearchAgent(
            ActiveVerifierLLM(inject_error=True),
            tools={"local_search": local_search},
            tool_schemas=[local_search_schema],
            verifier_mode="active",
            max_iterations=1,
            max_verification_tool_calls=2,
            verbose=False,
        )
        answer = agent.run("根据本地论文分析 V-JEPA 2。")

        self.assertFalse(agent.last_state["verification_passed"])
        self.assertEqual(len(agent.last_state["potential_errors"]), 1)
        self.assertEqual(len(agent.last_state["verification_questions"]), 1)
        self.assertEqual(agent.last_state["verification_tool_calls"], 1)
        self.assertTrue(agent.last_state["verification_evidence"])
        self.assertEqual(agent.last_state["verifier_feedback"][0]["suggested_action"], "targeted_rewrite")
        self.assertEqual(len(calls), 2)
        self.assertIn("最大主动验证轮次", answer)
        self.assertNotIn("V-JEPA 2 已完全开源机器人控制训练与推理代码。", answer)
        self.assertTrue(agent.last_state["claim_audits"])
        self.assertTrue(agent.last_state["claim_audits"][0]["changed"])

    def test_simple_verifier_remains_available(self) -> None:
        agent = LangGraphResearchAgent(DemoLLM(), verifier_mode="simple", verbose=False)
        nodes = set(agent.graph.get_graph().nodes)
        self.assertIn("verifier", nodes)
        self.assertNotIn("verification_error_detector", nodes)


if __name__ == "__main__":
    unittest.main()
