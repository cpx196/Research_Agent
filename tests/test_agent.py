import json
import unittest

from agent.agent import ResearchAgent


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        return next(self.responses)


def tool_call(name: str, arguments, call_id: str = "call-1"):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            }
        ]
    }


def final(content: str):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class AgentTests(unittest.TestCase):
    def test_returns_direct_answer_without_tool(self) -> None:
        llm = FakeLLM([final("Transformer 是一种神经网络架构。")])
        agent = ResearchAgent(llm, verbose=False)

        self.assertEqual(agent.run("简单解释什么是 Transformer。"), "Transformer 是一种神经网络架构。")
        self.assertEqual(agent.last_run_stats["tool_calls"], 0)
        self.assertEqual(agent.last_run_stats["steps"], 1)

    def test_runs_tool_then_returns_final_answer(self) -> None:
        llm = FakeLLM(
            [
                tool_call("calculator", json.dumps({"expression": "12 * 8"})),
                final("结果是 96。"),
            ]
        )
        agent = ResearchAgent(llm, verbose=False)

        self.assertEqual(agent.run("计算 12 * 8"), "结果是 96。")
        self.assertEqual(agent.last_run_stats["tool_calls"], 1)
        self.assertEqual(agent.last_run_stats["steps"], 2)
        self.assertTrue(
            any(
                message["role"] == "tool" and message["content"] == "96"
                for message in agent.last_messages
            )
        )
        self.assertTrue(llm.calls[0][1])

    def test_handles_invalid_json_and_gives_model_another_step(self) -> None:
        llm = FakeLLM(
            [
                tool_call("calculator", "{expression: 12 * 8}"),
                final("无法解析参数，已安全停止。"),
            ]
        )
        agent = ResearchAgent(llm, verbose=False)

        self.assertEqual(agent.run("计算"), "无法解析参数，已安全停止。")
        self.assertEqual(agent.last_run_stats["tool_errors"], 1)
        self.assertTrue(
            any(
                message["content"] == "ToolError: Invalid JSON arguments"
                for message in agent.last_messages
                if message["role"] == "tool"
            )
        )

    def test_stops_at_max_steps(self) -> None:
        llm = FakeLLM([tool_call("missing_tool", "{}") for _ in range(2)])
        agent = ResearchAgent(llm, max_steps=2, verbose=False)

        self.assertEqual(agent.run("keep going"), "Agent stopped because max_steps was reached.")
        self.assertEqual(agent.last_run_stats["steps"], 2)
        self.assertEqual(agent.last_run_stats["tool_errors"], 2)


if __name__ == "__main__":
    unittest.main()
