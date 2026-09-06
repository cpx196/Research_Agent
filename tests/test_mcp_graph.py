import unittest

from agent.graph import MCPResearchAgent
from agent.llm import DemoLLM


class MCPGraphTests(unittest.TestCase):
    def test_graph_mcp_executes_discovered_calculator(self) -> None:
        agent = MCPResearchAgent(DemoLLM(), verbose=False)
        answer = agent.run("计算 12 * 8")
        self.assertIn("96", answer)
        self.assertIn("calculator", agent.last_run_stats["mcp_tools"])
        self.assertEqual(agent.last_run_stats["tool_sources"], ["MCP"])


if __name__ == "__main__":
    unittest.main()
