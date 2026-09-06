import asyncio
import unittest

from mcp_layer.config import default_research_server_config
from mcp_layer.manager import MCPClientManager


class MCPClientTests(unittest.TestCase):
    def test_stdio_discovery_tool_resource_and_prompt(self) -> None:
        async def scenario() -> None:
            manager = MCPClientManager([default_research_server_config()])
            self.assertEqual((await manager.connect())["research-mcp"], "connected")
            tools = await manager.list_tools()
            self.assertEqual({tool["function"]["name"] for tool in tools}, {"calculator", "paper_search"})
            result = await manager.call_tool("calculator", {"expression": "6 * 7"})
            self.assertEqual(result.content, "42")
            resources = await manager.list_resources()
            self.assertTrue(any(str(getattr(item, "uri", "")).startswith("research://") for item in resources))
            self.assertIn("vjepa2.pdf", await manager.read_resource("research-mcp", "research://manifest"))
            prompts = await manager.list_prompts()
            self.assertTrue(any(getattr(item, "name", "") == "paper_research" for item in prompts))
            self.assertIn("JEPA", await manager.get_prompt("research-mcp", "paper_research", {"topic": "JEPA"}))
            await manager.close()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
