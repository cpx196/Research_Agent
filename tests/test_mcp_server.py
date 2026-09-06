import unittest

from mcp_layer.config import default_research_server_config
from mcp_layer.servers.research_server import mcp


class MCPServerTests(unittest.TestCase):
    def test_server_entrypoint_is_stdio_and_whitelisted(self) -> None:
        config = default_research_server_config()
        self.assertEqual(config.transport, "stdio")
        self.assertTrue(config.args[0].endswith("mcp_layer/servers/research_server.py"))
        self.assertEqual(config.name, "research-mcp")
        self.assertIsNotNone(mcp)


if __name__ == "__main__":
    unittest.main()
