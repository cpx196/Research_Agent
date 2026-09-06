import unittest

from mcp_layer.adapters import MCPToolAdapter, NativeToolAdapter
from mcp_layer.config import MCPServerConfig
from mcp_layer.manager import MCPClientManager
from tools.calculator import calculator, calculator_schema


class MCPFallbackTests(unittest.TestCase):
    def test_mcp_failure_falls_back_to_native_calculator(self) -> None:
        config = MCPServerConfig(
            name="missing-server",
            command="definitely-not-a-real-mcp-command",
            timeout_seconds=1,
        )
        adapter = MCPToolAdapter(
            MCPClientManager([config]),
            config,
            calculator_schema,
            fallback=NativeToolAdapter("calculator", calculator, calculator_schema),
        )
        result = adapter.call_sync({"expression": "9 + 1"})
        self.assertEqual(result.content, "10")
        self.assertTrue(result.fallback)
        self.assertFalse(result.is_error)


if __name__ == "__main__":
    unittest.main()
