import unittest

from mcp_layer.adapters import NativeToolAdapter, ToolCallResult
from mcp_layer.registry import UnifiedToolRegistry
from tools.calculator import calculator, calculator_schema


class MCPAdapterTests(unittest.TestCase):
    def test_schema_and_trace_fields_are_normalized(self) -> None:
        registry = UnifiedToolRegistry(
            native_registry={"calculator": calculator},
            native_schemas=[calculator_schema],
        )
        schema = registry.schemas[0]
        self.assertEqual(schema["function"]["name"], "calculator")
        result = registry.callables["calculator"](expression="2 + 3")
        self.assertIn("[Tool Source] Native", result)
        self.assertIn("[MCP Tool] calculator", result)
        self.assertIn("[Fallback] None", result)

    def test_native_adapter_keeps_original_tool_result(self) -> None:
        adapter = NativeToolAdapter("calculator", calculator, calculator_schema)
        result = adapter.call_sync({"expression": "4 * 5"})
        self.assertIsInstance(result, ToolCallResult)
        self.assertEqual(result.content, "20")
        self.assertFalse(result.is_error)


if __name__ == "__main__":
    unittest.main()
