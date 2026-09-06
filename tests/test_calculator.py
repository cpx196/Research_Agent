import unittest

from tools.calculator import calculator


class CalculatorTests(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        self.assertEqual(calculator("12 * 8"), "96")
        self.assertEqual(calculator("(10 + 2) / 3"), "4")

    def test_rejects_python_code(self) -> None:
        result = calculator("__import__('os').getcwd()")
        self.assertTrue(result.startswith("ToolError: Invalid expression:"))

    def test_large_smoke_case(self) -> None:
        self.assertEqual(calculator("3947 * 8123"), "32061481")


if __name__ == "__main__":
    unittest.main()
