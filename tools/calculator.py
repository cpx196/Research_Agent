"""Safe basic arithmetic tool."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any


MAX_EXPRESSION_LENGTH = 200
MAX_ABS_RESULT = 10**100
MAX_POWER = 100

_BINARY_OPERATORS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ValueError):
    """Raised for unsupported or unsafe arithmetic."""


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool) or not math.isfinite(float(node.value)):
            raise CalculatorError("only finite numeric constants are allowed")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        result = _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
        return _check_result(result)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER:
            raise CalculatorError(f"exponent must be between -{MAX_POWER} and {MAX_POWER}")
        try:
            result = _BINARY_OPERATORS[type(node.op)](left, right)
        except (ArithmeticError, OverflowError) as exc:
            raise CalculatorError(str(exc)) from exc
        return _check_result(result)
    raise CalculatorError("only basic arithmetic operators are allowed")


def _check_result(result: int | float) -> int | float:
    if isinstance(result, float) and not math.isfinite(result):
        raise CalculatorError("result is not finite")
    if abs(result) > MAX_ABS_RESULT:
        raise CalculatorError("result is too large")
    return result


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression without executing Python code."""

    if not isinstance(expression, str) or not expression.strip():
        return "ToolError: expression must be a non-empty string"
    expression = expression.strip()
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return "ToolError: expression is too long"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree)
    except (SyntaxError, CalculatorError) as exc:
        return f"ToolError: Invalid expression: {exc}"

    if isinstance(result, int):
        return str(result)
    return format(result, ".15g")


calculator_schema = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a basic mathematical expression safely.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}
