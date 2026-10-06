"""Safe arithmetic evaluator (AST-based, no eval())."""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
}
_CONSTS = {"pi": math.pi, "e": math.e}

_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


class CalculatorError(ValueError):
    pass


def _normalize(expression: str) -> str:
    expr = expression.strip()
    expr = expr.strip("`").strip()
    if expr.startswith("="):
        expr = expr[1:]
    expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    expr = _THOUSANDS_SEP.sub("", expr)  # 8,432 -> 8432 (but keeps round(x, 2))
    return expr.strip()


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"unsupported literal: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise CalculatorError("exponent too large")
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise CalculatorError("division by zero") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        if node.keywords:
            raise CalculatorError("keyword arguments are not supported")
        args = [_eval(a) for a in node.args]
        try:
            return _FUNCS[node.func.id](*args)
        except (ValueError, TypeError) as exc:
            raise CalculatorError(f"{node.func.id}: {exc}") from exc
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise CalculatorError(f"unsupported syntax: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """Evaluate an arithmetic expression. Returns {"expression": ..., "result": number}."""
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("expression must be a non-empty string")
    expr = _normalize(expression)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"could not parse expression {expression!r}: {exc.msg}") from exc
    value = _eval(tree)
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            raise CalculatorError("result is not finite")
        value = round(value, 10)
        if value.is_integer() and abs(value) < 1e15:
            value = int(value)
    return {"expression": expr, "result": value}
