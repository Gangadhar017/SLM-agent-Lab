"""Tool registry: JSON schemas (OpenAI/HF function-calling format) and a single dispatch entry point.

The dispatcher validates arguments and converts every tool exception into a structured ToolResult so the
agent loop never crashes on a bad model output, and the failure taxonomy can see *why* a call failed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import calculator, doc_search, sql_query, unit_convert


class ToolError(ValueError):
    """Raised for invalid arguments or tool execution failures (recorded, never propagated to the agent loop)."""


@dataclass
class ToolResult:
    name: str
    arguments: dict
    ok: bool
    output: Any = None
    error: str | None = None
    latency_s: float = 0.0
    error_kind: str | None = None  # "unknown_tool" | "missing_argument" | "bad_argument" | "execution"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "error_kind": self.error_kind,
            "latency_s": round(self.latency_s, 4),
        }

    def observation(self, max_chars: int = 1500) -> str:
        """What the model sees in the tool-role message."""
        payload = self.output if self.ok else {"error": self.error}
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) > max_chars:
            text = text[: max_chars - 15] + '..."TRUNCATED"}'
        return text


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate an arithmetic expression and return the numeric result. Supports + - * / ** % and "
                "parentheses, and the functions sqrt, abs, round, min, max, log, log10, exp, floor, ceil. "
                'Example arguments: {"expression": "17.5 / 100 * 8432"}'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression, e.g. '(120 + 30) * 2.5'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_convert",
            "description": (
                "Convert a numeric value between units. Length: m, km, cm, mm, mi, yd, ft, in. Mass: kg, g, mg, lb, "
                "oz, t. Temperature: c, f, k. Volume: l, ml, gal, qt, cup. Time: s, min, h, day, week. Speed: m/s, "
                'km/h, mph. Data: b, kb, mb, gb, tb. Example arguments: {"value": 48.3, "from_unit": "mi", '
                '"to_unit": "km"}'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "The numeric value to convert"},
                    "from_unit": {"type": "string", "description": "Unit of the input value, e.g. 'lb'"},
                    "to_unit": {"type": "string", "description": "Unit to convert to, e.g. 'kg'"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": (
                "Run a read-only SQL SELECT query against the company database (SQLite). Tables: "
                + sql_query.SCHEMA_DESCRIPTION
                + '. Example arguments: {"query": "SELECT COUNT(*) FROM employees WHERE department = \'Sales\'"}'
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A single SQL SELECT statement"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "doc_search",
            "description": (
                "Search the internal company documents (HR policies, travel policy, product specifications, "
                "logistics, IT security, reports) and return the most relevant documents with their full text. "
                'Example arguments: {"query": "travel policy meal per diem", "k": 2}'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search for"},
                    "k": {"type": "integer", "description": "Number of documents to return (1-5, default 3)"},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_NAMES: list[str] = [t["function"]["name"] for t in TOOL_SCHEMAS]
_SCHEMA_BY_NAME = {t["function"]["name"]: t["function"]["parameters"] for t in TOOL_SCHEMAS}

_IMPL: dict[str, Callable[..., dict]] = {
    "calculator": lambda expression: calculator.calculate(expression),
    "unit_convert": lambda value, from_unit, to_unit: unit_convert.convert(value, from_unit, to_unit),
    "sql_query": lambda query: sql_query.run_query(query),
    "doc_search": lambda query, k=3: doc_search.search(query, k),
}

_ARG_ALIASES: dict[str, dict[str, str]] = {
    # Small models frequently invent near-miss argument names; we accept the obvious ones and log it.
    "calculator": {"expr": "expression", "input": "expression", "equation": "expression", "formula": "expression"},
    "unit_convert": {"from": "from_unit", "to": "to_unit", "unit_from": "from_unit", "unit_to": "to_unit",
                     "amount": "value", "number": "value", "quantity": "value"},
    "sql_query": {"sql": "query", "statement": "query", "q": "query"},
    "doc_search": {"q": "query", "search": "query", "text": "query", "keywords": "query", "top_k": "k", "n": "k",
                   "limit": "k"},
}


def _coerce(name: str, arguments: dict) -> tuple[dict, list[str]]:
    """Normalise argument names/types. Returns (clean_args, notes) where notes record every leniency applied."""
    notes: list[str] = []
    schema = _SCHEMA_BY_NAME[name]
    props = schema["properties"]
    clean: dict[str, Any] = {}
    for key, val in arguments.items():
        k = key
        if k not in props and k in _ARG_ALIASES.get(name, {}):
            k = _ARG_ALIASES[name][k]
            notes.append(f"renamed argument {key}->{k}")
        if k not in props:
            notes.append(f"ignored unknown argument {key}")
            continue
        expected = props[k]["type"]
        if expected == "number" and isinstance(val, str):
            try:
                val = float(val.replace(",", ""))
                notes.append(f"coerced {k} from string to number")
            except ValueError:
                raise ToolError(f"argument {k!r} must be a number, got {val!r}")
        elif expected == "integer" and isinstance(val, str):
            try:
                val = int(float(val))
                notes.append(f"coerced {k} from string to integer")
            except ValueError:
                raise ToolError(f"argument {k!r} must be an integer, got {val!r}")
        elif expected == "string" and not isinstance(val, str):
            val = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
            notes.append(f"coerced {k} to string")
        clean[k] = val
    missing = [r for r in schema.get("required", []) if r not in clean]
    if missing:
        raise ToolError(f"missing required argument(s): {', '.join(missing)}")
    return clean, notes


def execute_tool(name: str, arguments: dict | None) -> ToolResult:
    """Execute a tool by name. Never raises; all failures are returned as ToolResult(ok=False)."""
    arguments = arguments if isinstance(arguments, dict) else {}
    t0 = time.perf_counter()
    if name not in _IMPL:
        return ToolResult(name, arguments, False, error=f"unknown tool: {name!r}", error_kind="unknown_tool",
                          latency_s=time.perf_counter() - t0)
    try:
        clean, notes = _coerce(name, arguments)
    except ToolError as exc:
        kind = "missing_argument" if "missing required" in str(exc) else "bad_argument"
        return ToolResult(name, arguments, False, error=str(exc), error_kind=kind,
                          latency_s=time.perf_counter() - t0)
    try:
        output = _IMPL[name](**clean)
    except (ValueError, TypeError, ArithmeticError) as exc:  # includes every tool-specific error class
        return ToolResult(name, clean, False, error=str(exc), error_kind="execution",
                          latency_s=time.perf_counter() - t0)
    except Exception as exc:  # defensive: a tool bug must not kill an evaluation run
        return ToolResult(name, clean, False, error=f"{type(exc).__name__}: {exc}", error_kind="execution",
                          latency_s=time.perf_counter() - t0)
    if notes:
        output = dict(output)
        output["_notes"] = notes
    return ToolResult(name, clean, True, output=output, latency_s=time.perf_counter() - t0)
