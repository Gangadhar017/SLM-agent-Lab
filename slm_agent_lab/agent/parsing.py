"""Robust tool-call parsing.

Small models emit tool calls in many dialects. We try, in order:
  1. <tool_call>{...}</tool_call> blocks (Hermes / Qwen2.5 / Granite 4.0), including unterminated blocks
  2. Granite 3.x style  <|tool_call|>[{...}]
  3. fenced ```json blocks containing {"name": ..., "arguments": {...}}
  4. bare JSON objects with name + arguments/parameters anywhere in the text
  5. python-style  tool_name(arg="value", ...)  for registered tool names only

Every call records which dialect it came from and whether the JSON needed repair, so the evaluation can report
*how* a model talks to tools, not just whether it did. Fragments that look like calls but cannot be parsed are
returned as `malformed`, which the failure taxonomy turns into a first-class failure category.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

from ..tools.registry import TOOL_NAMES, TOOL_SCHEMAS

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
_GRANITE_TAG = re.compile(r"<\|tool_call\|>(.*?)(?=<\|tool_call\|>|$)", re.DOTALL)
_FENCE = re.compile(r"```(?:json|python|tool_call)?\s*(.*?)```", re.DOTALL)
_PY_CALL = re.compile(r"\b(" + "|".join(re.escape(n) for n in TOOL_NAMES) + r")\s*\((.*?)\)(?=\s|$|[.,;])", re.DOTALL)
_LOOKS_LIKE_CALL = re.compile(r'"name"\s*:|<tool_call|<\|tool_call\|>|"arguments"\s*:|"function"\s*:', re.IGNORECASE)

_REQUIRED_ORDER = {t["function"]["name"]: list(t["function"]["parameters"]["properties"].keys()) for t in TOOL_SCHEMAS}


@dataclass
class ParsedCall:
    name: str
    arguments: dict
    raw: str
    source: str              # tool_call_tag | granite_tag | fenced | bare_json | python_call
    repaired: bool = False   # JSON needed lenient repair (single quotes, python literals, trailing commas)

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments, "source": self.source, "repaired": self.repaired,
                "raw": self.raw[:500]}


@dataclass
class ParseResult:
    calls: list[ParsedCall] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)   # fragments that looked like calls but could not be parsed
    final_text: str = ""                                  # text outside of any tool-call construct
    text_after_calls: str = ""                            # text generated *after* the last call (fabricated obs?)

    @property
    def has_calls(self) -> bool:
        return bool(self.calls)


# --------------------------------------------------------------------------------------------------------------
def _loads_lenient(s: str):
    """json.loads with a fallback for the usual small-model sins. Returns (obj, repaired) or (None, False)."""
    s = s.strip()
    if not s:
        return None, False
    try:
        return json.loads(s), False
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    try:
        return json.loads(fixed), True
    except json.JSONDecodeError:
        pass
    try:  # python dict literal: single quotes, True/False/None
        obj = ast.literal_eval(fixed)
        if isinstance(obj, (dict, list)):
            return obj, True
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass
    return None, False


def _find_json_spans(text: str) -> list[tuple[int, int]]:
    """Balanced-brace scan for top-level {...} objects (string-aware)."""
    spans = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, in_str, esc, j = 0, False, False, i
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        spans.append((i, j + 1))
                        break
            j += 1
        i = j + 1 if depth == 0 else i + 1
    return spans


def _normalise_call(obj, raw: str, source: str, repaired: bool) -> list[ParsedCall] | None:
    """Turn a decoded object into ParsedCall(s). Accepts several key spellings. None if it is not a call."""
    if isinstance(obj, list):
        calls = []
        for item in obj:
            c = _normalise_call(item, raw, source, repaired)
            if c:
                calls.extend(c)
        return calls or None
    if not isinstance(obj, dict):
        return None
    if "function" in obj and isinstance(obj["function"], dict):  # OpenAI wrapper {"type":"function","function":{...}}
        obj = obj["function"]
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name") or obj.get("function_name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    if name.startswith("functions."):
        name = name[len("functions."):]
    args = obj.get("arguments", obj.get("parameters", obj.get("args", obj.get("input", {}))))
    if isinstance(args, str):
        parsed, rep = _loads_lenient(args)
        if isinstance(parsed, dict):
            args, repaired = parsed, repaired or rep
        else:
            # a bare string argument for a single-parameter tool: {"name": "calculator", "arguments": "2+2"}
            order = _REQUIRED_ORDER.get(name)
            args = {order[0]: args} if order else {"value": args}
            repaired = True
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return [ParsedCall(name=name, arguments=args, raw=raw, source=source, repaired=repaired)]


def _parse_python_call(name: str, argtext: str, raw: str) -> ParsedCall | None:
    try:
        node = ast.parse(f"f({argtext})", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call):
        return None
    args: dict = {}
    order = _REQUIRED_ORDER.get(name, [])
    try:
        for i, a in enumerate(node.args):
            if i < len(order):
                args[order[i]] = ast.literal_eval(a)
        for kw in node.keywords:
            if kw.arg:
                args[kw.arg] = ast.literal_eval(kw.value)
    except (ValueError, SyntaxError):
        return None
    if not args:
        return None
    return ParsedCall(name=name, arguments=args, raw=raw, source="python_call", repaired=False)


def parse_tool_calls(text: str) -> ParseResult:
    result = ParseResult()
    if not text:
        return result
    consumed: list[tuple[int, int]] = []
    last_call_end = -1

    def take(start: int, end: int) -> None:
        nonlocal last_call_end
        consumed.append((start, end))
        last_call_end = max(last_call_end, end)

    # 1. <tool_call> blocks
    for m in _TOOL_CALL_BLOCK.finditer(text):
        body = m.group(1).strip()
        obj, repaired = _loads_lenient(body)
        calls = _normalise_call(obj, body, "tool_call_tag", repaired) if obj is not None else None
        if not calls:  # maybe several objects / junk around an object
            calls = []
            for s, e in _find_json_spans(body):
                obj, repaired = _loads_lenient(body[s:e])
                c = _normalise_call(obj, body[s:e], "tool_call_tag", repaired) if obj is not None else None
                if c:
                    calls.extend(c)
        if calls:
            result.calls.extend(calls)
        else:
            result.malformed.append(m.group(0)[:300])
        take(m.start(), m.end())

    # 2. Granite 3.x tag
    for m in _GRANITE_TAG.finditer(text):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        body = m.group(1).strip()
        obj, repaired = _loads_lenient(body)
        calls = _normalise_call(obj, body, "granite_tag", repaired) if obj is not None else None
        if not calls:
            calls = []
            for s, e in _find_json_spans(body):
                obj, repaired = _loads_lenient(body[s:e])
                c = _normalise_call(obj, body[s:e], "granite_tag", repaired) if obj is not None else None
                if c:
                    calls.extend(c)
        if calls:
            result.calls.extend(calls)
        else:
            result.malformed.append(m.group(0)[:300])
        take(m.start(), m.end())

    # 3. fenced blocks
    for m in _FENCE.finditer(text):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        body = m.group(1).strip()
        found = False
        for s, e in _find_json_spans(body):
            obj, repaired = _loads_lenient(body[s:e])
            c = _normalise_call(obj, body[s:e], "fenced", repaired) if obj is not None else None
            if c:
                result.calls.extend(c)
                found = True
        if found:
            take(m.start(), m.end())
        elif _LOOKS_LIKE_CALL.search(body):
            result.malformed.append(m.group(0)[:300])
            take(m.start(), m.end())

    # 4. bare JSON objects
    for s, e in _find_json_spans(text):
        if any(cs <= s < ce for cs, ce in consumed):
            continue
        obj, repaired = _loads_lenient(text[s:e])
        c = _normalise_call(obj, text[s:e], "bare_json", repaired) if obj is not None else None
        if c:
            result.calls.extend(c)
            take(s, e)
        elif _LOOKS_LIKE_CALL.search(text[s:e]):
            result.malformed.append(text[s:e][:300])
            take(s, e)

    # 5. python-style calls
    for m in _PY_CALL.finditer(text):
        if any(cs <= m.start() < ce for cs, ce in consumed):
            continue
        c = _parse_python_call(m.group(1), m.group(2), m.group(0))
        if c:
            result.calls.append(c)
            take(m.start(), m.end())

    # leftover text
    if consumed:
        pieces, pos = [], 0
        for s, e in sorted(consumed):
            if s > pos:
                pieces.append(text[pos:s])
            pos = max(pos, e)
        pieces.append(text[pos:])
        result.final_text = re.sub(r"\n{3,}", "\n\n", "".join(pieces)).strip()
        result.text_after_calls = text[last_call_end:].strip() if last_call_end >= 0 else ""
    else:
        result.final_text = text.strip()
        if _LOOKS_LIKE_CALL.search(text) and not result.calls:
            result.malformed.append(text[:300])
    return result
