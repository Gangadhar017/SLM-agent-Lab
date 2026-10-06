"""Answer scoring. Deterministic, tolerant of formatting, strict about values.

Numbers: the *last* number in the final-answer segment is the strict match; any number in the text is the lenient
match. Both are recorded so the report can show strict and lenient accuracy side by side.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_FINAL_MARK = re.compile(r"final answer\s*(?:is|:)?", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(no|not|cannot|can't|unable|isn't|doesn't|does not|did not|couldn't|could not|unavailable|not available|"
    r"not found|no information|no record|no such|no data|no employee|none|empty|missing|does not exist|doesn't exist|"
    r"not (?:contain|include|mention|specify|provide)|not listed|not (?:in|within) the)\b",
    re.IGNORECASE,
)


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            continue
    return out


def final_segment(text: str) -> str:
    """Text after the last 'Final answer' marker if present, else the whole text."""
    marks = list(_FINAL_MARK.finditer(text or ""))
    return text[marks[-1].end():] if marks else (text or "")


def _num_match(candidate: float, expected: float, abs_tol: float) -> bool:
    return abs(candidate - expected) <= abs_tol


def _string_match(answer: str, candidates: list[str]) -> bool:
    low = re.sub(r"\s+", " ", (answer or "").lower())
    for c in candidates:
        c = c.lower().strip()
        if not c:
            continue
        if len(c) <= 3 or c.isdigit():
            if re.search(r"(?<![a-z0-9])" + re.escape(c) + r"(?![a-z0-9])", low):
                return True
        elif c in low:
            return True
    return False


def score_answer(expected: dict, final_answer: str) -> dict:
    """Returns {"correct": bool, "strict": bool, "match_mode": str, "extracted": ..., "note": str}."""
    kind = expected.get("type")
    text = final_answer or ""
    if kind == "number":
        target, tol = float(expected["value"]), float(expected.get("abs_tol", 0.01))
        seg_nums = extract_numbers(final_segment(text))
        all_nums = extract_numbers(text)
        if seg_nums and _num_match(seg_nums[-1], target, tol):
            return {"correct": True, "strict": True, "match_mode": "last_number", "extracted": seg_nums[-1], "note": ""}
        if all_nums and _num_match(all_nums[-1], target, tol):
            return {"correct": True, "strict": True, "match_mode": "last_number", "extracted": all_nums[-1], "note": ""}
        for n in all_nums:
            if _num_match(n, target, tol):
                return {"correct": True, "strict": False, "match_mode": "any_number", "extracted": n,
                        "note": "expected value present but not the final number"}
        return {"correct": False, "strict": False, "match_mode": "none",
                "extracted": all_nums[-1] if all_nums else None, "note": "no matching number"}
    if kind == "string":
        ok = _string_match(final_segment(text), expected["any_of"]) or _string_match(text, expected["any_of"])
        return {"correct": ok, "strict": ok, "match_mode": "string" if ok else "none", "extracted": text[:120].strip(),
                "note": ""}
    if kind == "unanswerable":
        ok = bool(_NEGATION.search(text))
        return {"correct": ok, "strict": ok, "match_mode": "negation" if ok else "none", "extracted": text[:120].strip(),
                "note": "heuristic: answer must state that the information is unavailable"}
    raise ValueError(f"unknown expectation type: {kind}")
