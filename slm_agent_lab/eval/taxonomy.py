"""Failure taxonomy: turns a (task, trajectory, score) triple into one primary label plus secondary flags.

The primary label is assigned by a fixed decision tree so that every failure gets exactly one, mutually exclusive
cause. The order matters: syntactic failures (could not even talk to a tool) are checked before planning failures
(talked to the wrong tool) before execution failures (right tool, wrong use) before reasoning failures (right
tool, right result, wrong final answer).

A distinctive check is `answer_in_observations`: if the expected answer already appears in a tool output, then
the agent *had* the information and failed at synthesis, which is a different bug from never retrieving it.
"""

from __future__ import annotations

import json
import re

from .scoring import _NUM, _string_match

FAILURE_LABELS = [
    "success",
    "no_final_answer",             # ran out of steps / loop / truncated with nothing to score
    "malformed_tool_call",         # tried to call a tool but no call could be parsed
    "hallucinated_tool",           # called a tool that does not exist
    "invalid_arguments",           # last tool call rejected (bad/missing args, SQL error, unknown unit...)
    "no_tool_call",                # task needed tools, model answered from memory
    "wrong_tool",                  # used only tools unrelated to the task
    "incomplete_chain",            # multi-step task: some required tools never called
    "semantically_wrong_call",     # right tools, calls executed, but never retrieved the needed value
    "wrong_answer_after_correct_tools",  # the needed value was in a tool output, final answer still wrong
    "unnecessary_tool_then_wrong", # no-tool task, model used tools and got it wrong
    "knowledge_error",             # no-tool task, answered directly, wrong
    "fabricated_answer",           # unanswerable task, model asserted an answer
]


def _numbers_in(obj) -> list[float]:
    text = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    out = []
    for m in _NUM.finditer(text):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def answer_in_observations(task_expected: dict, traj: dict) -> bool:
    outputs = [r.get("output") for s in traj["steps"] for r in s.get("tool_results", []) if r.get("ok")]
    if not outputs:
        return False
    if task_expected["type"] == "number":
        target, tol = float(task_expected["value"]), float(task_expected.get("abs_tol", 0.01))
        return any(abs(n - target) <= tol for o in outputs for n in _numbers_in(o))
    if task_expected["type"] == "string":
        return any(_string_match(json.dumps(o, default=str), task_expected["any_of"]) for o in outputs)
    return False


def _last_tool_result(traj: dict) -> dict | None:
    for s in reversed(traj["steps"]):
        if s.get("tool_results"):
            return s["tool_results"][-1]
    return None


def classify(task: dict, traj: dict, score: dict) -> dict:
    """task: Task.to_dict(); traj: Trajectory.to_dict(); score: score_answer(...). Returns label + flags."""
    expected_tools = task.get("expected_tools") or []
    used = traj.get("tools_used") or []
    results = [r for s in traj["steps"] for r in s.get("tool_results", [])]
    flags = {
        "recovered_from_tool_error": any(not r["ok"] for r in results) and bool(results) and results[-1]["ok"],
        "json_repaired": traj.get("n_repaired_json", 0) > 0,
        "fabricated_text_after_call": bool(traj.get("fabricated_text_after_call")),
        "unnecessary_tool_use": task["category"] == "no_tool" and bool(used),
        "lenient_match": bool(score.get("correct")) and not score.get("strict", True),
        "extra_steps": max(0, traj.get("n_tool_calls", 0) - len(task.get("gold_calls") or [])),
        "had_malformed": bool(traj.get("had_malformed")),
    }
    if score.get("correct"):
        return {"label": "success", "flags": flags}

    cat = task["category"]
    last = _last_tool_result(traj)
    final = (traj.get("final_answer") or "").strip()

    if traj.get("terminated_by") in ("max_steps", "loop_detected", "backend_error") or not final:
        if traj.get("had_malformed") and traj.get("n_tool_calls", 0) == 0:
            label = "malformed_tool_call"
        elif any(r.get("error_kind") == "unknown_tool" for r in results):
            label = "hallucinated_tool"
        else:
            label = "no_final_answer"
    elif traj.get("had_malformed") and traj.get("n_tool_calls", 0) == 0:
        label = "malformed_tool_call"
    elif any(r.get("error_kind") == "unknown_tool" for r in results) and not any(r["ok"] for r in results):
        label = "hallucinated_tool"
    # planning errors are checked before execution errors: calling the wrong tool badly is still "wrong tool"
    elif cat == "no_tool":
        label = "unnecessary_tool_then_wrong" if used else "knowledge_error"
    elif cat == "unanswerable":
        label = "fabricated_answer"
    elif expected_tools and not used:
        label = "no_tool_call"
    elif expected_tools and not (set(used) & set(expected_tools)):
        label = "wrong_tool"
    elif last is not None and not last["ok"]:
        label = "hallucinated_tool" if last.get("error_kind") == "unknown_tool" else "invalid_arguments"
    elif expected_tools and not set(expected_tools) <= set(used) and len(expected_tools) > 1 \
            and not answer_in_observations(task["expected"], traj):
        label = "incomplete_chain"
    elif answer_in_observations(task["expected"], traj):
        label = "wrong_answer_after_correct_tools"
    else:
        label = "semantically_wrong_call"
    return {"label": label, "flags": flags}


def is_syntactic(label: str) -> bool:
    return label in {"malformed_tool_call", "hallucinated_tool", "invalid_arguments", "no_final_answer"}


def is_planning(label: str) -> bool:
    return label in {"no_tool_call", "wrong_tool", "incomplete_chain", "unnecessary_tool_then_wrong"}


def is_reasoning(label: str) -> bool:
    return label in {"semantically_wrong_call", "wrong_answer_after_correct_tools", "knowledge_error", "fabricated_answer"}


LABEL_GROUP = {l: ("syntax" if is_syntactic(l) else "planning" if is_planning(l) else "reasoning" if is_reasoning(l) else "success")
               for l in FAILURE_LABELS}

_ = re  # keep import for potential downstream use
