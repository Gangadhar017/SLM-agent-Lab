"""Build chat-format SFT examples from (a) oracle trajectories replayed through the real tools, or (b) a teacher
model's logged trajectories filtered to the correct ones (rejection-sampled sequence-level distillation).

Each example is a list of messages: system, user, assistant(tool_calls), tool, ..., assistant(final). The same
messages render through the student's chat template at training time, so the student learns the exact dialect
the harness parses.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..agent.loop import SYSTEM_PROMPT
from ..tasks.schema import Task
from ..tools.registry import execute_tool


def oracle_example(task: Task, system_prompt: str = SYSTEM_PROMPT) -> dict | None:
    """Replay the task's gold calls through the real tools; one tool call per assistant turn."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task.prompt}]
    for c in task.gold_calls:
        res = execute_tool(c["name"], c["arguments"])
        if not res.ok:
            return None  # gold call failed -> never train on it
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}]})
        messages.append({"role": "tool", "name": c["name"], "content": res.observation()})
    messages.append({"role": "assistant", "content": task.gold_answer})
    return {"task_id": task.id, "category": task.category, "source": "oracle", "messages": messages}


def teacher_example(record: dict, task: Task, system_prompt: str = SYSTEM_PROMPT) -> dict | None:
    """Rebuild the message list from a logged, *correct* teacher trajectory."""
    if not record.get("correct"):
        return None
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task.prompt}]
    for step in record["steps"]:
        calls = step.get("parsed_calls") or []
        results = step.get("tool_results") or []
        if calls:
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in calls[: len(results)]]})
            for c, r in zip(calls, results):
                payload = r["output"] if r["ok"] else {"error": r["error"]}
                messages.append({"role": "tool", "name": c["name"], "content": json.dumps(payload, default=str)[:1500]})
        else:
            messages.append({"role": "assistant", "content": record["final_answer"].strip()})
    if messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls"):
        return None
    return {"task_id": task.id, "category": task.category, "source": f"teacher:{record.get('model')}", "messages": messages}


def build_sft_file(tasks: list[Task], out_path: str | Path, source: str = "oracle", run_dir: str | Path | None = None,
                   skip_categories: tuple[str, ...] = ("unanswerable",)) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {t.id: t for t in tasks}
    examples: list[dict] = []
    stats = {"total_tasks": len(tasks), "kept": 0, "dropped": 0}
    if source == "oracle":
        for t in tasks:
            if t.category in skip_categories:
                stats["dropped"] += 1
                continue
            ex = oracle_example(t)
            (examples.append(ex) if ex else None)
            stats["kept" if ex else "dropped"] += 1
    else:
        with open(Path(run_dir) / "trajectories.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                t = by_id.get(rec["task_id"])
                if t is None or t.category in skip_categories:
                    stats["dropped"] += 1
                    continue
                ex = teacher_example(rec, t)
                (examples.append(ex) if ex else None)
                stats["kept" if ex else "dropped"] += 1
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    stats["path"] = str(out_path)
    return stats
