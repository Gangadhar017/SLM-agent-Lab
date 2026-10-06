"""Task schema. A task is a prompt plus an exact, machine-checkable expectation and a reference tool trajectory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CATEGORIES = ["calc_single", "convert_single", "sql_single", "doc_single", "multi_step", "no_tool", "unanswerable"]


@dataclass
class Task:
    id: str
    category: str
    template: str
    prompt: str
    # {"type": "number", "value": 42, "abs_tol": 0.5} | {"type": "string", "any_of": [...]} | {"type": "unanswerable"}
    expected: dict
    expected_tools: list[str]          # tools used by the reference solution, in order ([] for no_tool/unanswerable)
    gold_calls: list[dict]             # [{"name": ..., "arguments": {...}}] reference trajectory (oracle SFT data)
    gold_answer: str                   # reference final answer text
    split: str = "test"                # "test" | "train"
    ood: bool = False                  # template/document held out from the train split
    difficulty: int = 0                # number of reference tool calls
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(**d)


def save_tasks(tasks: list[Task], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_tasks(path: str | Path) -> list[Task]:
    with open(path, encoding="utf-8") as f:
        return [Task.from_dict(json.loads(line)) for line in f if line.strip()]
