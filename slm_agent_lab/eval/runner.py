"""Runs a task set through a backend, scores and classifies each trajectory, and logs everything to JSONL.

Resumable: re-running with the same output directory skips tasks that already have a logged trajectory, so a
CPU run can be stopped and continued.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from ..agent.loop import run_task
from ..tasks.schema import Task
from .scoring import score_answer
from .taxonomy import classify


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ids.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def run_eval(backend, tasks: list[Task], out_dir: str | Path, max_steps: int = 6, resume: bool = True,
             progress: bool = True, extra_meta: dict | None = None) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"
    meta = {
        "model": getattr(backend, "model_id", None),
        "adapter": getattr(backend, "adapter", None),
        "label": backend.label,
        "backend": type(backend).__name__,
        "max_new_tokens": getattr(backend, "max_new_tokens", None),
        "max_steps": max_steps,
        "n_tasks": len(tasks),
        "started": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        **(extra_meta or {}),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    done = _done_ids(traj_path) if resume else set()
    todo = [t for t in tasks if t.id not in done]
    records: list[dict] = []
    iterator = todo
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(todo, desc=backend.label, unit="task")
        except ImportError:
            pass

    t0 = time.perf_counter()
    n_correct = 0
    with open(traj_path, "a", encoding="utf-8") as f:
        for task in iterator:
            traj = run_task(backend, task, max_steps=max_steps).to_dict()
            score = score_answer(task.expected, traj["final_answer"])
            tax = classify(task.to_dict(), traj, score)
            rec = {
                **traj,
                "split": task.split,
                "ood": task.ood,
                "template": task.template,
                "difficulty": task.difficulty,
                "expected": task.expected,
                "expected_tools": task.expected_tools,
                "correct": score["correct"],
                "strict": score["strict"],
                "match_mode": score["match_mode"],
                "extracted": score["extracted"],
                "label": tax["label"],
                "flags": tax["flags"],
            }
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            records.append(rec)
            n_correct += int(score["correct"])
            if progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(acc=f"{n_correct / len(records):.2f}", label=tax["label"][:18])
    meta["finished"] = datetime.now(timezone.utc).isoformat()
    meta["elapsed_s"] = round(time.perf_counter() - t0, 1)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return records


def load_records(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "trajectories.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
