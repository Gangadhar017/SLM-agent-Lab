"""Debugging tool for agents: pretty-print trajectories from a run, filtered by failure label or task id.

  python scripts/inspect_run.py results/runs/granite-4.0-350m --label wrong_tool --n 3
  python scripts/inspect_run.py results/runs/granite-4.0-350m --task test-sql_single-1a2b3c4d
  python scripts/inspect_run.py results/runs/granite-4.0-350m --summary
"""

import argparse
import json
from collections import Counter

import _bootstrap  # noqa: F401

from slm_agent_lab.eval.runner import load_records
from slm_agent_lab.tasks.schema import load_tasks
from slm_agent_lab.paths import TASKS_DIR


def show(rec: dict, task_prompt: str | None) -> None:
    print("=" * 100)
    print(f"task {rec['task_id']}  [{rec['category']}]  label={rec['label']}  correct={rec['correct']}  "
          f"terminated_by={rec['terminated_by']}")
    if task_prompt:
        print("PROMPT   :", task_prompt)
    print("EXPECTED :", json.dumps(rec.get("expected")), " tools:", rec.get("expected_tools"))
    for s in rec["steps"]:
        print(f"--- step {s['index']}  ({s['completion_tokens']} tok, {s['latency_s']}s, stop={s['stop_reason']})")
        print("MODEL    :", s["raw_output"][:700].replace("\n", "\n           "))
        for r in s.get("tool_results", []):
            status = "OK " if r["ok"] else f"ERR({r.get('error_kind')})"
            out = json.dumps(r["output"] if r["ok"] else r["error"], default=str)
            print(f"TOOL {status}: {r['name']}({json.dumps(r['arguments'])}) -> {out[:300]}")
        if s.get("malformed"):
            print("MALFORMED:", [m[:120] for m in s["malformed"]])
    print("FINAL    :", rec["final_answer"][:300])
    print("FLAGS    :", {k: v for k, v in rec["flags"].items() if v})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--label", default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--tasks", default=str(TASKS_DIR / "test.jsonl"))
    args = ap.parse_args()

    recs = load_records(args.run)
    prompts = {}
    try:
        prompts = {t.id: t.prompt for t in load_tasks(args.tasks)}
    except FileNotFoundError:
        pass
    if args.summary:
        print(f"{len(recs)} trajectories, accuracy {sum(r['correct'] for r in recs) / len(recs):.1%}")
        for label, n in Counter(r["label"] for r in recs).most_common():
            print(f"  {label:36s} {n}")
        print("call dialects:", Counter(src for r in recs for src in r.get("call_sources", [])))
        return
    shown = 0
    for r in recs:
        if args.task and r["task_id"] != args.task:
            continue
        if args.label and r["label"] != args.label:
            continue
        if args.category and r["category"] != args.category:
            continue
        show(r, prompts.get(r["task_id"]))
        shown += 1
        if shown >= args.n:
            break
    if not shown:
        print("no matching trajectories")


if __name__ == "__main__":
    main()
