"""End-to-end smoke test: load a model, run a handful of tasks, print what happened. Use before a long run.

  python scripts/smoke_test.py --model ibm-granite/granite-4.0-350m --n 3
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401

from slm_agent_lab.agent.backends import HFBackend
from slm_agent_lab.agent.loop import run_task
from slm_agent_lab.eval.scoring import score_answer
from slm_agent_lab.eval.taxonomy import classify
from slm_agent_lab.paths import TASKS_DIR
from slm_agent_lab.tasks.schema import load_tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.0-350m")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--categories", default="calc_single,sql_single,doc_single,no_tool,multi_step")
    args = ap.parse_args()

    t0 = time.perf_counter()
    backend = HFBackend(args.model, max_new_tokens=args.max_new_tokens)
    print(f"loaded {backend.label} in {time.perf_counter() - t0:.1f}s; native tool template: {backend.renderer.native_tools}")
    tasks = load_tasks(TASKS_DIR / "test.jsonl")
    cats = args.categories.split(",")
    picked = []
    for c in cats:
        picked.extend([t for t in tasks if t.category == c][: max(1, args.n // len(cats))])
    picked = picked[: args.n] if args.n < len(picked) else picked
    for task in picked:
        print("\n" + "=" * 90)
        print(f"[{task.category}] {task.prompt}")
        print("expected:", json.dumps(task.expected), "| tools:", task.expected_tools)
        traj = run_task(backend, task).to_dict()
        for s in traj["steps"]:
            print(f"  step {s['index']} ({s['completion_tokens']} tok, {s['latency_s']}s): {s['raw_output'][:400]!r}")
            for r in s["tool_results"]:
                print(f"    -> {r['name']}({json.dumps(r['arguments'])}) ok={r['ok']} "
                      f"{json.dumps(r['output'] if r['ok'] else r['error'], default=str)[:200]}")
        score = score_answer(task.expected, traj["final_answer"])
        tax = classify(task.to_dict(), traj, score)
        print(f"  final: {traj['final_answer'][:200]!r}")
        print(f"  correct={score['correct']} label={tax['label']} terminated_by={traj['terminated_by']} "
              f"tokens={traj['completion_tokens']} time={traj['wall_time_s']}s")


if __name__ == "__main__":
    main()
