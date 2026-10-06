"""Evaluate a model on a task set and log every trajectory.

Examples
  python scripts/run_eval.py --model ibm-granite/granite-4.0-350m --limit 60
  python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --categories sql_single,multi_step
  python scripts/run_eval.py --model ibm-granite/granite-4.0-350m --adapter results/sweep/r8_lr2e-4_s0
  python scripts/run_eval.py --backend openai --base-url http://localhost:8000 --model ibm-granite/granite-4.0-1b
"""

import argparse
import random
import re
from pathlib import Path

import _bootstrap  # noqa: F401

from slm_agent_lab.eval.report import summarize, load_runs
from slm_agent_lab.eval.runner import run_eval
from slm_agent_lab.paths import RESULTS_DIR, TASKS_DIR
from slm_agent_lab.tasks.schema import load_tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id (or served model name with --backend openai)")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter to merge before evaluation")
    ap.add_argument("--backend", choices=["hf", "openai"], default="hf")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--tokenizer", default=None, help="tokenizer id for --backend openai (defaults to --model)")
    ap.add_argument("--tasks", default=str(TASKS_DIR / "test.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="evaluate a stratified subset of this many tasks")
    ap.add_argument("--categories", default=None, help="comma-separated category filter")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--out", default=None, help="run directory (default results/runs/<model-label>)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="seed for the stratified subset")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    if args.categories:
        keep = set(args.categories.split(","))
        tasks = [t for t in tasks if t.category in keep]
    if args.limit and args.limit < len(tasks):
        # stratified: keep category proportions, deterministic order
        rng = random.Random(args.seed)
        by_cat: dict[str, list] = {}
        for t in tasks:
            by_cat.setdefault(t.category, []).append(t)
        chosen = []
        for cat, items in by_cat.items():
            k = max(1, round(args.limit * len(items) / len(tasks)))
            chosen.extend(rng.sample(items, min(k, len(items))))
        tasks = chosen[: args.limit]

    if args.backend == "hf":
        from slm_agent_lab.agent.backends import HFBackend
        backend = HFBackend(args.model, adapter=args.adapter, dtype=args.dtype, max_new_tokens=args.max_new_tokens,
                            num_threads=args.threads)
    else:
        from slm_agent_lab.agent.backends import OpenAICompatBackend
        backend = OpenAICompatBackend(args.base_url, args.model, tokenizer_id=args.tokenizer,
                                      max_new_tokens=args.max_new_tokens)

    label = re.sub(r"[^A-Za-z0-9._+-]+", "_", backend.label)
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "runs" / label
    print(f"model={backend.label}  tasks={len(tasks)}  out={out_dir}")
    run_eval(backend, tasks, out_dir, max_steps=args.max_steps, resume=not args.no_resume,
             extra_meta={"tasks_file": args.tasks, "limit": args.limit, "categories": args.categories})
    summary = summarize(load_runs([out_dir]))
    for m, s in summary["models"].items():
        print(f"\n{m}: accuracy={s['accuracy']:.1%} (strict {s['strict_accuracy']:.1%}) n={s['n']}")
        print("  by category:", s["accuracy_by_category"])
        print("  failures   :", {k: v for k, v in s["failure_labels"].items() if k != 'success'})
        print(f"  tokens/s   : {s['completion_tokens_per_s']}   mean tool calls: {s['mean_tool_calls']}")


if __name__ == "__main__":
    main()
