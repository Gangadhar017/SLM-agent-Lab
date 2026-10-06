"""Ablation: is the 0 % on document questions a harness artefact or model behaviour?

Runs the doc_single tasks of the standard evaluation subset under controlled variants with the *same* model,
decoding and scoring. Only one thing changes per variant:

  baseline        exactly the main-run configuration (should reproduce the main-run number)
  doc_tool_first  doc_search listed first in the tool list (ordering bias?)
  cue_prefix      question prefixed with "According to the company documents, " (surface-cue hypothesis)
  sql_removed     sql_query removed from the tool list (is the database tool an attractor?)
  doc_tool_only   doc_search is the only tool offered (can the model use the tool at all?)

  python scripts/ablate_doc_routing.py --model ibm-granite/granite-4.0-350m
"""

import argparse
import dataclasses
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

from slm_agent_lab.eval.report import load_runs, summarize
from slm_agent_lab.eval.runner import load_records, run_eval
from slm_agent_lab.eval.subset import stratified_subset
from slm_agent_lab.paths import RESULTS_DIR, TASKS_DIR
from slm_agent_lab.tasks.schema import load_tasks
from slm_agent_lab.tools.registry import TOOL_SCHEMAS


def tools_named(*names: str) -> list[dict]:
    return [t for n in names for t in TOOL_SCHEMAS if t["function"]["name"] == n]


VARIANTS = {
    "baseline": {},
    "doc_tool_first": {"tools": tools_named("doc_search", "calculator", "unit_convert", "sql_query")},
    "cue_prefix": {"prefix": "According to the company documents, "},
    "sql_removed": {"tools": tools_named("calculator", "unit_convert", "doc_search")},
    "doc_tool_only": {"tools": tools_named("doc_search")},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.0-350m")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--category", default="doc_single")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    tasks = [t for t in stratified_subset(load_tasks(TASKS_DIR / "test.jsonl"), args.limit, args.seed)
             if t.category == args.category]
    from slm_agent_lab.agent.backends import HFBackend
    backend = HFBackend(args.model, adapter=args.adapter, dtype=args.dtype, max_new_tokens=args.max_new_tokens)
    label = re.sub(r"[^A-Za-z0-9._+-]+", "_", backend.label)
    root = RESULTS_DIR / "ablations" / "doc_routing" / label
    rows = []
    for name in args.variants.split(","):
        cfg = VARIANTS[name]
        variant_tasks = tasks
        if "prefix" in cfg:
            variant_tasks = [dataclasses.replace(t, prompt=cfg["prefix"] + t.prompt[0].lower() + t.prompt[1:]) for t in tasks]
        out = root / name
        print(f"\n=== variant {name}: {len(variant_tasks)} tasks -> {out}")
        run_eval(backend, variant_tasks, out, tools=cfg.get("tools"),
                 extra_meta={"variant": name, "category": args.category, "limit": args.limit, "seed": args.seed})
        recs = load_records(out)
        s = next(iter(summarize(load_runs([out]))["models"].values()))
        used = sum("doc_search" in r["tools_used"] for r in recs)
        first = sum(bool(r["tools_used"]) and r["tools_used"][0] == "doc_search" for r in recs)
        labels = {k: v for k, v in s["failure_labels"].items() if k != "success"}
        rows.append({"variant": name, "n": len(recs), "accuracy": s["accuracy"], "doc_search_used": used / len(recs),
                     "doc_search_first_call": first / len(recs), "tool_error_rate": s["tool_error_rate"],
                     "failure_labels": labels})
        print(json.dumps(rows[-1]))

    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = ["| variant | n | accuracy | doc_search used | doc_search first call | top failure labels |", "|---|---|---|---|---|---|"]
    for r in rows:
        top = ", ".join(f"{k} {v}" for k, v in sorted(r["failure_labels"].items(), key=lambda kv: -kv[1])[:3])
        md.append(f"| {r['variant']} | {r['n']} | {r['accuracy']:.0%} | {r['doc_search_used']:.0%} | "
                  f"{r['doc_search_first_call']:.0%} | {top} |")
    (root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
