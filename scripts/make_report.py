"""Build summary tables + figures from one or more run directories.

  python scripts/make_report.py                      # all runs under results/runs
  python scripts/make_report.py results/runs/a results/runs/b --out results/report
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from slm_agent_lab.eval.report import write_report
from slm_agent_lab.paths import FIGURES_DIR, RESULTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run directories (default: every dir under results/runs)")
    ap.add_argument("--out", default=str(RESULTS_DIR / "report"))
    ap.add_argument("--figures", default=str(FIGURES_DIR))
    args = ap.parse_args()
    runs = [Path(r) for r in args.runs] or sorted(p for p in (RESULTS_DIR / "runs").glob("*") if p.is_dir())
    summary = write_report(runs, args.out, args.figures)
    print(json.dumps({m: {k: v for k, v in s.items() if k in ("n", "accuracy", "strict_accuracy", "accuracy_ood",
                                                               "completion_tokens_per_s")}
                      for m, s in summary["models"].items()}, indent=2))
    print(f"wrote {args.out}/summary.md and figures to {args.figures}")


if __name__ == "__main__":
    main()
