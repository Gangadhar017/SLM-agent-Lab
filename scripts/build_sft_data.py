"""Build SFT/distillation data.

  python scripts/build_sft_data.py --source oracle                     # gold tool trajectories replayed through tools
  python scripts/build_sft_data.py --source teacher --run results/runs/granite-4.0-h-micro_train
"""

import argparse
import json

import _bootstrap  # noqa: F401

from slm_agent_lab.paths import SFT_DIR, TASKS_DIR
from slm_agent_lab.tasks.schema import load_tasks
from slm_agent_lab.train.sft_data import build_sft_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["oracle", "teacher"], default="oracle")
    ap.add_argument("--run", default=None, help="teacher run directory (with --source teacher)")
    ap.add_argument("--tasks", default=str(TASKS_DIR / "train.jsonl"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(SFT_DIR / f"{args.source}_train.jsonl")
    stats = build_sft_file(load_tasks(args.tasks), out, source=args.source, run_dir=args.run)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
