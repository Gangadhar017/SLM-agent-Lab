"""Generate the test and train task sets (deterministic). Usage: python scripts/make_tasks.py [--seed 0]"""

import argparse
import json

import _bootstrap  # noqa: F401

from slm_agent_lab.paths import TASKS_DIR, DATA_DIR
from slm_agent_lab.tasks.generate import generate_tasks, task_counts
from slm_agent_lab.tasks.schema import save_tasks
from slm_agent_lab.tools.sql_query import export_sqlite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    test = generate_tasks(seed=args.seed, split="test")
    train = generate_tasks(seed=args.seed + 1, split="train", exclude_prompts={t.prompt for t in test})
    assert not ({t.prompt for t in test} & {t.prompt for t in train}), "train/test prompt overlap"
    save_tasks(test, TASKS_DIR / "test.jsonl")
    save_tasks(train, TASKS_DIR / "train.jsonl")
    export_sqlite(DATA_DIR / "company.sqlite")
    print("test :", len(test), json.dumps(task_counts(test)))
    print("train:", len(train), json.dumps(task_counts(train)))
    print("ood tasks in test:", sum(t.ood for t in test))
    print("saved to", TASKS_DIR)


if __name__ == "__main__":
    main()
