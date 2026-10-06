"""Controlled LoRA sweep: vary rank, learning rate and seed; hold everything else fixed.

  # smoke test on CPU (minutes):
  python scripts/lora_sweep.py --model ibm-granite/granite-4.0-350m --ranks 4 --lrs 2e-4 --seeds 0 --epochs 0.2 --max-examples 40
  # full sweep (GPU, e.g. Kaggle T4):
  python scripts/lora_sweep.py --model ibm-granite/granite-4.0-350m --ranks 1,2,4,8,16,64 --lrs 1e-4,2e-4,5e-4 --seeds 0,1,2 --eval-tasks 100
"""

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

from slm_agent_lab.paths import RESULTS_DIR, SFT_DIR, TASKS_DIR
from slm_agent_lab.train.lora import TrainConfig, append_csv, train_lora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.0-350m")
    ap.add_argument("--data", default=str(SFT_DIR / "oracle_train.jsonl"))
    ap.add_argument("--ranks", default="4,8,16")
    ap.add_argument("--lrs", default="2e-4")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--targets", default=",".join(["q_proj", "k_proj", "v_proj", "o_proj"]))
    ap.add_argument("--out", default=None, help="sweep root (default results/sweep/<model>)")
    ap.add_argument("--eval-tasks", type=int, default=0, help="after each run, evaluate on this many test tasks")
    ap.add_argument("--eval-max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    label = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model.split("/")[-1])
    root = Path(args.out) if args.out else RESULTS_DIR / "sweep" / label
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "sweep.csv"

    for rank in [int(x) for x in args.ranks.split(",")]:
        for lr in [float(x) for x in args.lrs.split(",")]:
            for seed in [int(x) for x in args.seeds.split(",")]:
                run_dir = root / f"r{rank}_lr{lr:g}_s{seed}"
                if (run_dir / "train_result.json").exists():
                    print(f"skip {run_dir.name} (done)")
                    continue
                print(f"\n=== rank={rank} lr={lr:g} seed={seed} -> {run_dir}")
                cfg = TrainConfig(model_id=args.model, data_path=args.data, out_dir=str(run_dir), rank=rank, lr=lr,
                                  seed=seed, epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
                                  max_len=args.max_len, max_examples=args.max_examples, dtype=args.dtype,
                                  target_modules=tuple(args.targets.split(",")))
                result = train_lora(cfg)
                if args.eval_tasks:
                    from slm_agent_lab.agent.backends import HFBackend
                    from slm_agent_lab.eval.report import load_runs, summarize
                    from slm_agent_lab.eval.runner import run_eval
                    from slm_agent_lab.tasks.schema import load_tasks
                    import random
                    tasks = load_tasks(TASKS_DIR / "test.jsonl")
                    random.Random(0).shuffle(tasks)
                    tasks = sorted(tasks[: args.eval_tasks], key=lambda t: t.id)
                    backend = HFBackend(args.model, adapter=str(run_dir), dtype=args.dtype,
                                        max_new_tokens=args.eval_max_new_tokens)
                    eval_dir = run_dir / "eval"
                    run_eval(backend, tasks, eval_dir, resume=False)
                    s = summarize(load_runs([eval_dir]))["models"]
                    s = next(iter(s.values()))
                    result.update({"eval_accuracy": s["accuracy"], "eval_n": s["n"],
                                   "eval_malformed_rate": s["malformed_rate"], "eval_tool_error_rate": s["tool_error_rate"]})
                    del backend
                print(json.dumps({k: result[k] for k in ("rank", "lr", "seed", "eval_loss_before", "eval_loss_after",
                                                          "final_train_loss", "trainable_pct", "train_time_s")}))
                append_csv(csv_path, result)
    print("sweep table:", csv_path)


if __name__ == "__main__":
    main()
