"""Stage 2/3 figure + table: task accuracy and eval loss vs LoRA rank, with the un-tuned baseline as a reference.

  python scripts/plot_sweep.py --sweep results/sweep/granite-4.0-350m --baseline results/runs/granite-4.0-350m
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from slm_agent_lab.eval.report import load_runs, summarize
from slm_agent_lab.paths import FIGURES_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/sweep/granite-4.0-350m")
    ap.add_argument("--baseline", default="results/runs/granite-4.0-350m")
    ap.add_argument("--out", default=str(FIGURES_DIR / "lora_rank_sweep.png"))
    args = ap.parse_args()

    df = pd.read_csv(Path(args.sweep) / "sweep.csv")
    df = df[df["eval_accuracy"].notna()] if "eval_accuracy" in df else df
    base = summarize(load_runs([args.baseline]))["models"]
    base_s = next(iter(base.values()))
    base_acc = base_s["accuracy"]

    # per-rank eval breakdown from the adapter eval runs (same subset as the baseline)
    rows = []
    for _, r in df.sort_values(["rank", "lr", "seed"]).iterrows():
        run_dir = Path(r["out_dir"]) / "eval"
        if not run_dir.is_absolute():
            run_dir = Path(args.sweep).parent.parent.parent / run_dir if not run_dir.exists() else run_dir
        s = next(iter(summarize(load_runs([run_dir]))["models"].values())) if (run_dir / "trajectories.jsonl").exists() else None
        rows.append({
            "rank": int(r["rank"]), "lr": float(r["lr"]), "seed": int(r["seed"]),
            "trainable_pct": round(float(r["trainable_pct"]), 3),
            "eval_loss_before": float(r["eval_loss_before"]), "eval_loss_after": float(r["eval_loss_after"]),
            "accuracy": s["accuracy"] if s else float(r.get("eval_accuracy", float("nan"))),
            "accuracy_ood": s["accuracy_ood"] if s else None,
            "doc_single": s["accuracy_by_category"].get("doc_single") if s else None,
            "multi_step": s["accuracy_by_category"].get("multi_step") if s else None,
            "malformed_rate": s["malformed_rate"] if s else None,
            "planning_failures": s["failure_groups"].get("planning", 0) if s else None,
            "train_time_s": float(r["train_time_s"]),
        })
    tab = pd.DataFrame(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for lr, g in tab.groupby("lr"):
        gg = g.groupby("rank")["accuracy"].agg(["mean", "std", "count"]).reset_index()
        ax.errorbar(gg["rank"], gg["mean"], yerr=gg["std"].fillna(0), marker="o", capsize=3, label=f"LoRA, lr={lr:g}")
    ax.axhline(base_acc, ls="--", color="gray", label=f"no fine-tuning ({base_acc:.0%})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(tab["rank"].unique()))
    ax.set_xticklabels([str(x) for x in sorted(tab["rank"].unique())])
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("task accuracy (150-task subset)")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy vs. rank")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for lr, g in tab.groupby("lr"):
        gg = g.groupby("rank")["eval_loss_after"].mean().reset_index()
        ax.plot(gg["rank"], gg["eval_loss_after"], marker="s", label=f"after training, lr={lr:g}")
    ax.axhline(tab["eval_loss_before"].mean(), ls="--", color="gray", label="before training")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(tab["rank"].unique()))
    ax.set_xticklabels([str(x) for x in sorted(tab["rank"].unique())])
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("held-out SFT loss (assistant tokens)")
    ax.set_title("Eval loss vs. rank")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"{base_s and list(base)[0]}: LoRA rank sweep (alpha = 2r, 2 epochs, oracle trajectories)", fontsize=10)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    tab.to_csv(Path(args.sweep) / "sweep_table.csv", index=False)
    print(f"| rank | trainable % | eval loss before -> after | accuracy | ood | doc_single | multi_step | planning failures |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| none (baseline) | 0 | - | {base_acc:.1%} | {base_s['accuracy_ood']:.1%} | "
          f"{base_s['accuracy_by_category'].get('doc_single', 0):.0%} | {base_s['accuracy_by_category'].get('multi_step', 0):.0%} | "
          f"{base_s['failure_groups'].get('planning', 0)} |")
    for _, r in tab.iterrows():
        acc_ood = f"{r['accuracy_ood']:.1%}" if r["accuracy_ood"] is not None else "-"
        print(f"| {r['rank']} | {r['trainable_pct']} | {r['eval_loss_before']:.2f} -> {r['eval_loss_after']:.2f} | "
              f"{r['accuracy']:.1%} | {acc_ood} | {r['doc_single']:.0%} | {r['multi_step']:.0%} | {r['planning_failures']} |")
    print("figure:", out)


if __name__ == "__main__":
    main()
