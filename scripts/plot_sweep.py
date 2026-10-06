"""Stage 2/3 figure + table: task accuracy and eval loss vs LoRA rank, with the un-tuned baseline as reference.

Reads each run's train_result.json under the sweep directory. Accuracy comes from whichever evaluation exists
for that run: a PyTorch eval (<run>/eval/) or the merged adapter served through llama.cpp
(results/serve/harness_lora-r<rank>-f16/, plus -q4_k_m for the int4 column).

  python scripts/plot_sweep.py --sweep results/sweep/granite-4.0-350m
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from slm_agent_lab.eval.report import load_runs, summarize
from slm_agent_lab.paths import FIGURES_DIR, RESULTS_DIR


def run_summary(run_dir: Path):
    if (run_dir / "trajectories.jsonl").exists():
        return next(iter(summarize(load_runs([run_dir]))["models"].values()))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/sweep/granite-4.0-350m")
    ap.add_argument("--baseline-torch", default="results/runs/granite-4.0-350m")
    ap.add_argument("--baseline-f16", default="results/serve/harness_base-bf16",
                    help="served un-tuned baseline (bf16 to match the adapters; falls back to the f16 run)")
    ap.add_argument("--baseline-q4", default="results/serve/harness_base-q4_k_m")
    ap.add_argument("--out", default=str(FIGURES_DIR / "lora_rank_sweep.png"))
    args = ap.parse_args()

    base_torch = run_summary(Path(args.baseline_torch))
    base_f16 = run_summary(Path(args.baseline_f16)) or run_summary(RESULTS_DIR / "serve" / "harness_base-f16")
    base_q4 = run_summary(Path(args.baseline_q4))

    rows = []
    for tr in sorted(Path(args.sweep).glob("r*_lr*_s*/train_result.json")):
        res = json.loads(tr.read_text(encoding="utf-8"))["result"]
        run_dir = tr.parent
        rank, lr, seed = int(res["rank"]), float(res["lr"]), int(res["seed"])
        # adapters are served from bf16 (fine-tuned weights overflow float16); fall back to f16 if that is what exists
        s_f16 = None
        if seed == 0:
            s_f16 = run_summary(RESULTS_DIR / "serve" / f"harness_lora-r{rank}-bf16") or \
                    run_summary(RESULTS_DIR / "serve" / f"harness_lora-r{rank}-f16")
        s_q4 = run_summary(RESULTS_DIR / "serve" / f"harness_lora-r{rank}-q4_k_m") if seed == 0 else None
        s_torch = run_summary(run_dir / "eval")
        s = s_f16 or s_torch
        rows.append({
            "rank": rank, "lr": lr, "seed": seed,
            "trainable_pct": round(float(res["trainable_pct"]), 3),
            "eval_loss_before": float(res["eval_loss_before"]), "eval_loss_after": float(res["eval_loss_after"]),
            "final_train_loss": res.get("final_train_loss"), "train_time_s": float(res["train_time_s"]),
            "eval_backend": "llama.cpp bf16" if s_f16 else ("torch fp32" if s_torch else None),
            "accuracy": s["accuracy"] if s else None,
            "accuracy_ood": s["accuracy_ood"] if s else None,
            "doc_single": s["accuracy_by_category"].get("doc_single") if s else None,
            "multi_step": s["accuracy_by_category"].get("multi_step") if s else None,
            "malformed_rate": s["malformed_rate"] if s else None,
            "planning_failures": s["failure_groups"].get("planning", 0) if s else None,
            "accuracy_int4": s_q4["accuracy"] if s_q4 else None,
            "malformed_rate_int4": s_q4["malformed_rate"] if s_q4 else None,
        })
    tab = pd.DataFrame(rows)
    if tab.empty:
        raise SystemExit("no train_result.json found under " + args.sweep)
    tab.to_csv(Path(args.sweep) / "sweep_table.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = tab[tab["accuracy"].notna() & (tab["seed"] == 0)].sort_values("rank")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    if not ev.empty:
        ax.plot(ev["rank"], ev["accuracy"], marker="o", label="LoRA adapter, bf16 (llama.cpp)" if ev["eval_backend"].iloc[0].startswith("llama") else "LoRA adapter (torch)")
        if ev["accuracy_int4"].notna().any():
            ax.plot(ev["rank"], ev["accuracy_int4"], marker="s", ls="--", label="LoRA adapter, int4 (Q4_K_M)")
    if base_f16:
        ax.axhline(base_f16["accuracy"], ls="--", color="gray", label=f"no fine-tuning, served ({base_f16['accuracy']:.0%})")
    if base_q4:
        ax.axhline(base_q4["accuracy"], ls=":", color="gray", label=f"no fine-tuning, int4 ({base_q4['accuracy']:.0%})")
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
    s0 = tab[tab["seed"] == 0].sort_values("rank")
    ax.plot(s0["rank"], s0["eval_loss_after"], marker="s", label="after 2 epochs")
    ax.axhline(tab["eval_loss_before"].mean(), ls="--", color="gray", label="before training")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(tab["rank"].unique()))
    ax.set_xticklabels([str(x) for x in sorted(tab["rank"].unique())])
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("held-out SFT loss (assistant tokens)")
    ax.set_title("Eval loss vs. rank")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Granite-4.0-350M: LoRA rank sweep (alpha = 2r, lr 2e-4, 2 epochs on 290 oracle trajectories)", fontsize=10)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    def pct(x):
        return "-" if x is None or pd.isna(x) else f"{x:.1%}"

    print("| rank | seed | trainable % | eval loss before -> after | accuracy (served, bf16) | ood | doc_single | multi_step | planning failures | accuracy (int4) | malformed (int4) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    if base_f16:
        print(f"| none (baseline) | - | 0 | - | {pct(base_f16['accuracy'])} | {pct(base_f16['accuracy_ood'])} | "
              f"{pct(base_f16['accuracy_by_category'].get('doc_single'))} | {pct(base_f16['accuracy_by_category'].get('multi_step'))} | "
              f"{base_f16['failure_groups'].get('planning', 0)} | {pct(base_q4['accuracy']) if base_q4 else '-'} | "
              f"{pct(base_q4['malformed_rate']) if base_q4 else '-'} |")
    for _, r in tab.sort_values(["rank", "seed"]).iterrows():
        print(f"| {r['rank']} | {r['seed']} | {r['trainable_pct']} | {r['eval_loss_before']:.2f} -> {r['eval_loss_after']:.2f} | "
              f"{pct(r['accuracy'])} | {pct(r['accuracy_ood'])} | {pct(r['doc_single'])} | {pct(r['multi_step'])} | "
              f"{'-' if pd.isna(r['planning_failures']) else int(r['planning_failures'])} | {pct(r['accuracy_int4'])} | {pct(r['malformed_rate_int4'])} |")
    print("figure:", out)


if __name__ == "__main__":
    main()
