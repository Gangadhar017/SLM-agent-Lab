"""Stage 4 figure + table: accuracy and failure groups per served precision (f16 / int8 / int4), with file size and,
when serve/bench.py has run, throughput.

  python scripts/plot_quant.py --tag base
  python scripts/plot_quant.py --tag base --tag lora-r16
"""

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401

from slm_agent_lab.eval.report import LABEL_COLORS, LABEL_GROUP, FAILURE_LABELS, load_runs, summarize
from slm_agent_lab.paths import FIGURES_DIR, RESULTS_DIR, ROOT

PRECISIONS = ["f16", "q8_0", "q4_k_m"]
PRETTY = {"f16": "f16", "q8_0": "int8 (Q8_0)", "q4_k_m": "int4 (Q4_K_M)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", default=None, help="tag prefix(es), e.g. base, lora-r16")
    ap.add_argument("--out", default=str(FIGURES_DIR / "quantization_accuracy.png"))
    args = ap.parse_args()
    tags = args.tag or ["base"]

    bench = {}
    bench_csv = RESULTS_DIR / "serve" / "bench.csv"
    if bench_csv.exists():
        with open(bench_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if int(r["concurrency"]) == 1:
                    bench[r["tag"]] = float(r["throughput_tok_s"])

    rows = []
    for tag in tags:
        for p in PRECISIONS:
            label = f"{tag}-{p}"
            run = RESULTS_DIR / "serve" / f"harness_{label}"
            if not (run / "trajectories.jsonl").exists():
                continue
            s = next(iter(summarize(load_runs([run]))["models"].values()))
            gguf = ROOT / "models" / f"{label}.gguf"
            rows.append({"tag": tag, "precision": p, "label": label, "summary": s,
                         "size_mb": round(gguf.stat().st_size / 1e6) if gguf.exists() else None,
                         "tok_s": bench.get(label)})
    if not rows:
        raise SystemExit("no served-harness runs found under results/serve/")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    labels = [f"{r['tag']}\n{PRETTY[r['precision']]}" for r in rows]
    accs = [r["summary"]["accuracy"] for r in rows]
    sizes = [r["size_mb"] or 0 for r in rows]
    bars = ax.bar(labels, accs, color=["#1b7837", "#5aae61", "#a6dba0"] * (len(rows) // 3 + 1))
    for b, r in zip(bars, rows):
        txt = f"{r['summary']['accuracy']:.0%}\n{r['size_mb']} MB" if r["size_mb"] else f"{r['summary']['accuracy']:.0%}"
        if r["tok_s"]:
            txt += f"\n{r['tok_s']:.0f} tok/s"
        ax.annotate(txt, (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("task accuracy (150-task subset)")
    ax.set_title("Accuracy vs. served precision (llama.cpp, CPU)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    present = [l for l in FAILURE_LABELS if any(r["summary"]["failure_labels"].get(l) for r in rows)]
    bottom = np.zeros(len(rows))
    for l in present:
        vals = np.array([r["summary"]["failure_labels"].get(l, 0) / r["summary"]["n"] for r in rows])
        ax.bar(labels, vals, bottom=bottom, color=LABEL_COLORS.get(l, "#999"), edgecolor="white", linewidth=0.5,
               label=l if l == "success" else f"{LABEL_GROUP.get(l, '')}: {l}")
        bottom += vals
    ax.set_ylabel("share of tasks")
    ax.set_title("Failure taxonomy per precision")
    ax.legend(fontsize=6.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print("| model | precision | file | accuracy | strict | malformed rate | tool error rate | syntax / planning / reasoning failures | tokens/s (c=1) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["summary"]
        g = s["failure_groups"]
        print(f"| {r['tag']} | {PRETTY[r['precision']]} | {r['size_mb'] or '-'} MB | {s['accuracy']:.1%} | {s['strict_accuracy']:.1%} | "
              f"{s['malformed_rate']:.1%} | {s['tool_error_rate']:.1%} | {g.get('syntax', 0)} / {g.get('planning', 0)} / {g.get('reasoning', 0)} | "
              f"{r['tok_s'] or '-'} |")
    print("figure:", out)


if __name__ == "__main__":
    main()
