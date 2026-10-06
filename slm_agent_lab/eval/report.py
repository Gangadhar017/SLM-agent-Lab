"""Aggregate one or more runs into tables and figures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .taxonomy import FAILURE_LABELS, LABEL_GROUP

CATEGORY_ORDER = ["calc_single", "convert_single", "sql_single", "doc_single", "multi_step", "no_tool", "unanswerable"]

# fixed, group-coherent palette: green = success, reds = syntax, blues = planning, purples = reasoning
LABEL_COLORS = {
    "success": "#1b7837",
    "malformed_tool_call": "#67001f",
    "hallucinated_tool": "#b2182b",
    "invalid_arguments": "#ef8a62",
    "no_final_answer": "#fddbc7",
    "no_tool_call": "#2166ac",
    "wrong_tool": "#4393c3",
    "incomplete_chain": "#92c5de",
    "unnecessary_tool_then_wrong": "#d1e5f0",
    "semantically_wrong_call": "#762a83",
    "wrong_answer_after_correct_tools": "#9970ab",
    "knowledge_error": "#c2a5cf",
    "fabricated_answer": "#e7d4e8",
}


def load_runs(run_dirs: list[str | Path]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        d = Path(d)
        traj = d / "trajectories.jsonl"
        if not traj.exists():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        with open(traj, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                rows.append({
                    "run": d.name,
                    "model": r.get("model") or meta.get("label"),
                    "task_id": r["task_id"],
                    "category": r["category"],
                    "template": r.get("template"),
                    "ood": bool(r.get("ood")),
                    "difficulty": r.get("difficulty", 0),
                    "correct": bool(r["correct"]),
                    "strict": bool(r.get("strict", r["correct"])),
                    "label": r["label"],
                    "group": LABEL_GROUP.get(r["label"], "other"),
                    "n_steps": r.get("n_steps", 0),
                    "n_tool_calls": r.get("n_tool_calls", 0),
                    "n_tool_errors": r.get("n_tool_errors", 0),
                    "prompt_tokens": r.get("prompt_tokens", 0),
                    "completion_tokens": r.get("completion_tokens", 0),
                    "generation_time_s": r.get("generation_time_s", 0.0),
                    "wall_time_s": r.get("wall_time_s", 0.0),
                    "terminated_by": r.get("terminated_by"),
                    "had_malformed": bool(r.get("had_malformed")),
                    "json_repaired": bool((r.get("flags") or {}).get("json_repaired")),
                    "unnecessary_tool_use": bool((r.get("flags") or {}).get("unnecessary_tool_use")),
                    "fabricated_text_after_call": bool(r.get("fabricated_text_after_call")),
                    "call_sources": ",".join(sorted(set(r.get("call_sources") or []))),
                })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    out: dict = {"models": {}}
    for model, g in df.groupby("model"):
        tokens = g["completion_tokens"].sum()
        gen_time = g["generation_time_s"].sum()
        by_cat = g.groupby("category")["correct"].mean().reindex(CATEGORY_ORDER).dropna().round(3).to_dict()
        labels = g["label"].value_counts().to_dict()
        groups = g["group"].value_counts().to_dict()
        out["models"][model] = {
            "n": int(len(g)),
            "accuracy": round(float(g["correct"].mean()), 4),
            "strict_accuracy": round(float(g["strict"].mean()), 4),
            "accuracy_iid": round(float(g[~g["ood"]]["correct"].mean()), 4) if (~g["ood"]).any() else None,
            "accuracy_ood": round(float(g[g["ood"]]["correct"].mean()), 4) if g["ood"].any() else None,
            "accuracy_by_category": by_cat,
            "failure_labels": labels,
            "failure_groups": groups,
            "mean_steps": round(float(g["n_steps"].mean()), 2),
            "mean_tool_calls": round(float(g["n_tool_calls"].mean()), 2),
            "tool_error_rate": round(float(g["n_tool_errors"].sum() / max(1, g["n_tool_calls"].sum())), 3),
            "malformed_rate": round(float(g["had_malformed"].mean()), 3),
            "json_repaired_rate": round(float(g["json_repaired"].mean()), 3),
            "unnecessary_tool_rate": round(float(g[g["category"] == "no_tool"]["unnecessary_tool_use"].mean()), 3)
            if (g["category"] == "no_tool").any() else None,
            "mean_completion_tokens": round(float(g["completion_tokens"].mean()), 1),
            "mean_prompt_tokens": round(float(g["prompt_tokens"].mean()), 1),
            "completion_tokens_per_s": round(float(tokens / gen_time), 2) if gen_time > 0 else None,
            "mean_wall_time_s": round(float(g["wall_time_s"].mean()), 2),
        }
    return out


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def write_markdown(summary: dict, df: pd.DataFrame, path: Path) -> None:
    models = list(summary["models"].keys())
    parts = ["# Evaluation summary\n"]
    rows = []
    for m in models:
        s = summary["models"][m]
        rows.append([m, s["n"], f"{s['accuracy']:.1%}", f"{s['strict_accuracy']:.1%}",
                     f"{s['accuracy_iid']:.1%}" if s["accuracy_iid"] is not None else "-",
                     f"{s['accuracy_ood']:.1%}" if s["accuracy_ood"] is not None else "-",
                     s["mean_tool_calls"], f"{s['tool_error_rate']:.1%}", f"{s['malformed_rate']:.1%}",
                     s["mean_completion_tokens"], s["completion_tokens_per_s"] or "-"])
    parts.append(_md_table(["model", "n", "acc", "strict acc", "acc (iid)", "acc (ood)", "tool calls/task",
                            "tool error rate", "malformed rate", "compl. tokens/task", "tokens/s"], rows))
    parts.append("\n\n## Accuracy by category\n")
    cats = [c for c in CATEGORY_ORDER if c in set(df["category"])]
    rows = []
    for m in models:
        s = summary["models"][m]["accuracy_by_category"]
        rows.append([m] + [f"{s[c]:.0%}" if c in s else "-" for c in cats])
    parts.append(_md_table(["model"] + cats, rows))
    parts.append("\n\n## Failure taxonomy (count of tasks)\n")
    labels = [l for l in FAILURE_LABELS if l != "success" and l in set(df["label"])]
    rows = []
    for m in models:
        s = summary["models"][m]["failure_labels"]
        rows.append([m] + [s.get(l, 0) for l in labels])
    parts.append(_md_table(["model"] + labels, rows))
    parts.append("\n\n## Failure groups\n")
    rows = []
    for m in models:
        s = summary["models"][m]["failure_groups"]
        n = summary["models"][m]["n"]
        rows.append([m] + [f"{s.get(gname, 0)} ({s.get(gname, 0) / n:.0%})" for gname in ["syntax", "planning", "reasoning"]])
    parts.append(_md_table(["model", "syntax (can't talk to tools)", "planning (wrong/no tool)", "reasoning (wrong use/answer)"], rows))
    parts.append("\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_figures(summary: dict, df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    models = list(summary["models"].keys())
    paths = []

    # 1. accuracy by category
    cats = [c for c in CATEGORY_ORDER if c in set(df["category"])]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / max(1, len(models))
    x = np.arange(len(cats))
    for i, m in enumerate(models):
        s = summary["models"][m]["accuracy_by_category"]
        ax.bar(x + i * width, [s.get(c, 0) for c in cats], width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(cats, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Task accuracy by category")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    p = fig_dir / "accuracy_by_category.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 2. failure taxonomy stacked bars (share of all tasks)
    labels = [l for l in FAILURE_LABELS if l in set(df["label"])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(models))
    for l in labels:
        vals = np.array([summary["models"][m]["failure_labels"].get(l, 0) / summary["models"][m]["n"] for m in models])
        name = l if l == "success" else f"{LABEL_GROUP.get(l, 'other')}: {l}"
        ax.bar(models, vals, bottom=bottom, label=name, color=LABEL_COLORS.get(l, "#999999"), edgecolor="white",
               linewidth=0.5)
        bottom += vals
    ax.set_ylabel("share of tasks")
    ax.set_title("Where each model fails (failure taxonomy)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    p = fig_dir / "failure_taxonomy.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 3. accuracy vs throughput
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in models:
        s = summary["models"][m]
        if s["completion_tokens_per_s"]:
            ax.scatter(s["completion_tokens_per_s"], s["accuracy"], s=80)
            ax.annotate(m, (s["completion_tokens_per_s"], s["accuracy"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("generation throughput (completion tokens / s)")
    ax.set_ylabel("task accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy vs. throughput")
    ax.grid(alpha=0.3)
    p = fig_dir / "accuracy_vs_throughput.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def write_report(run_dirs: list[str | Path], out_dir: str | Path, fig_dir: str | Path | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(run_dirs)
    if df.empty:
        raise SystemExit("no trajectories found in the given run directories")
    summary = summarize(df)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv(out_dir / "records.csv", index=False)
    write_markdown(summary, df, out_dir / "summary.md")
    make_figures(summary, df, Path(fig_dir) if fig_dir else out_dir / "figures")
    return summary
