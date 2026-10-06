"""LoRA paper (Hu et al. 2021, §7.2) subspace-similarity reproduction on our own adapters.

  python scripts/subspace_analysis.py results/sweep/granite-4.0-350m/r8_lr0.0002_s0 results/sweep/granite-4.0-350m/r64_lr0.0002_s0
  python scripts/subspace_analysis.py A B --module q_proj --out docs/figures/subspace_r8_r64.png
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from slm_agent_lab.paths import FIGURES_DIR
from slm_agent_lab.train.subspace import compare_adapters, load_lora_A, plot_grid, random_baseline, write_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_a")
    ap.add_argument("adapter_b")
    ap.add_argument("--module", default=None, help="only modules whose name contains this (e.g. q_proj)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = compare_adapters(args.adapter_a, args.adapter_b, args.module)
    name_a, name_b = Path(args.adapter_a).name, Path(args.adapter_b).name
    out = Path(args.out) if args.out else FIGURES_DIR / f"subspace_{name_a}_vs_{name_b}.png"
    plot_grid(res["mean_grid"], f"phi({name_a}, {name_b}) averaged over {len(res['modules'])} modules", out,
              xlabel=f"j (top-j directions of {name_b})", ylabel=f"i (top-i directions of {name_a})")
    dim = next(iter(load_lora_A(args.adapter_a).values())).shape[1]
    base = random_baseline(res["rank_a"], res["rank_b"], dim)
    plot_grid(base, f"random baseline: phi for Gaussian A (r={res['rank_a']} vs r={res['rank_b']}, d={dim})",
              out.with_name(out.stem + "_random_baseline.png"))
    res["random_baseline_top1"] = float(base[0, 0])
    res["random_baseline_diag"] = [float(base[i, i]) for i in range(min(res["rank_a"], res["rank_b"]))]
    write_summary(res, out.with_suffix(".json"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("mean_grid", "modules")}, indent=2))
    print(f"modules compared: {len(res['modules'])}; figure: {out}")


if __name__ == "__main__":
    main()
