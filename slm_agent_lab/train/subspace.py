"""Reproduction of the subspace-similarity analysis from Hu et al., "LoRA: Low-Rank Adaptation of Large Language
Models" (2021), Section 7.2 / Figure 3.

For two adapters with ranks r1 and r2 trained on the same data, take the down-projection matrices A_r1, A_r2 of the
same weight, compute their right singular vectors U, and measure how much the top-i singular directions of one
overlap with the top-j of the other:

    phi(A_r1, A_r2, i, j) = || U_{A_r1}^{i T} U_{A_r2}^{j} ||_F^2 / min(i, j)      in [0, 1]

The paper's claim: the top singular directions of r=8 and r=64 overlap strongly (phi > 0.5), while the remaining
directions are mostly noise -- i.e. a very low intrinsic rank suffices. We recompute phi on our own adapters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_lora_A(adapter_dir: str | Path) -> dict[str, np.ndarray]:
    """Returns {module_name: A (r x d_in)} for every lora_A weight in the adapter."""
    from safetensors.numpy import load_file
    adapter_dir = Path(adapter_dir)
    weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
    out = {}
    for k, v in weights.items():
        if "lora_A" in k:
            name = k.replace(".lora_A.weight", "").replace("base_model.model.", "")
            out[name] = v.astype(np.float32)
    return out


def right_singular_vectors(A: np.ndarray) -> np.ndarray:
    """A is (r x d). Rows span an r-dim subspace of R^d; return orthonormal basis ordered by singular value (d x r)."""
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    return vt.T  # d x r


def phi(U1: np.ndarray, U2: np.ndarray, i: int, j: int) -> float:
    m = U1[:, :i].T @ U2[:, :j]
    return float(np.linalg.norm(m, "fro") ** 2 / min(i, j))


def phi_grid(A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
    U1, U2 = right_singular_vectors(A1), right_singular_vectors(A2)
    r1, r2 = U1.shape[1], U2.shape[1]
    return np.array([[phi(U1, U2, i, j) for j in range(1, r2 + 1)] for i in range(1, r1 + 1)])


def compare_adapters(dir_a: str | Path, dir_b: str | Path, module_filter: str | None = None) -> dict:
    A, B = load_lora_A(dir_a), load_lora_A(dir_b)
    common = [k for k in A if k in B and (module_filter is None or module_filter in k)]
    if not common:
        raise ValueError("no common LoRA modules between the two adapters")
    grids = [phi_grid(A[k], B[k]) for k in common]
    mean_grid = np.mean(grids, axis=0)
    ra, rb = mean_grid.shape
    return {
        "modules": common,
        "rank_a": ra,
        "rank_b": rb,
        "mean_grid": mean_grid,
        "top1_overlap": float(mean_grid[0, 0]),
        "topk_overlap_diag": [float(mean_grid[i, i]) for i in range(min(ra, rb))],
        "full_overlap": float(mean_grid[ra - 1, rb - 1]),
    }


def compare_seeds(dir_a: str | Path, dir_b: str | Path, module_filter: str | None = None) -> dict:
    """Same-rank adapters from different seeds: how much of the learned subspace is seed-independent (signal)?"""
    return compare_adapters(dir_a, dir_b, module_filter)


def random_baseline(rank_a: int, rank_b: int, dim: int, n: int = 20, seed: int = 0) -> np.ndarray:
    """phi grid for random Gaussian A matrices: what overlap looks like when there is no shared structure."""
    rng = np.random.default_rng(seed)
    grids = [phi_grid(rng.standard_normal((rank_a, dim)), rng.standard_normal((rank_b, dim))) for _ in range(n)]
    return np.mean(grids, axis=0)


def plot_grid(grid: np.ndarray, title: str, path: str | Path, xlabel: str = "j (rank-b directions)",
              ylabel: str = "i (rank-a directions)") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto",
                   extent=[0.5, grid.shape[1] + 0.5, 0.5, grid.shape[0] + 0.5])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="phi (normalised subspace similarity)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_summary(result: dict, path: str | Path) -> None:
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")
