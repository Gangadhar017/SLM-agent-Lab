"""Repository paths. Everything is resolved relative to the repo root so scripts work from any cwd."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SLM_AGENT_LAB_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus" / "docs.json"
TASKS_DIR = DATA_DIR / "tasks"
SFT_DIR = DATA_DIR / "sft"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "docs" / "figures"


def ensure_dirs() -> None:
    for d in (TASKS_DIR, SFT_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
