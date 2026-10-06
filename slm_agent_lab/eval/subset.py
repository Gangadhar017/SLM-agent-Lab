"""The stratified evaluation subset. One function so every experiment (baselines, adapters, ablations, served
models) scores the *same* tasks."""

from __future__ import annotations

import random

from ..tasks.schema import Task


def stratified_subset(tasks: list[Task], limit: int | None, seed: int = 0) -> list[Task]:
    """Keep category proportions; deterministic for a given (tasks file, limit, seed)."""
    if not limit or limit >= len(tasks):
        return list(tasks)
    rng = random.Random(seed)
    by_cat: dict[str, list[Task]] = {}
    for t in tasks:
        by_cat.setdefault(t.category, []).append(t)
    chosen: list[Task] = []
    for items in by_cat.values():
        k = max(1, round(limit * len(items) / len(tasks)))
        chosen.extend(rng.sample(items, min(k, len(items))))
    return chosen[:limit]
