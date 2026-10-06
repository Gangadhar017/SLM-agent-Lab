"""BM25 keyword search over a small local corpus of synthetic company documents. Deterministic, no ML."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from ..paths import CORPUS_PATH

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = {
    "the", "a", "an", "of", "to", "in", "is", "for", "and", "or", "on", "at", "by", "with", "what", "how",
    "many", "much", "does", "do", "are", "per", "according", "company", "from", "that", "this", "it", "its",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class DocSearchError(ValueError):
    pass


class BM25Index:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d["title"] + " " + d["text"]) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(docs))
        df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [{t: toks.count(t) for t in set(toks)} for toks in self.doc_tokens]

    def score(self, query_tokens: list[str], idx: int) -> float:
        s = 0.0
        tf, dl = self.tf[idx], self.doc_len[idx]
        for t in query_tokens:
            if t not in tf:
                continue
            f = tf[t]
            s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = tokenize(query)
        if not q:
            raise DocSearchError("query has no searchable terms")
        scored = [(self.score(q, i), i) for i in range(len(self.docs))]
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = []
        for s, i in scored[:k]:
            if s <= 0:
                continue
            d = self.docs[i]
            out.append({"id": d["id"], "title": d["title"], "score": round(s, 3), "text": d["text"]})
        return out


@lru_cache(maxsize=1)
def load_corpus(path: str | None = None) -> list[dict]:
    p = Path(path) if path else CORPUS_PATH
    with open(p, encoding="utf-8") as f:
        return json.load(f)["documents"]


@lru_cache(maxsize=1)
def get_index() -> BM25Index:
    return BM25Index(load_corpus())


def search(query: str, k: int = 3) -> dict:
    """Search the corpus. Returns {"query": ..., "results": [{"id","title","score","text"}, ...]}."""
    if not isinstance(query, str) or not query.strip():
        raise DocSearchError("query must be a non-empty string")
    try:
        k = int(k)
    except (TypeError, ValueError) as exc:
        raise DocSearchError(f"k must be an integer, got {k!r}") from exc
    k = max(1, min(k, 5))
    results = get_index().search(query, k=k)
    return {"query": query, "results": results, "n_results": len(results)}
