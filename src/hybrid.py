import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.chunking import CHUNKERS
from src.ingest import load_folder
from src.search import search

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "with", "and", "or", "is",
    "are", "it", "its", "how", "do", "does", "what", "why", "you", "can", "that",
    "this", "me",
}
_cache: dict = {}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def _index(strategy: str):
    if strategy not in _cache:
        chunks = CHUNKERS[strategy](load_folder(Path("data")))
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        _cache[strategy] = (bm25, chunks)
    return _cache[strategy]


def keyword_search(query: str, strategy: str = "page", k: int = 10) -> list[dict]:
    bm25, chunks = _index(strategy)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in ranked]


def _key(r: dict) -> tuple:
    return (r["source"], r["page"], r["text"])


def hybrid_search(query: str, strategy: str = "page", k: int = 5, pool: int = 20, rrf_k: int = 60) -> list[dict]:
    """Merge dense and BM25 rankings with Reciprocal Rank Fusion."""
    dense = search(query, strategy=strategy, k=pool)
    sparse = keyword_search(query, strategy=strategy, k=pool)
    fused: dict = {}
    items: dict = {}
    for results in (dense, sparse):
        for rank, r in enumerate(results, start=1):
            key = _key(r)
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank)
            items[key] = r
    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{**items[key], "score": fused[key]} for key in best]
