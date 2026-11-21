"""BM25 keyword search, implemented from scratch with an inverted index.

For a query q and chunk d, BM25 sums over each query term t:

    idf(t) * tf(t,d) * (k1 + 1) / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )

    idf(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

- tf(t,d): how often t appears in d. More is better, but with diminishing returns (k1 controls
  how fast the score saturates).
- idf(t): rare terms are worth more than common ones ("seo" beats "marketing" in a marketing course).
- |d| / avgdl: long chunks naturally contain more words, so b normalises by length.

The inverted index maps each term to the chunks containing it, so a search only touches
chunks that share at least one term with the query instead of scanning everything.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(chunk_idx, term_freq)]
        self.idf: dict[str, float] = {}
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.avgdl = 0.0
        self.n_docs = 0

    def build(self, corpus_tokens: list[list[str]]) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, tokens in enumerate(corpus_tokens):
            for term, tf in Counter(tokens).items():
                postings[term].append((idx, tf))
        self.postings = dict(postings)
        self.n_docs = len(corpus_tokens)
        self.doc_len = np.array([len(t) for t in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        n = self.n_docs
        self.idf = {t: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1.0) for t, p in self.postings.items()}
        return self

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 score of every chunk for this query (0 for chunks sharing no terms)."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in query_tokens:                # repeated query terms count again, as in classic BM25
            plist = self.postings.get(term)
            if not plist:
                continue
            idx = np.fromiter((p[0] for p in plist), dtype=np.int64, count=len(plist))
            tf = np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist))
            scores[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[idx])
        return scores

    def search(self, query_tokens: list[str], top_k: int = 10,
               allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-k (chunk_idx, score) pairs. `allowed` is an optional boolean mask (document filter)."""
        s = self.scores(query_tokens)
        if allowed is not None:
            s = np.where(allowed, s, 0.0)
        return top_k_positive(s, top_k)


def top_k_positive(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices of the k highest strictly-positive scores, best first (O(n) selection, then sort k)."""
    k = min(k, int((scores > 0).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx]
