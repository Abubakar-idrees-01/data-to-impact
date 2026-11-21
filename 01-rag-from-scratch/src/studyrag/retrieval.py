"""The retrieval pipeline: query -> BM25 + vectors -> RRF fusion -> cross-encoder rerank -> top-k.

Returns not just the hits but a full trace (each retriever's ranks, fused score, rerank score),
which the app shows in its "How I found this" panel and the evals use to diagnose failures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .chunking import Chunk
from .hybrid import reciprocal_rank_fusion
from .text import tokenize


@dataclass
class Hit:
    chunk: Chunk
    idx: int
    score: float                       # final ranking score
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None  # cross-encoder relevance probability, 0..1


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top_relevance(self) -> float:
        """Best reranker probability (or 1.0 if reranking was off, i.e. unknown)."""
        scores = [h.rerank_score for h in self.hits if h.rerank_score is not None]
        return max(scores) if scores else (1.0 if self.hits else 0.0)


class Retriever:
    def __init__(self, library, embedder, reranker=None, settings=None):
        self.lib = library
        self.embedder = embedder
        self.reranker = reranker
        self.s = settings or library.settings

    def retrieve(self, query: str, top_k: int | None = None, doc_ids: list[str] | None = None,
                 mode: str = "hybrid", rerank: bool | None = None) -> RetrievalResult:
        top_k = top_k or self.s.top_k
        rerank = self.s.use_reranker if rerank is None else rerank
        rerank = rerank and self.reranker is not None
        timings: dict[str, float] = {}
        if not self.lib.chunks:
            return RetrievalResult(query, [], timings)
        allowed = self.lib.allowed_mask(doc_ids)
        n = self.s.candidates

        bm25_hits, vec_hits = [], []
        if mode in ("hybrid", "bm25"):
            t = time.perf_counter()
            bm25_hits = self.lib.bm25.search(tokenize(query), n, allowed)
            timings["bm25"] = (time.perf_counter() - t) * 1000
        if mode in ("hybrid", "vector"):
            t = time.perf_counter()
            qv = self.embedder.embed_query(query)
            vec_hits = self.lib.vindex.search(qv, n, allowed)
            timings["vector"] = (time.perf_counter() - t) * 1000

        bm25_rank = {i: r for r, (i, _) in enumerate(bm25_hits, 1)}
        vec_rank = {i: r for r, (i, _) in enumerate(vec_hits, 1)}
        rankings = [r for r in ([i for i, _ in bm25_hits], [i for i, _ in vec_hits]) if r]
        fused = reciprocal_rank_fusion(rankings, k=self.s.rrf_k)[:n]
        hits = [Hit(self.lib.chunks[i], i, score, bm25_rank.get(i), vec_rank.get(i), score)
                for i, score in fused]

        if rerank and hits:
            t = time.perf_counter()
            probs = self.reranker.score(query, [h.chunk.index_text for h in hits])
            for h, p in zip(hits, probs):
                h.rerank_score = p
                h.score = p
            hits.sort(key=lambda h: -h.score)
            timings["rerank"] = (time.perf_counter() - t) * 1000

        return RetrievalResult(query, hits[:top_k], timings)
