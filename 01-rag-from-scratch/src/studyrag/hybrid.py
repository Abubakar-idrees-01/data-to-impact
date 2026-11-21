"""Hybrid retrieval with Reciprocal Rank Fusion (RRF).

BM25 and vector search produce scores on completely different scales (BM25: 0..~20,
cosine: -1..1), so we can't simply add them. RRF ignores the raw scores and uses only ranks:

    RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))

A chunk ranked highly by *either* retriever gets a good score, and one ranked highly by *both*
gets the best score. k (default 60, from the original paper by Cormack et al., 2009) dampens
the gap between rank 1 and rank 2 so one retriever can't dominate.
"""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """rankings: one list of chunk indices per retriever, best first. Returns (idx, rrf_score), best first."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))
