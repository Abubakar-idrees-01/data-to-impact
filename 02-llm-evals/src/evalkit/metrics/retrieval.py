"""Retrieval metrics over a ranked list of results.

`relevant` is either a collection of relevant ids, or a predicate `item -> bool` (useful when relevance
means "overlaps the gold page", not an exact id). Ranks are 1-based.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Collection, Sequence

Relevance = Collection[Any] | Callable[[Any], bool]


def _is_rel(relevant: Relevance) -> Callable[[Any], bool]:
    return relevant if callable(relevant) else (lambda x: x in relevant)


def first_relevant_rank(ranked: Sequence[Any], relevant: Relevance, k: int | None = None) -> int | None:
    is_rel = _is_rel(relevant)
    for rank, item in enumerate(ranked[:k] if k else ranked, 1):
        if is_rel(item):
            return rank
    return None


def hit_at_k(ranked: Sequence[Any], relevant: Relevance, k: int) -> float:
    """1 if any relevant item is in the top k."""
    return float(first_relevant_rank(ranked, relevant, k) is not None)


def reciprocal_rank(ranked: Sequence[Any], relevant: Relevance, k: int | None = None) -> float:
    """1/rank of the first relevant item (0 if none in the top k). Its mean over queries is MRR."""
    rank = first_relevant_rank(ranked, relevant, k)
    return 1.0 / rank if rank else 0.0


def precision_at_k(ranked: Sequence[Any], relevant: Relevance, k: int) -> float:
    is_rel = _is_rel(relevant)
    top = ranked[:k]
    return sum(map(is_rel, top)) / k if k else 0.0


def recall_at_k(ranked: Sequence[Any], relevant: Collection[Any], k: int) -> float:
    """Share of all relevant ids found in the top k (needs the full relevant set, not a predicate)."""
    if not relevant:
        return 1.0
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def ndcg_at_k(ranked: Sequence[Any], gains: dict[Any, float] | Collection[Any], k: int) -> float:
    """Normalized discounted cumulative gain: rewards relevant items near the top, with graded relevance.

    gains: {id: relevance grade}, or a collection of ids (all grade 1).
    """
    g = gains if isinstance(gains, dict) else {x: 1.0 for x in gains}
    dcg = sum(g.get(item, 0.0) / math.log2(i + 1) for i, item in enumerate(ranked[:k], 1))
    ideal = sorted(g.values(), reverse=True)[:k]
    idcg = sum(v / math.log2(i + 1) for i, v in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0
