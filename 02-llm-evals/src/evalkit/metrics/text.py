"""Reference-based text metrics: compare an answer with a reference answer or a list of key facts.

These are cheap and deterministic but literal: a correct answer worded differently can score low.
That limitation is measured, not assumed, in docs/RESULTS.md (heuristic vs LLM judge on StudyRAG answers).
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
STOPWORDS = frozenset("""a an the and or but if of to in on at by for with from as is are was were be been being
it its this that these those which who whom what when where why how not no do does did can could should would
may might will shall has have had than then there their they them he she his her we our you your i me my""".split())


def normalize(text: str) -> str:
    """SQuAD-style: lowercase, drop punctuation and articles, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") or ch in string.punctuation else ch
                   for ch in text)
    return " ".join(_ARTICLES.sub(" ", text).split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall (bag of words), as in SQuAD."""
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """F-measure of the longest common subsequence of tokens (order-aware, gaps allowed)."""
    p, r = tokens(prediction), tokens(reference)
    if not p or not r:
        return float(p == r)
    prev = [0] * (len(r) + 1)                        # LCS by dynamic programming, O(len(p) * len(r))
    for x in p:
        cur = [0]
        for j, y in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


def fact_found(fact: str, answer: str, threshold: float = 0.6) -> bool:
    """A key fact counts as present when >= threshold of its content words appear in the answer."""
    fact_toks = content_tokens(fact) or tokens(fact)
    if not fact_toks:
        return True
    answer_toks = set(tokens(answer))
    return sum(t in answer_toks for t in fact_toks) / len(fact_toks) >= threshold


def key_fact_recall(answer: str, facts: list[str], threshold: float = 0.6) -> float:
    """Share of the reference's key facts that the answer contains."""
    if not facts:
        return 1.0
    return sum(fact_found(f, answer, threshold) for f in facts) / len(facts)
