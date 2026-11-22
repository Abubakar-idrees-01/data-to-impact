"""Cheap, non-LLM hallucination detectors. An LLM judge has to beat these to be worth its cost.

- lexical:  a response sentence is unsupported when too few of its content words occur in the source
- numbers:  the response states a number that never appears in the source
- constant: always "hallucinated" / always "supported" (shows what chance-level agreement looks like)
"""

from __future__ import annotations

import re

from .metrics.text import STOPWORDS, tokens

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if len(s.split()) >= 3]


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _content(text: str) -> list[str]:
    return [_stem(t) for t in tokens(text) if t not in STOPWORDS]


def sentence_coverage(response: str, source: str) -> list[tuple[str, float]]:
    """(sentence, share of its content words found in the source) for each response sentence."""
    src = set(_content(source))
    out = []
    for s in sentences(response):
        words = _content(s)
        if words:
            out.append((s, sum(w in src for w in words) / len(words)))
    return out


def lexical_score(response: str, source: str) -> float:
    """Lowest sentence coverage (0..1). Low = some sentence is mostly words the source never uses."""
    cov = [c for _, c in sentence_coverage(response, source)]
    return min(cov) if cov else 1.0


def lexical_hallucination(response: str, source: str, threshold: float = 0.5) -> bool:
    return lexical_score(response, source) < threshold


def novel_numbers(response: str, source: str) -> list[str]:
    """Numbers in the response that never occur in the source (commas ignored: 1,200 == 1200)."""
    src = {n.replace(",", "") for n in _NUM.findall(source or "")}
    return [n for n in _NUM.findall(response or "") if n.replace(",", "") not in src]


def numbers_hallucination(response: str, source: str) -> bool:
    return bool(novel_numbers(response, source))
