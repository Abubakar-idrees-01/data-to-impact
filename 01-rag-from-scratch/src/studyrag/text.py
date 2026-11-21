"""Tokenisation for keyword search (BM25), built from scratch.

Steps: lowercase -> split into word tokens -> drop stopwords -> light stemming.
Stemming maps "marketing", "marketed", "markets" to the same stem so a query for
"market" still matches. We use a small rule-based suffix stripper rather than a full
Porter stemmer: it is easy to read, predictable, and good enough for retrieval
(docs/concepts/02-bm25.md shows the measured effect).
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been before
being below between both but by can can't cannot could couldn't did didn't do does doesn't doing
don't down during each few for from further had hadn't has hasn't have haven't having he he'd
he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into
is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only
or other ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up very was
wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while
who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves also may might must shall will just can use used using via etc e.g i.e
""".split())

# (suffix, replacement, minimum stem length left after stripping)
_SUFFIX_RULES = [
    ("ational", "ate", 3), ("ization", "ize", 3), ("fulness", "ful", 3), ("iveness", "ive", 3),
    ("ements", "", 4), ("ement", "", 4), ("ments", "", 4), ("ment", "", 4),
    ("ities", "", 3), ("ity", "", 3), ("ingly", "", 3), ("edly", "", 3),
    ("ing", "", 3), ("ies", "y", 2), ("ied", "y", 2), ("ed", "", 3),
    ("ers", "", 3), ("er", "", 3), ("es", "", 3), ("s", "", 3),
]


def stem(word: str) -> str:
    """'advertising', 'advertised', 'advertisement', 'advertise' -> 'advertis'."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith(("ss", "us", "is")):        # "business", "status", "analysis" keep their s
        return word
    base = word
    for suffix, repl, min_len in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_len:
            base = word[: len(word) - len(suffix)] + repl
            # undo doubled consonant: "shopping" -> "shopp" -> "shop"
            if suffix in ("ing", "ed", "er", "ers") and len(base) > 3 and base[-1] == base[-2] \
                    and base[-1] not in "lsz":
                base = base[:-1]
            break
    if base.endswith("e") and len(base) > 3:     # Porter's final-e rule: "page"/"pages" -> "pag"
        base = base[:-1]
    return base


def tokenize(text: str, use_stemming: bool = True, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    tokens = [t.split("'")[0] for t in tokens]               # "brand's" -> "brand"
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if use_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens
