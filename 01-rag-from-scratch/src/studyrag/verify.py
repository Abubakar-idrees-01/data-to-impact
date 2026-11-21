"""Answer verification: is each sentence of the answer actually backed by a source?

LLMs can cite [2] and still say something passage [2] never says. So after generation we check
every factual sentence against the passages:

  support(sentence, passage) = 0.5 * lexical_recall + 0.5 * semantic_match

- lexical_recall: share of the sentence's content words (stemmed) that appear in the passage.
  Catches invented names, numbers and terms.
- semantic_match: best cosine similarity between the sentence and any single sentence of the
  passage, rescaled to 0..1. Catches faithful paraphrases that use different words.

A sentence is "supported" if its best support score (over the sources it cites, or over all
sources when it cites none) reaches settings.support_threshold. This is a cheap, transparent
heuristic, not a proof. Its precision/recall are measured in eval/ and reported in docs/RESULTS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .chunking import split_sentences
from .text import tokenize

_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_COS_FLOOR, _COS_CEIL = 0.55, 0.90     # bge-small cosine range mapped to 0..1


def extract_citations(text: str) -> list[int]:
    nums: list[int] = []
    for group in _CITE.findall(text):
        nums.extend(int(n) for n in re.split(r"[,;]", group))
    return nums


_MD = re.compile(r"(\*\*|__|`|^#+\s*|^\s*[-*•]\s+|^\s*\d+[.)]\s+)", re.M)


def strip_citations(text: str) -> str:
    """Remove [n] citations and Markdown decoration, leaving only the claim's words."""
    text = re.sub(r"\s*" + _CITE.pattern, "", text)
    return _MD.sub("", text).strip()


@dataclass
class SentenceCheck:
    sentence: str                # claim text (citations and Markdown removed)
    cited: list[int]
    support: float               # best support over cited sources (or all, if uncited)
    best_source: int | None      # 1-based source number that supports it best
    supported: bool
    raw: str = ""                # the sentence exactly as it appears in the answer


@dataclass
class Verification:
    checks: list[SentenceCheck] = field(default_factory=list)

    @property
    def supported_ratio(self) -> float:
        return sum(c.supported for c in self.checks) / len(self.checks) if self.checks else 1.0

    @property
    def unsupported(self) -> list[SentenceCheck]:
        return [c for c in self.checks if not c.supported]


def _lexical_recall(sent_tokens: set[str], passage_tokens: set[str]) -> float:
    return len(sent_tokens & passage_tokens) / len(sent_tokens) if sent_tokens else 0.0


def verify_answer(answer: str, sources, embedder, threshold: float = 0.55) -> Verification:
    """sources: the chunks shown to the LLM, in the same order as their [n] numbers."""
    sentences = [s for s in split_sentences(answer) if len(tokenize(strip_citations(s))) >= 3]
    if not sentences or not sources:
        return Verification()

    # sentence-level embeddings of every passage, computed once per answer
    passage_sents = [split_sentences(c.text) or [c.text] for c in sources]
    flat = [s for ps in passage_sents for s in ps]
    owners = np.array([i for i, ps in enumerate(passage_sents) for _ in ps])
    clean = [strip_citations(s) for s in sentences]
    vecs = embedder.embed_documents(clean + flat)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    ans_v, src_v = vecs[: len(clean)], vecs[len(clean):]
    sims = ans_v @ src_v.T                                  # (answer sentences) x (passage sentences)
    passage_tokens = [set(tokenize(c.text)) for c in sources]

    checks = []
    for i, (raw, text) in enumerate(zip(sentences, clean)):
        cited = [n for n in extract_citations(raw) if 1 <= n <= len(sources)]
        candidates = [n - 1 for n in cited] or list(range(len(sources)))
        toks = set(tokenize(text))
        best, best_src = 0.0, None
        for j in candidates:
            cos = float(sims[i, owners == j].max())
            semantic = min(1.0, max(0.0, (cos - _COS_FLOOR) / (_COS_CEIL - _COS_FLOOR)))
            score = 0.5 * _lexical_recall(toks, passage_tokens[j]) + 0.5 * semantic
            if score > best:
                best, best_src = score, j + 1
        checks.append(SentenceCheck(text, cited, round(best, 3), best_src, best >= threshold, raw))
    return Verification(checks)
