"""Question generation: MCQs, short-answer questions and flashcards, each grounded in one passage.

How it works:
1. Pick source passages: by topic (retrieval) or spread evenly across the chosen documents.
2. For each passage, ask the LLM for one item as JSON (grammar-constrained, so always valid JSON).
3. Validate the item and fix known small-model habits:
   - answer position bias: small models put the right answer at "A" far too often, so we shuffle
     options and remap the answer letter;
   - duplicates: near-identical questions (embedding cosine > 0.9) are dropped;
   - grounding: the correct option must be supported by the passage (same scorer as verify.py),
     otherwise the item is flagged.
4. Yield items one by one so the UI can show each as soon as it is ready (CPU generation is slow).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Iterator

import numpy as np

from ..chunking import Chunk
from ..prompts import (FLASHCARD_SCHEMA, GEN_SYSTEM, MCQ_SCHEMA, SHORT_Q_SCHEMA, flashcards_user,
                       mcq_user, short_question_user)
from ..text import tokenize

BANNED_OPTIONS = ("all of the above", "none of the above", "both a and b", "all of these")


@dataclass
class MCQ:
    question: str
    options: list[str]           # 4 options, already shuffled
    answer_index: int            # 0..3
    explanation: str
    source: Chunk
    grounded: bool = True        # correct option supported by the passage?
    issues: list[str] = field(default_factory=list)

    @property
    def answer_letter(self) -> str:
        return "ABCD"[self.answer_index]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = {"doc_title": self.source.doc_title, "location": self.source.location,
                       "chunk_id": self.source.chunk_id}
        return d


@dataclass
class ShortQuestion:
    question: str
    answer: str
    source: Chunk


@dataclass
class Flashcard:
    front: str
    back: str
    source: Chunk


# ------------------------------------------------------------------ passage selection
def select_passages(library, retriever, n: int, doc_ids: list[str] | None = None,
                    topic: str | None = None, min_words: int = 25, seed: int | None = None) -> list[Chunk]:
    """Passages to generate from. With a topic: the most relevant ones. Without: an even spread."""
    if topic:
        res = retriever.retrieve(topic, top_k=max(n, 4), doc_ids=doc_ids)
        good = [h.chunk for h in res.hits if h.rerank_score is None or h.rerank_score >= 0.05]
        return good[:n] if good else [h.chunk for h in res.hits[:n]]
    pool = [c for c in library.chunks if (not doc_ids or c.doc_id in doc_ids) and c.word_count >= min_words]
    if not pool:
        pool = [c for c in library.chunks if not doc_ids or c.doc_id in doc_ids]
    if len(pool) <= n:
        return pool
    if seed is not None:                      # random but reproducible sample, kept in document order
        picks = sorted(random.Random(seed).sample(range(len(pool)), n))
    else:                                     # evenly spaced: covers the whole lecture
        picks = np.linspace(0, len(pool) - 1, n).round().astype(int).tolist()
    return [pool[i] for i in picks]


# ------------------------------------------------------------------ grounding / dedup helpers
def support_score(claim: str, passage: str) -> float:
    """Share of the claim's content words found in the passage (0..1)."""
    toks = set(tokenize(claim))
    return len(toks & set(tokenize(passage))) / len(toks) if toks else 1.0


class _Dedup:
    def __init__(self, embedder, threshold: float = 0.9):
        self.embedder, self.threshold, self.vecs = embedder, threshold, []

    def is_dup(self, text: str) -> bool:
        if self.embedder is None:
            return False
        v = self.embedder.embed_documents([text])[0]
        v = v / max(np.linalg.norm(v), 1e-12)
        if any(float(v @ u) > self.threshold for u in self.vecs):
            return True
        self.vecs.append(v)
        return False


# ------------------------------------------------------------------ MCQs
def validate_mcq(raw: dict, chunk: Chunk, rng: random.Random) -> MCQ | None:
    q = (raw.get("question") or "").strip()
    opts_raw = raw.get("options") or {}
    letters = list("ABCD")
    options = [str(opts_raw.get(k, "")).strip() for k in letters]
    ans = str(raw.get("answer", "")).strip().upper()[:1]
    if not q or ans not in letters or any(not o for o in options):
        return None
    if len({o.lower() for o in options}) < 4:            # duplicate options
        return None
    issues = []
    if any(o.lower().rstrip(".") in BANNED_OPTIONS for o in options):
        issues.append("uses 'all/none of the above'")
    correct = options[letters.index(ans)]
    grounded = support_score(correct, chunk.text) >= 0.5 or support_score(
        f"{correct} {raw.get('explanation', '')}", chunk.text) >= 0.6
    if not grounded:
        issues.append("correct option not clearly supported by the passage")
    order = list(range(4))
    rng.shuffle(order)                                    # remove the model's "answer is A" bias
    shuffled = [options[i] for i in order]
    return MCQ(q, shuffled, shuffled.index(correct), (raw.get("explanation") or "").strip(),
               chunk, grounded, issues)


def generate_mcqs(llm, passages: list[Chunk], difficulty: str = "medium", embedder=None,
                  seed: int = 0, retries: int = 1) -> Iterator[MCQ]:
    rng, seen, dedup = random.Random(seed), [], _Dedup(embedder)
    for chunk in passages:
        for attempt in range(retries + 1):
            msgs = [{"role": "system", "content": GEN_SYSTEM},
                    {"role": "user", "content": mcq_user(chunk, difficulty, seen)}]
            try:
                raw = llm.generate_json(msgs, MCQ_SCHEMA, max_tokens=450, temperature=0.4 + 0.2 * attempt)
            except Exception:
                continue
            mcq = validate_mcq(raw, chunk, rng)
            if mcq and not dedup.is_dup(mcq.question):
                seen.append(mcq.question)
                yield mcq
                break


# ------------------------------------------------------------------ short questions & flashcards
def generate_short_questions(llm, passages: list[Chunk], difficulty: str = "medium",
                             embedder=None) -> Iterator[ShortQuestion]:
    seen, dedup = [], _Dedup(embedder)
    for chunk in passages:
        msgs = [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": short_question_user(chunk, difficulty, seen)}]
        try:
            raw = llm.generate_json(msgs, SHORT_Q_SCHEMA, max_tokens=300)
        except Exception:
            continue
        q, a = (raw.get("question") or "").strip(), (raw.get("answer") or "").strip()
        if q and a and not dedup.is_dup(q):
            seen.append(q)
            yield ShortQuestion(q, a, chunk)


def generate_flashcards(llm, passages: list[Chunk], embedder=None) -> Iterator[Flashcard]:
    seen, dedup = [], _Dedup(embedder)
    for chunk in passages:
        msgs = [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": flashcards_user(chunk, seen)}]
        try:
            raw = llm.generate_json(msgs, FLASHCARD_SCHEMA, max_tokens=400)
        except Exception:
            continue
        for card in raw.get("cards", [])[:3]:
            front, back = (card.get("front") or "").strip(), (card.get("back") or "").strip()
            if front and back and not dedup.is_dup(front):
                seen.append(front)
                yield Flashcard(front, back, chunk)
