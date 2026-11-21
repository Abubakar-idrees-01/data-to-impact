"""Grounded question answering: retrieve -> (maybe refuse) -> generate with citations -> verify.

The flow is split into prepare() / finalize() so the web app can stream the model's tokens
in between, while scripts and evals simply call ask().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Settings
from .prompts import NOT_FOUND, QA_SYSTEM, qa_user
from .retrieval import RetrievalResult, Retriever
from .verify import Verification, extract_citations, verify_answer

REFUSAL_TEXT = "I couldn't find this in your documents."


@dataclass
class Prepared:
    question: str
    retrieval: RetrievalResult
    messages: list[dict]
    refused_early: bool          # True when retrieval found nothing relevant: skip the LLM entirely
    t_start: float = field(default_factory=time.perf_counter)


@dataclass
class Answer:
    question: str
    text: str
    status: str                  # "answered" | "not_found"
    sources: list                # Hits shown to the LLM; source [n] = sources[n-1]
    cited: list[int]
    verification: Verification
    confidence: str              # "high" | "medium" | "low" | "none"
    retrieval: RetrievalResult
    seconds: float


class QAEngine:
    def __init__(self, retriever: Retriever, llm, embedder, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.s = settings

    def prepare(self, question: str, doc_ids: list[str] | None = None) -> Prepared:
        res = self.retriever.retrieve(question, doc_ids=doc_ids)
        refused = not res.hits or res.top_relevance < self.s.refusal_threshold
        msgs = [{"role": "system", "content": QA_SYSTEM},
                {"role": "user", "content": qa_user(question, [h.chunk for h in res.hits])}]
        return Prepared(question, res, msgs, refused)

    def finalize(self, prep: Prepared, raw_text: str) -> Answer:
        hits = prep.retrieval.hits
        elapsed = time.perf_counter() - prep.t_start
        text = raw_text.strip()
        if prep.refused_early or not text or text.upper().startswith(NOT_FOUND) or NOT_FOUND in text[:40]:
            return Answer(prep.question, REFUSAL_TEXT, "not_found", hits, [], Verification(), "none",
                          prep.retrieval, elapsed)
        cited = sorted({n for n in extract_citations(text) if 1 <= n <= len(hits)})
        ver = verify_answer(text, [h.chunk for h in hits], self.embedder, self.s.support_threshold)
        return Answer(prep.question, text, "answered", hits, cited, ver,
                      self.confidence(prep.retrieval.top_relevance, ver, cited), prep.retrieval, elapsed)

    @staticmethod
    def confidence(top_relevance: float, ver: Verification, cited: list[int]) -> str:
        """Combine 'did we find relevant material?' with 'is the answer backed by it?'."""
        r = ver.supported_ratio
        if top_relevance >= 0.5 and r >= 0.8 and cited:
            return "high"
        if top_relevance >= 0.1 and r >= 0.5:
            return "medium"
        return "low"

    def ask(self, question: str, doc_ids: list[str] | None = None, max_tokens: int = 400) -> Answer:
        prep = self.prepare(question, doc_ids)
        raw = "" if prep.refused_early else self.llm.generate(prep.messages, max_tokens=max_tokens,
                                                              temperature=self.s.temperature)
        return self.finalize(prep, raw)
