"""The study-notes service: what the MCP tools do, as plain Python (no MCP, so it's easy to test).

It wraps StudyRAG (project 1): its persistent library (your indexed slides, PDFs and notes) and its hybrid
retriever (BM25 + vectors + RRF + cross-encoder). Everything here is read-only and needs no LLM: the AI
client that calls these tools *is* the language model. It searches, reads the passages, writes the answer,
and can ask `check_answer` whether each sentence is backed by the passages it cites.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

MAX_TOP_K = 10
MAX_ANSWER_CHARS = 8000


class NotesError(ValueError):
    """A problem with the request (unknown document, bad location...): shown to the model as a tool error."""


@dataclass
class Backend:
    """The StudyRAG pieces the tools need. Built from a real StudyRAG, or from fakes in tests."""
    library: Any
    retriever: Any
    embedder: Any
    settings: Any


def load_backend() -> Backend:
    """Load StudyRAG with its default settings (index in 01-rag-from-scratch/storage, models in its models/)."""
    from studyrag.core import StudyRAG
    rag = StudyRAG()
    return Backend(rag.library, rag.retriever, rag.embedder, rag.settings)


class Notes:
    def __init__(self, backend: Backend | None = None):
        self._backend = backend
        self._lock = threading.Lock()

    @property
    def b(self) -> Backend:
        # Lazy: loading the embedding + reranker models takes seconds, and the MCP handshake must be instant.
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    self._backend = load_backend()
        return self._backend

    # ------------------------------------------------------------------ helpers
    def _doc(self, name: str):
        d = self.b.library.find_doc(name.strip())
        if d is None:
            titles = ", ".join(sorted(x.title for x in self.b.library.docs.values())[:12]) or "none indexed"
            raise NotesError(f"no document matches {name!r}. Available: {titles}")
        return d

    @staticmethod
    def _passage(chunk, n: int | None = None, relevance: float | None = None) -> dict:
        p = {"citation": chunk.header, "document": chunk.doc_title, "location": chunk.location,
             "chunk_id": chunk.chunk_id, "text": chunk.text}
        if n is not None:
            p = {"n": n, **p}
        if relevance is not None:
            p["relevance"] = round(float(relevance), 3)
        return p

    # ------------------------------------------------------------------ tools
    def search(self, query: str, top_k: int = 4, document: str | None = None) -> dict:
        query = query.strip()
        if not query:
            raise NotesError("query is empty")
        if not 1 <= top_k <= MAX_TOP_K:
            raise NotesError(f"top_k must be between 1 and {MAX_TOP_K}")
        doc_ids = [self._doc(document).doc_id] if document else None
        res = self.b.retriever.retrieve(query, top_k=top_k, doc_ids=doc_ids)
        threshold = self.b.settings.refusal_threshold
        found = bool(res.hits) and res.top_relevance >= threshold
        passages = [self._passage(h.chunk, n, h.rerank_score) for n, h in enumerate(res.hits, 1)]
        note = ("Cite passages as [n] with their citation. Answer only from them."
                if found else
                "Nothing relevant enough was found (best relevance below the calibrated threshold "
                f"{threshold}). Say the notes don't cover this rather than answering from general knowledge.")
        return {"query": query, "found": found, "best_relevance": round(float(res.top_relevance), 3),
                "passages": passages, "guidance": note}

    def list_documents(self) -> dict:
        docs = sorted(self.b.library.docs.values(), key=lambda d: d.title.lower())
        return {"count": len(docs), "documents": [
            {"title": d.title, "doc_id": d.doc_id, "type": d.doc_type, "chunks": d.n_chunks, "words": d.n_words,
             "ocr_pages": d.n_ocr} for d in docs]}

    def outline(self, document: str) -> dict:
        from studyrag.study.summarize import outline
        d = self._doc(document)
        return {"document": d.title, "outline": outline(self.b.library.chunks_of(d.doc_id))}

    def get_passage(self, document: str, location: int) -> dict:
        """The passage(s) covering slide/page `location` of a document (the unit citations point to)."""
        d = self._doc(document)
        chunks = [c for c in self.b.library.chunks_of(d.doc_id) if c.loc_start <= location <= c.loc_end]
        if not chunks:
            rng = [c.loc_start for c in self.b.library.chunks_of(d.doc_id)]
            span = f"{min(rng)}–{max(c.loc_end for c in self.b.library.chunks_of(d.doc_id))}" if rng else "none"
            raise NotesError(f"{d.title} has no passage at location {location} (available: {span})")
        return {"document": d.title, "passages": [self._passage(c) for c in chunks]}

    def check_answer(self, answer: str, chunk_ids: list[str]) -> dict:
        """Check each sentence of an answer against the passages it cites ([n] = chunk_ids[n-1])."""
        from studyrag.verify import verify_answer
        answer = answer.strip()
        if not answer:
            raise NotesError("answer is empty")
        if len(answer) > MAX_ANSWER_CHARS:
            raise NotesError(f"answer is longer than {MAX_ANSWER_CHARS} characters")
        if not chunk_ids:
            raise NotesError("give the chunk_ids of the passages the answer cites, in [n] order")
        by_id = {c.chunk_id: c for c in self.b.library.chunks}
        missing = [i for i in chunk_ids if i not in by_id]
        if missing:
            raise NotesError(f"unknown chunk_id(s): {', '.join(missing[:5])} (use ids returned by search_notes)")
        sources = [by_id[i] for i in chunk_ids]
        ver = verify_answer(answer, sources, self.b.embedder, self.b.settings.support_threshold)
        checks = [{"sentence": c.sentence, "cited": c.cited, "supported": c.supported,
                   "support": round(float(c.support), 3), "best_source": c.best_source} for c in ver.checks]
        return {"supported_ratio": round(ver.supported_ratio, 3), "sentences": checks,
                "unsupported": [c["sentence"] for c in checks if not c["supported"]],
                "method": "word overlap + embedding similarity (StudyRAG's verifier; a heuristic, not entailment)"}

    def document_text(self, document: str, max_chars: int = 20000) -> str:
        d = self._doc(document)
        parts = [f"## {c.location}\n{c.text}" for c in self.b.library.chunks_of(d.doc_id)]
        text = f"# {d.title}\n\n" + "\n\n".join(parts)
        return text if len(text) <= max_chars else text[:max_chars] + "\n\n[... truncated]"

