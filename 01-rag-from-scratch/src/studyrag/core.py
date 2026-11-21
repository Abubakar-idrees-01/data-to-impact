"""StudyRAG: one object that wires every component together.

    rag = StudyRAG()                     # loads embedder, reranker and the saved library
    rag.library.sync_folder("data/my_docs")
    print(rag.ask("What is SEO?").text)

The LLM is loaded lazily (on first use) because it is the slowest and largest component;
retrieval-only use (search, evals of retrieval) never pays that cost.
"""

from __future__ import annotations

from functools import cached_property

from .answer import Answer, QAEngine
from .config import Settings
from .embeddings import Embedder, Reranker
from .library import Library
from .retrieval import Retriever


class StudyRAG:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        s = self.settings
        cache = s.models_dir / "fastembed"
        self.embedder = Embedder(s.embedding_model, cache)
        self.reranker = Reranker(s.reranker_model, cache) if s.use_reranker else None
        self.library = Library(s, self.embedder)
        self.retriever = Retriever(self.library, self.embedder, self.reranker, s)

    @cached_property
    def llm(self):
        from .llm import load_llm
        return load_llm(self.settings)

    @property
    def qa(self) -> QAEngine:
        return QAEngine(self.retriever, self.llm, self.embedder, self.settings)

    def ask(self, question: str, doc_ids: list[str] | None = None) -> Answer:
        return self.qa.ask(question, doc_ids)
