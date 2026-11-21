"""studyrag: a private, offline study assistant built on retrieval-augmented generation (RAG).

Pipeline: load documents -> chunk -> index (BM25 + vectors) -> hybrid retrieval -> rerank
-> grounded generation with citations -> claim verification.
"""

__version__ = "0.1.0"
