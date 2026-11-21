"""Text embeddings (meaning -> vector) and cross-encoder reranking.

We use fastembed, which runs small transformer models through ONNX Runtime: no PyTorch, ~100 MB
of model files, fast on CPU. The model is off-the-shelf; everything we do *with* the vectors
(indexing, search, fusion) is our own code.

- Embedder (bi-encoder): encodes query and chunk *separately* -> fast, can pre-compute all chunks.
- Reranker (cross-encoder): reads query and chunk *together* -> much more accurate, but must run
  per (query, chunk) pair, so we only use it on the top ~20 candidates. See docs/concepts/05-reranking.md.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# BGE models were trained with this instruction on the *query* side for retrieval tasks.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str, cache_dir: Path, threads: int | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), threads=threads)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))
        self._query_prefix = _BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(list(self._model.embed(texts, batch_size=batch_size))).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model.embed([self._query_prefix + text]))), dtype=np.float32)



def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    def __init__(self, model_name: str, cache_dir: Path, threads: int | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir), threads=threads)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance probability in [0, 1] for each passage (sigmoid of the cross-encoder logit)."""
        if not passages:
            return []
        return [sigmoid(float(s)) for s in self._model.rerank(query, passages)]
