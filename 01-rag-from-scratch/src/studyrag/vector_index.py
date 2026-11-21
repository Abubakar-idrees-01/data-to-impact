"""Vector (semantic) search, implemented from scratch with NumPy.

Each chunk is an embedding vector. We L2-normalise all vectors once, so cosine similarity
becomes a plain dot product:  cos(q, d) = q . d  when |q| = |d| = 1.
Searching is then a single matrix-vector multiply over all chunks ("exact" / brute-force search).

Why not an approximate index (HNSW, IVF)? A study library holds thousands of chunks, not millions.
Brute force over 10,000 x 384 floats takes ~1-2 ms on a laptop CPU and is exact, so it is the
right tool here. scripts/benchmark_vector.py compares it against FAISS to show this.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


class VectorIndex:
    def __init__(self, vectors: np.ndarray | None = None):
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        if vectors is not None and len(vectors):
            self.build(vectors)

    def build(self, vectors: np.ndarray) -> "VectorIndex":
        self.matrix = l2_normalize(vectors)
        return self

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def scores(self, query_vec: np.ndarray) -> np.ndarray:
        if not len(self):
            return np.zeros(0, dtype=np.float32)
        return self.matrix @ l2_normalize(query_vec)

    def search(self, query_vec: np.ndarray, top_k: int = 10,
               allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        s = self.scores(query_vec)
        if not len(s):
            return []
        if allowed is not None:
            s = np.where(allowed, s, -np.inf)
            k = min(top_k, int(allowed.sum()))
        else:
            k = min(top_k, len(s))
        if k <= 0:
            return []
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx], kind="stable")]
        return [(int(i), float(s[i])) for i in idx]
