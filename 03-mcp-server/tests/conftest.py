"""Fakes so the server's logic is tested in seconds, without downloading or running any model."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from studyrag.config import Settings
from studyrag.library import Library
from studyrag.retrieval import Retriever
from studyrag.text import tokenize
from studyrag_mcp.evals import Evals
from studyrag_mcp.notes import Backend, Notes


class FakeEmbedder:
    """Hashing bag-of-words: texts sharing words get similar vectors."""
    model_name = "fake-bow"
    dim = 128

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v if v.any() else np.ones(self.dim, dtype=np.float32) * 1e-3

    def embed_documents(self, texts, batch_size=32):
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def notes(tmp_path) -> Notes:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "seo.md").write_text(
        "# Search Engine Optimization\n"
        "SEO is the practice of improving a website so it ranks higher in organic search results.\n"
        "Key on-page factors include title tags, meta descriptions and keyword placement.\n\n"
        "# Backlinks\n"
        "Backlinks are links from other websites. Search engines treat quality backlinks as votes of trust.\n")
    (docs / "email.md").write_text(
        "# Email Marketing\n"
        "Email marketing sends targeted messages to a list of subscribers who opted in.\n"
        "Open rate and click-through rate are the main email metrics.\n")
    settings = Settings(storage_dir=tmp_path / "storage", models_dir=tmp_path / "models",
                        ocr_cache_file=tmp_path / "ocr.json", use_reranker=False, chunk_size=60, min_chunk_words=5)
    emb = FakeEmbedder()
    lib = Library(settings, emb)
    lib.sync_folder(docs)
    return Notes(Backend(lib, Retriever(lib, emb, None, settings), emb, settings))


@pytest.fixture
def evals(tmp_path) -> Evals:
    from evalkit.runs import Run
    root = tmp_path / "results"
    Run.build("demo", "v1", [{"id": f"q{i}", "correct": i % 3 != 0} for i in range(30)]).save(root)
    Run.build("demo", "v2", [{"id": f"q{i}", "correct": True} for i in range(30)]).save(root)
    (root / "secret.txt").write_text(json.dumps({"password": "x"}))
    return Evals(root)
