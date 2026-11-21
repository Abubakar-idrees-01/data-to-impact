"""Test fakes: deterministic stand-ins for the embedding model and the LLM.

They let us test our own logic (chunking, BM25, fusion, refusal, verification, incremental
indexing) in milliseconds, without downloading or running any model.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from studyrag.config import Settings
from studyrag.text import tokenize


class FakeEmbedder:
    """Hashing bag-of-words: texts sharing stemmed words get similar vectors."""
    model_name = "fake-bow"
    dim = 128

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v if v.any() else np.ones(self.dim, dtype=np.float32) * 1e-3

    def embed_documents(self, texts, batch_size: int = 32):
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """Returns scripted outputs in order; records the prompts it received."""
    name = "fake-llm"

    def __init__(self, outputs=None, json_outputs=None):
        self.outputs = list(outputs or [])
        self.json_outputs = list(json_outputs or [])
        self.calls = []

    def generate(self, messages, max_tokens=512, temperature=0.2, stream=False):
        self.calls.append(messages)
        text = self.outputs.pop(0) if self.outputs else "NOT_FOUND"
        return iter([text]) if stream else text

    def generate_json(self, messages, schema, max_tokens=768, temperature=0.3):
        self.calls.append(messages)
        out = self.json_outputs.pop(0)
        return json.loads(out) if isinstance(out, str) else out


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def settings(tmp_path):
    return Settings(storage_dir=tmp_path / "storage", models_dir=tmp_path / "models",
                    use_reranker=False, chunk_size=60, min_chunk_words=10)


@pytest.fixture
def docs_folder(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "seo.md").write_text(
        "# Search Engine Optimization\n"
        "SEO is the practice of improving a website so it ranks higher in organic search results.\n"
        "Key on-page factors include title tags, meta descriptions and keyword placement.\n\n"
        "# Backlinks\n"
        "Backlinks are links from other websites. Search engines treat quality backlinks as votes of trust.\n")
    (d / "email.md").write_text(
        "# Email Marketing\n"
        "Email marketing sends targeted messages to a list of subscribers who opted in.\n"
        "Open rate and click-through rate are the main email metrics.\n\n"
        "# Segmentation\n"
        "Segmentation divides subscribers into groups by behaviour or demographics for personalised campaigns.\n")
    return d
