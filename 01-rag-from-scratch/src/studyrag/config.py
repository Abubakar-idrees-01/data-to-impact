"""Central configuration.

Every tunable value lives here so experiments (and docs/DECISIONS.md) can point at one place.
Any field can be overridden with an environment variable named STUDYRAG_<FIELD_NAME>,
e.g. STUDYRAG_LLM_BACKEND=openai or STUDYRAG_TOP_K=8.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    # --- where things live ---
    storage_dir: Path = PROJECT_ROOT / "storage"   # built index (gitignored)
    models_dir: Path = PROJECT_ROOT / "models"     # downloaded models (gitignored)

    # --- ingestion ---
    ocr: bool = True                    # read text from image-only slides / scanned pages

    # --- chunking ---
    chunk_strategy: str = "structure"   # "fixed" | "structure" | "semantic"
    chunk_size: int = 180               # target max words per chunk
    chunk_overlap: int = 30             # words of overlap (fixed strategy)
    min_chunk_words: int = 50           # structure strategy merges smaller neighbours

    # --- retrieval ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    candidates: int = 10                # results taken from each retriever before fusion (measured: D13)
    rrf_k: int = 60                     # Reciprocal Rank Fusion constant
    top_k: int = 4                      # chunks given to the LLM
    use_reranker: bool = True
    refusal_threshold: float = 0.2      # min reranker probability to attempt an answer (calibrated: D14)

    # --- generation ---
    llm_backend: str = "llamacpp"       # "llamacpp" (local, offline) | "openai" (any OpenAI-compatible API)
    llm_model_file: str = "Qwen3.5-2B-Q4_K_M.gguf"
    llm_context: int = 4096
    llm_threads: int = 0                # 0 = auto
    api_base_url: str = "https://api.openai.com/v1"
    api_model: str = "gpt-4o-mini"
    api_key: str = ""
    temperature: float = 0.2

    # --- verification ---
    support_threshold: float = 0.55     # min support score for a sentence to count as grounded

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        for f in fields(cls):
            raw = os.environ.get(f"STUDYRAG_{f.name.upper()}")
            if raw is None:
                continue
            current = getattr(s, f.name)
            if isinstance(current, bool):
                value = raw.strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int):
                value = int(raw)
            elif isinstance(current, float):
                value = float(raw)
            elif isinstance(current, Path):
                value = Path(raw)
            else:
                value = raw
            setattr(s, f.name, value)
        if not s.api_key:
            s.api_key = os.environ.get("OPENAI_API_KEY", "")
        return s

    @property
    def llm_model_path(self) -> Path:
        return self.models_dir / "llm" / self.llm_model_file
