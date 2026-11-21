"""Download every model StudyRAG needs into ./models (resumable).

    python scripts/download_models.py                  # default chat model + embedder + reranker
    python scripts/download_models.py --llm qwen3.5-0.8b   # smaller/faster chat model for weak machines
    python scripts/download_models.py --list

Then point the app at a different chat model with:  STUDYRAG_LLM_MODEL_FILE=<file name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# name -> (Hugging Face repo, file, approx size, note)
LLMS = {
    "qwen3.5-2b": ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", "1.3 GB", "default: best quality/speed balance on CPU"),
    "qwen3.5-0.8b": ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf", "0.5 GB", "fastest; weaker at MCQs"),
    "qwen3.5-4b": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", "2.7 GB", "best quality; slow without a GPU"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf", "1.1 GB", "older, widely supported"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="qwen3.5-2b", choices=list(LLMS))
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        for k, (repo, f, size, note) in LLMS.items():
            print(f"{k:<14} {size:>7}  {f:<38} {note}")
        return

    from studyrag.config import Settings
    s = Settings.from_env()

    print("1/2  Embedding + reranker models (~150 MB, via fastembed)...")
    from studyrag.embeddings import Embedder, Reranker
    Embedder(s.embedding_model, s.models_dir / "fastembed")
    Reranker(s.reranker_model, s.models_dir / "fastembed")
    print("     ok")

    if a.skip_llm:
        return
    repo, fname, size, _ = LLMS[a.llm]
    print(f"2/2  Chat model {fname} ({size}) from {repo} — resumable if interrupted...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo, filename=fname, local_dir=str(s.models_dir / "llm"))
    print(f"     saved to {path}")
    if fname != s.llm_model_file:
        print(f"\nThis is not the default model. Use it with:\n  export STUDYRAG_LLM_MODEL_FILE={fname}")


if __name__ == "__main__":
    main()
