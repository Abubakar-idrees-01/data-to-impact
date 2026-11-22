"""Export StudyRAG's public evaluation questions with the exact passages its answers were generated from.

Project 1's result files store answers and scores but not the passage texts. Retrieval is deterministic, so
re-running it on the same index with the same settings reproduces the sources the model saw.

Run with project 1's environment (it imports studyrag):
    ../01-rag-from-scratch/.venv/bin/python experiments/studyrag_export.py
Writes data/studyrag_public.jsonl (gitignored: derived from the public Wikipedia sample corpus).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RAG = HERE.parent / "01-rag-from-scratch"
sys.path.insert(0, str(RAG / "src"))
sys.path.insert(0, str(RAG / "eval"))

from run_eval import SETS, build_library, models  # noqa: E402
from studyrag.config import Settings  # noqa: E402
from studyrag.retrieval import Retriever  # noqa: E402


def main() -> None:
    settings = Settings()
    data_path, docs_dir, _ = SETS["sample"]
    questions = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    emb, rr = models(settings)
    lib = build_library(settings, emb, docs_dir, f"sample-{settings.chunk_strategy}")
    retriever = Retriever(lib, emb, rr, settings)
    out = HERE / "data" / "studyrag_public.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for q in questions:
            res = retriever.retrieve(q["question"])
            passages = [f"[{n}] {h.chunk.header}\n{h.chunk.text}" for n, h in enumerate(res.hits, 1)]
            f.write(json.dumps({"id": q["id"], "question": q["question"], "answer": q.get("answer"),
                                "key_facts": q.get("key_facts", []), "answerable": q["answerable"],
                                "passages": passages, "top_relevance": res.top_relevance},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(questions)} questions -> {out}")


if __name__ == "__main__":
    main()
