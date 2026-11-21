# 📚 StudyRAG: a private study assistant that only answers from *your* material

Upload your lecture slides, PDFs and notes. Ask questions, get **summaries**, and practise with
**MCQs, short-answer questions and flashcards**. Every answer cites the exact slide or page it came from,
and when the answer isn't in your material it says so instead of making something up.

Runs **fully offline on a 2015 laptop CPU** (no GPU, no API key), with an optional API backend for speed.

<!-- RESULTS_SUMMARY -->

## Why this project exists
Students get course material as slide decks and PDFs, and general chatbots answer from the internet, not from
*their* syllabus. They confidently invent things and can't point to the slide an exam question will come from.
StudyRAG answers only from what your teacher actually gave you, and shows its sources.

The first real test exposed a problem that most "chat with your PDF" demos ignore: **all 150 slides of the
three test lectures were images**, with zero extractable text. So StudyRAG includes an OCR stage that reads
image-only slides, scanned PDFs and photos of notes.

## Features
| | |
|---|---|
| 💬 **Grounded Q&A** | Answers with clickable citations (`Lec 2 — Slide 14`) and a streaming response |
| 🛑 **Honest refusal** | "Not in your documents" when nothing relevant is found: decided *before* calling the LLM |
| ✅ **Claim verification** | Every answer sentence is checked against its cited source; weak claims are flagged |
| 🎯 **Confidence badge** | Combines retrieval relevance with the verified-sentence ratio |
| 📝 **Summaries** | Instant outline from slide titles + AI summary (map-reduce for long lectures) |
| ❓ **MCQ practice** | 4 options, explanation and source per question; exam mode with score; answer-position bias removed |
| ✍️ **Short answers** | Type an answer and the AI grades it against the model answer and source passage |
| 🃏 **Flashcards** | Exportable to Anki |
| 📷 **OCR** | Image-only slides, scanned PDFs, PNG/JPG photos (RapidOCR, cached) |
| 🔍 **How I found this** | Each answer's full retrieval trace: BM25 rank, vector rank, fusion score, reranker score |
| 📂 **Formats** | PDF, PPTX, DOCX, TXT, MD, images, plus legacy DOC/PPT via LibreOffice |
| ⚡ **Incremental index** | Adding a file only processes that file; OCR results are cached |

## Built from scratch vs. off-the-shelf
| Built in this repo | Used off-the-shelf |
|---|---|
| Chunking (fixed / structure-aware / semantic) | File parsers (python-pptx, python-docx, pypdf) |
| BM25 with inverted index, tokenizer and stemmer | Embedding model (bge-small, via fastembed/ONNX) |
| Vector search (NumPy cosine) | Cross-encoder reranker (MiniLM, via fastembed) |
| Reciprocal Rank Fusion | LLM runtime (llama.cpp) + model (Qwen3.5-2B, 4-bit) |
| Prompting, citation parsing, refusal logic, confidence | OCR models (RapidOCR) |
| Claim verifier, MCQ validation and de-biasing | Web UI framework (Streamlit) |
| Persistent incremental index, OCR layout reconstruction | |
| Evaluation harness and labelled datasets | |

## How it works
```
files ─► load + OCR ─► chunk ─► BM25 index + vector index
question ─► BM25 top-20 ┐
            vector top-20 ┴► Reciprocal Rank Fusion ─► cross-encoder rerank ─► relevant? ──no──► "not in your documents"
                                                                                  │yes
                                         local LLM with numbered sources ◄────────┘
                                                      │
                                    answer + [citations] ─► verify each sentence ─► confidence
```
Full walkthrough: **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**

## Quickstart
```bash
cd 01-rag-from-scratch
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # compiles llama.cpp: 5-15 min
python scripts/download_models.py                     # ~1.5 GB, resumable
streamlit run app/streamlit_app.py
```
Put your files in `data/my_docs/` (never pushed to git) and click **Sync**. Full guide, Windows/macOS notes and
troubleshooting: **[docs/SETUP.md](docs/SETUP.md)**

## Results
<!-- RESULTS_TABLES -->
All numbers come from `eval/run_eval.py` and are explained in **[docs/RESULTS.md](docs/RESULTS.md)**. Design choices
backed by these measurements are in **[docs/DECISIONS.md](docs/DECISIONS.md)**.

## Learn how it works
Concept notes, written to be understood and explained in an interview:

1. [Chunking](docs/concepts/01-chunking.md) · 2. [BM25](docs/concepts/02-bm25.md) · 3. [Embeddings & vector search](docs/concepts/03-embeddings-and-vector-search.md) ·
4. [Hybrid search & RRF](docs/concepts/04-hybrid-search-rrf.md) · 5. [Reranking](docs/concepts/05-reranking.md) ·
6. [Grounded generation](docs/concepts/06-grounded-generation.md) · 7. [Verification](docs/concepts/07-verification.md) ·
8. [OCR](docs/concepts/08-ocr.md) · 9. [Local LLMs on CPU](docs/concepts/09-local-llms.md) ·
10. [Question generation](docs/concepts/10-question-generation.md) · 11. [Evaluation](docs/concepts/11-evaluation.md)

## Project layout
```
src/studyrag/   the library (ingest, chunking, bm25, vector_index, hybrid, retrieval, llm, answer, verify, study)
app/            Streamlit web app
eval/           labelled datasets + evaluation harness
tests/          unit + pipeline tests (fake models, run in seconds)
scripts/        model download, sample corpus fetch
docs/           setup, architecture, concepts, decisions, results
```

## Limitations
- Small local models are slow on old CPUs (see the speed results) and weaker than large cloud models at nuanced
  questions; the API backend exists for that trade-off.
- The claim verifier is a fast heuristic (word overlap + semantic similarity), not logical entailment.
- OCR handles printed text well; handwriting and complex diagrams are out of scope.
- English only (embedding model, stemmer and stopwords).

## License
Code: MIT. Sample corpus: Wikipedia articles, CC BY-SA 4.0 (attribution in each file).
