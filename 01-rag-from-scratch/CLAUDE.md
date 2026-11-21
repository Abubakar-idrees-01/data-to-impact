# 01 — RAG from Scratch

## Goal
**Study assistant.** Upload your own study material (PDF, DOCX, legacy DOC via LibreOffice, PPTX, TXT, MD) and get answers grounded **only**
in them, with exact citations. If the answer isn't in the documents, the system says so.

## "From scratch" scope (build ourselves, to show understanding)
- Document parsing + chunking strategies (fixed, recursive, semantic) — compare them
- BM25 keyword search implemented by hand
- Vector index implemented by hand (NumPy brute-force cosine first; compare against FAISS)
- Hybrid retrieval with Reciprocal Rank Fusion
- Prompt assembly, citation tracking, refusal logic
Allowed off-the-shelf: embedding model (small sentence-transformers model), the LLM
(small ~1–3B open model, 4-bit GGUF via llama.cpp), PDF text extraction library.

## Stand-out features
- Sentence-level citations that link to the exact page/passage
- Answer verification: each claim checked against its cited chunk; unsupported claims flagged
- Confidence score + "not in your documents" refusal
- Reranking step (cross-encoder) with before/after metrics
- Incremental indexing (add/remove docs without rebuilding)
- Fully offline / private on a CPU laptop; Streamlit or Gradio UI; deploy to HF Spaces
- Evaluated with 02-llm-evals: retrieval recall@k, MRR, faithfulness, answer correctness

## Study features (decided 2026-11-21)
All grounded in the uploaded material and cited:
- Q&A with citations; summaries (whole doc / chapter / slide deck)
- Generate MCQs (4 options, answer + explanation + source), short-answer questions, flashcards
- Quiz mode: user answers, app grades and explains with the source passage
- Pluggable LLM backend: local GGUF model by default (offline, free); optional API key backend
  for faster/better generation on weak machines. Small local models are slow for long outputs
  (e.g. 10 MCQs), so stream output and generate in batches.
- Owner's test files go in `data/my_docs/` (gitignored). RAG indexes them; no model training.

## Decisions
- Owner learns after the full build → build the complete project, docs written alongside code.
- Owner will verify Phase 6 eval questions; Claude builds the eval set + metrics and must
  report measured accuracy/correctness (retrieval, faithfulness, answer and MCQ quality).
- Deployment (HF Spaces) is optional, decide later.

## Web app / demo
- Streamlit app: document library sidebar (upload/remove), chat, clickable citations that open the
  source passage, confidence badge, "show retrieval details" panel for learning/debugging.
- Same app runs locally (one command) and on Hugging Face Spaces (free CPU) as the public demo.
- Public demo uses openly licensed documents only (e.g. OpenStax textbooks, arXiv papers);
  owner's private files stay local and gitignored.

## Status (2026-11-21)
Built: full package `src/studyrag`, Streamlit app, CLI, OCR, eval harness, docs (SETUP, HOW_IT_WORKS,
11 concept notes, DECISIONS, RESULTS). 22 unit tests pass.
- Owner's lectures (3 PPTX, 150 slides) are image-only -> OCR (RapidOCR, use_cls=False) is essential.
- Default LLM Qwen3.5-2B Q4_K_M (~37 s/answer on the owner's i5-5200U). Prefill ~25 tok/s, decode ~5.7 tok/s.
- Private eval set: eval/datasets/private/my_docs_qa.jsonl (35 q, gitignored); owner to verify in Phase 6.
- Remaining: model comparison (qwen2.5-1.5b, qwen3.5-0.8b downloaded), screenshots, final README numbers.

## Environment gotchas
- venv at `.venv` on NTFS; PyPI downloads are very slow (~30-80 KB/s), Hugging Face is fast (~1 MB/s).
- llama-cpp-python compiled with CMAKE_ARGS GGML_NATIVE=OFF + AVX2/FMA/F16C.
- Shell is zsh: use ${=var} to word-split variables in loops.
