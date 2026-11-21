# 01 — RAG from Scratch

## Goal
Upload your own documents (PDF, DOCX, TXT, MD, web pages) and get answers grounded **only**
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

## Web app / demo
- Streamlit app: document library sidebar (upload/remove), chat, clickable citations that open the
  source passage, confidence badge, "show retrieval details" panel for learning/debugging.
- Same app runs locally (one command) and on Hugging Face Spaces (free CPU) as the public demo.
- Public demo uses openly licensed documents only (e.g. OpenStax textbooks, arXiv papers);
  owner's private files stay local and gitignored.

## Status
Not started.
