# Design Decisions

Each decision records **what** we chose, **what else** we considered, and **why**: with measurements where
the choice was empirical. Numbers refer to [RESULTS.md](RESULTS.md).

---

### D1. Target hardware: a 2015 laptop CPU
**Context:** Intel i5-5200U (2 cores / 4 threads, AVX2), no GPU, 15 GB RAM, slow and unstable internet
(~80 KB/s from PyPI during development).
**Decision:** everything must run on this machine, offline after a one-time download.
**Why:** most students don't have GPUs; a demo that only runs on cloud GPUs proves little about engineering
under constraints. This one constraint drives most of the decisions below.

### D2. No PyTorch: fastembed (ONNX Runtime) for embeddings and reranking
**Considered:** sentence-transformers (PyTorch).
**Decision:** fastembed runs the same models through ONNX Runtime.
**Why:** PyTorch CPU wheels are ~200 MB to download and ~1 GB installed, and our only use for them is inference
of two small models. ONNX Runtime is ~15 MB, starts faster and is quick on CPU. On a slow connection, that is
the difference between minutes and hours of setup.

### D3. Local LLM: Qwen3.5-2B, Q4_K_M, via llama.cpp
**Considered:** Qwen2.5-1.5B/3B, Llama-3.2-3B, Gemma-3-4B, Qwen3.5-0.8B/4B; Ollama as the runtime.
**Decision:** Qwen3.5-2B (released Feb 2026) quantized to Q4_K_M (1.3 GB), run with `llama-cpp-python`.
**Why:**
- 4B-class models were measured by the owner as too slow on this CPU; 2B is the largest that keeps answers
  interactive (see the speed table in RESULTS.md).
- The newest small generation beats older 3B models at instruction following, which matters for citation discipline.
- `llama-cpp-python` is pip-installable and embeds in our process: no separate server to install (Ollama would
  need a system install), and we get grammar-constrained JSON decoding.
- `0.8B` and `4B` remain one command away (`scripts/download_models.py --llm ...`) for weaker or stronger machines.

### D4. Thinking mode disabled
**Decision:** pre-fill an empty `<think></think>` block (what the official template does for `enable_thinking=False`).
**Why:** on this CPU, hidden reasoning would add tens of seconds to every answer before the first visible word.
Grounded QA over 4 short passages rarely needs multi-step reasoning.

### D5. Brute-force NumPy vector search instead of FAISS/HNSW
**Why:** a study library has hundreds to a few thousand chunks. Exact search over 10k × 384 floats is a single
matrix multiply of a few milliseconds, with 100% recall and zero extra dependencies. Approximate indexes start
paying off at roughly a million vectors. See the scaling measurement in RESULTS.md.

### D6. Hybrid retrieval (BM25 + vectors with RRF) + cross-encoder reranking
**Why:** course material is full of exact terms (CPC, ROAS, "Porter's Five Forces") where BM25 shines, and of
paraphrased questions where embeddings shine. RRF merges the two without score calibration; the reranker then
re-orders the top 20 precisely. *Measured:* see the retrieval grid in RESULTS.md.

### D7. Structure-aware chunking as the default
**Why:** slides and sections are already "one idea" units made by the author. Citations stay precise
("Slide 4", not "words 1,200–1,380"). *Measured* against fixed-window and semantic chunking: see RESULTS.md.

### D8. OCR with RapidOCR, triggered automatically
**Context:** all 150 slides of the real test lectures were images; plain extraction yielded 0 words.
**Considered:** Tesseract (needs an OS-level install, which is awkward on Windows and impossible without admin
rights); cloud OCR (breaks privacy); a vision LLM (far too slow on CPU).
**Decision:** RapidOCR (PaddleOCR models on ONNX Runtime, pip-only), run when a slide/page has fewer than 8
extractable words; results cached by image hash.

### D9. Refusal decided before calling the LLM
**Why:** if nothing relevant was retrieved, a 2B model asked to answer "only from the sources" will often still
try. Refusing on the reranker score is both more reliable and instant. The threshold is calibrated on
answerable vs unanswerable questions (RESULTS.md).

### D10. One passage per generated question, grammar-constrained JSON, shuffled options
**Why:** short prompts are fast on CPU; every item gets an exact source; grammar constraints make invalid JSON
impossible; shuffling removes small models' strong "answer is A" bias (measured in RESULTS.md).

### D11. Streamlit for the UI
**Considered:** Gradio, FastAPI + React.
**Why:** a pure-Python multi-tab app with sidebar, chat, streaming, forms and downloads in one file; runs locally
with one command and deploys free to Hugging Face Spaces. A custom frontend would add a second language and
build step without improving what this project is about.

### D12. Gold labels as locations, not chunk IDs
**Why:** labelling "the answer is on Lec 1, slide 4" stays valid when chunking changes, so one dataset can fairly
compare every chunking strategy.

<!-- MEASURED_DECISIONS -->
