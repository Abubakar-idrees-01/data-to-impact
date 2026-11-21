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

*Measured afterwards* ([RESULTS.md §6](RESULTS.md)): against Qwen2.5-1.5B and Qwen3.5-0.8B, the 2B model is the
slowest (5.7 vs 14.2 and 18.9 tok/s; its hybrid attention is less optimized in llama.cpp on CPU). On clean text all
three answer about equally well. On the owner's OCR'd slides, the 2B's own citations were supported 98% of the
time vs 80% for the 1.5B, and it was mostly correct on 96% vs 89%. It stays the default; the faster models
are documented alternatives.

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
impossible. Shuffling guards against position bias, which is common in small models. *Measured:* Qwen3.5-2B
turned out fairly balanced (raw positions A 4, B 5, C 3, D 3 over 15 questions), so shuffling here is a cheap safeguard
rather than a fix. Shuffling did create a bug, though: see D20.

### D11. Streamlit for the UI
**Considered:** Gradio, FastAPI + React.
**Why:** a pure-Python multi-tab app with sidebar, chat, streaming, forms and downloads in one file; runs locally
with one command and deploys free to Hugging Face Spaces. A custom frontend would add a second language and
build step without improving what this project is about.

### D12. Gold labels as locations, not chunk IDs
**Why:** labelling "the answer is on Lec 1, slide 4" stays valid when chunking changes, so one dataset can fairly
compare every chunking strategy.

## Decisions changed by measurement

### D13. Rerank only the top 10 fused candidates (was 20)
**Measured:** 20 → 10 candidates halved reranking latency (2.1 s → 1.0 s per query) and *raised* hit@1 slightly
(0.91 → 0.94); hit@3 stayed 1.00. The cross-encoder costs ~100 ms per pair on this CPU.

### D14. Refusal threshold 0.2 (was a guessed 0.02)
**Measured:** answerable questions' best reranker score was 0.80–1.00, unanswerable ones ≤ 0.065. The initial 0.02
let 1 in 8 unanswerable questions through to the LLM; anything from 0.1 to 0.5 separates them perfectly.
0.2 sits in the middle of the gap (log scale).

### D15. Auto-citation of verified but uncited sentences
**Observed** in the first end-to-end run: answers were correct, but the 2B model often dropped `[n]` markers when
writing bullet lists, so the student saw facts with no source. **Fix:** (1) the prompt now requires a citation on
every bullet; (2) when the verifier finds an uncited sentence *is* supported by a source (support ≥ 0.55), the
system appends that citation itself. Unsupported sentences never get one, and stay flagged.

### D16. OCR without the text-direction classifier
**Measured on the 150 real slides:** the classifier flipped 3.4% of lines upside down, producing gibberish; with it
disabled, 0.0% garbled lines and ~15% faster OCR. Slides and scans are upright by nature.
See [concepts/08-ocr.md](concepts/08-ocr.md).

### D17. One shared OCR cache
Every index (the app's, and one per chunking strategy in the evals) reuses `storage/ocr_cache.json`, so OCR runs
once per unique image, ever. Without this, evaluating three chunking strategies would have re-OCR'd 150 slides three times.

### D18. Linearize tables before indexing
**Found on real data:** the owner's course outline stores the grading scheme in a table. Retrieval ranked it #1
for "how much is the final exam worth?", but the reranker gave it 0.005, so the system *refused*. That was the only
false refusal among 63 answerable questions. Pipe-separated rows read as noise to a model trained on prose.
**Decision:** each table row becomes a sentence with its column headers ("Final examination — Weight: 40%; …").
Reranker relevance for "weightage of the final exam" went from 0.02 to 0.80. Three formats were tested and the best kept.
**Considered:** lowering the refusal threshold (would let unanswerable questions through, whose scores reach
0.065); adding a BM25-score rule (it would have separated this one case, but tuning a rule on one example is overfitting).

### D19. A refusal shows the closest passages
A refusal is a threshold decision, so it can be wrong. Instead of a dead end, the student sees the two closest
passages with their text. A false refusal then costs one click, not a wrong conclusion that "it isn't in my notes".

### D20. Remap option letters in MCQ explanations after shuffling
**Found by reviewing generated MCQs by hand:** explanations say "Option C is correct because…", but shuffling had
moved that option to D, so the student saw a contradiction. Some questions also repeated "A) … B) …" inside the
stem. **Fix:** remap every letter reference with the shuffle permutation, strip inline options, and ask the model
to refer to options by content. Unit-tested in `tests/test_pipeline.py`.

### D21. MCQs are self-checked before the student sees them
**Found in the human review:** about 1 in 5 generated questions had two defensible options (e.g. a distractor
that the passage also supports), and the model's explanation then contradicted itself. **Decision:** after
generating a question, the model answers it from the passage like a student (one short, grammar-constrained call
at temperature 0). A disagreement triggers one regeneration; if it persists, the question is kept with a visible
warning instead of being silently dropped. **Cost:** one extra short call per question (the passage is already
short). Measured effect in [RESULTS.md §7](RESULTS.md).

<!-- MEASURED_DECISIONS -->
