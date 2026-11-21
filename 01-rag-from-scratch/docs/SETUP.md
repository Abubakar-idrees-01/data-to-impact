# Setup Guide

This guide takes you from a fresh clone to a running study assistant. Everything runs on a CPU;
no GPU and no paid API are required.

**You need:** Python 3.10+ (3.12 tested), about **3 GB of free disk** (models plus dependencies), 8 GB RAM
recommended (4 GB works with the smallest model), and an internet connection for the one-time downloads.

---

## 1. Get the code

```bash
git clone https://github.com/Abubakar-idrees-01/data-to-impact.git
cd data-to-impact/01-rag-from-scratch
```

## 2. Create a virtual environment

<details open><summary><b>Linux / macOS</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```
</details>

<details><summary><b>Windows (PowerShell)</b></summary>

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
</details>

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

`llama-cpp-python` (the local LLM engine) ships as source code and is **compiled on your machine**.
That takes 5–15 minutes and needs a C++ compiler:

| OS | Compiler |
|---|---|
| Ubuntu/Debian | `sudo apt install build-essential cmake` |
| macOS | `xcode-select --install` |
| Windows | "Desktop development with C++" from the Visual Studio Build Tools installer |

**Old CPU or compile errors?** Build with portable flags (this is what we use on a 2015 i5):
```bash
CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" pip install --no-cache-dir llama-cpp-python
```
CPUs without AVX2 (very old or low-power chips) should use `-DGGML_AVX2=OFF -DGGML_FMA=OFF`.

**Don't want a local model at all?** Skip `llama-cpp-python` and use the API backend (step 6).

## 4. Download the models

```bash
python scripts/download_models.py            # ~1.5 GB total, resumable
python scripts/download_models.py --list     # other model sizes
```

| Model | Size | Role |
|---|---|---|
| `Qwen3.5-2B-Q4_K_M.gguf` | 1.3 GB | Writes answers, MCQs and summaries (default) |
| `BAAI/bge-small-en-v1.5` | 130 MB | Embeddings for semantic search |
| `Xenova/ms-marco-MiniLM-L-6-v2` | 90 MB | Cross-encoder reranker |
| RapidOCR (inside the pip package) | ~15 MB | Reads text from image-only slides |

On a weak machine (4 GB RAM), use the 0.8B model:
```bash
python scripts/download_models.py --llm qwen3.5-0.8b
export STUDYRAG_LLM_MODEL_FILE=Qwen3.5-0.8B-Q4_K_M.gguf      # Windows: $env:STUDYRAG_LLM_MODEL_FILE="..."
```

## 5. Add your documents and run

Put PDFs, slides (PPTX), Word files (DOCX), notes (TXT/MD) or photos (PNG/JPG) in `data/my_docs/`.
That folder is gitignored, so your files never leave your computer.

```bash
streamlit run app/streamlit_app.py
```
The app opens at http://localhost:8501. Click **Sync data/my_docs folder** in the sidebar, or upload files there.

There is also a command-line interface:
```bash
python -m studyrag sync data/my_docs
python -m studyrag ask "What is a market space?"
python -m studyrag mcq -n 5 --doc "Lec 1"
python -m studyrag search "porter five forces" --mode bm25      # retrieval only, shows the ranking trace
```
(Run these from the project folder with the venv active. `pip install -e .` also gives you a `studyrag` command.)

**No documents of your own yet?** Fetch the public sample corpus of Wikipedia articles on digital marketing:
```bash
python scripts/fetch_sample_docs.py
python -m studyrag sync data/sample_docs
```

## 6. Optional: use an API instead of the local model

Faster and higher quality on weak machines, but your question and the retrieved passages are sent to the provider.
Any OpenAI-compatible endpoint works. In the app, open **Settings → Language model → API**, or set:

```bash
export STUDYRAG_LLM_BACKEND=openai
export STUDYRAG_API_BASE_URL=https://api.groq.com/openai/v1     # or OpenAI, OpenRouter, a local Ollama...
export STUDYRAG_API_MODEL=<model name from your provider>
export STUDYRAG_API_KEY=<your key>
```

## 7. Run the tests and the evaluation

```bash
pytest                                        # unit tests: fast, no models needed
python eval/run_eval.py retrieval --set sample
python eval/run_eval.py answers --set sample  # end-to-end: slow on CPU (about 1 min per question)
```
See [RESULTS.md](RESULTS.md) for the published numbers.

---

## Configuration reference

Every setting in [`src/studyrag/config.py`](../src/studyrag/config.py) can be overridden with an environment
variable `STUDYRAG_<NAME>`:

| Variable | Default | Meaning |
|---|---|---|
| `STUDYRAG_CHUNK_STRATEGY` | `structure` | `fixed`, `structure` or `semantic` |
| `STUDYRAG_TOP_K` | `4` | passages given to the LLM per answer |
| `STUDYRAG_USE_RERANKER` | `true` | cross-encoder reranking on/off |
| `STUDYRAG_REFUSAL_THRESHOLD` | `0.02` | min reranker relevance to attempt an answer |
| `STUDYRAG_OCR` | `true` | OCR image-only slides and scanned pages |
| `STUDYRAG_LLM_THREADS` | `0` (auto) | CPU threads for the LLM |
| `STUDYRAG_STORAGE_DIR` | `storage/` | where the index is saved |

Changing the chunking, embedding model or OCR settings triggers an automatic re-index on next start.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model file not found` | Run `python scripts/download_models.py`, or check `STUDYRAG_LLM_MODEL_FILE` |
| `llama-cpp-python` build fails | Install the compiler (step 3), retry with the portable `CMAKE_ARGS` |
| `Illegal instruction` when loading the model | Your CPU lacks AVX2: rebuild with `-DGGML_AVX2=OFF -DGGML_FMA=OFF` |
| Slides indexed with 0 words | They are images: make sure `rapidocr` is installed (`pip install rapidocr`) and `STUDYRAG_OCR=true` |
| Answers are very slow | Use the 0.8B model, lower `STUDYRAG_TOP_K` to 3, or use the API backend |
| `.doc` / `.ppt` files fail | Install LibreOffice (used to convert legacy Office formats) |
| Downloads keep failing | Re-run the same command: downloads resume where they stopped |
