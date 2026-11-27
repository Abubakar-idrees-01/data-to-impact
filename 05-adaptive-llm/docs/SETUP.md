# Setup

Part A (the tiny GPT) needs Python 3.10+ and PyTorch's CPU build. No GPU is used or needed.

## 1. Install

**Linux / macOS**
```bash
cd 05-adaptive-llm
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # CPU-only torch (~210 MB download), numpy, pytest
```
On macOS, drop the `+cpu` suffix from the torch line (the normal PyPI build is the right one there).

**Windows (PowerShell)**
```powershell
cd 05-adaptive-llm
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Little space on your system drive?** pip downloads and unpacks into your home folder and `/tmp`. Point both at a
drive with room (the project's `.tmp/` is gitignored):
```bash
export TMPDIR="$PWD/.tmp" PIP_CACHE_DIR="$PWD/.tmp/pip-cache"; mkdir -p .tmp
```
After installing, `.tmp/` can be deleted.

## 2. Data
```bash
bash scripts/get_data.sh                                  # Windows: run it in Git Bash, or download the two files by hand
PYTHONPATH=src .venv/bin/python -m tinygpt.data           # Windows: $env:PYTHONPATH="src"; .venv\Scripts\python -m tinygpt.data
```
- `get_data.sh` fetches TinyStories V2 from Hugging Face: the 22 MB validation file and the **first 200 MB** of the
  2.2 GB training file (`TRAIN_MB=400 bash scripts/get_data.sh` for more). Files land in `data/raw/` (gitignored).
- `tinygpt.data` trains the tokenizer on the first 20 MB of stories (seconds), then encodes everything into
  `data/tokens/*.bin` (~1 minute, ~115 MB). Options: `--vocab 4096 --sample-mb 20`.

## 3. Check
```bash
.venv/bin/python -m pytest -q
```

## 4. Part B: the llama.cpp engine
Part B runs real open models with `llama-cpp-python`, which **compiles llama.cpp on your machine** (~10 minutes on
a 2-core laptop; needs a C++ compiler and cmake: `sudo apt install build-essential cmake` on Ubuntu, Xcode command
line tools on macOS, "Desktop development with C++" from Visual Studio Build Tools on Windows).
```bash
CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" .venv/bin/pip install -r requirements-engine.txt
# Apple Silicon: plain `pip install -r requirements-engine.txt` builds with Metal.  No AVX2: drop the CMAKE_ARGS.
```
**Already built it for project 1?** The build is relocatable (its libraries find each other next to themselves), so
it can be copied instead of compiled again:
```bash
S=../01-rag-from-scratch/.venv/lib/python3.12/site-packages
cp -r $S/llama_cpp $S/llama_cpp_python-0.3.37.dist-info .venv/lib/python3.12/site-packages/
.venv/bin/pip install diskcache==5.6.3 && .venv/bin/pip check
```
Models: the three GGUF files from project 1 (`../01-rag-from-scratch/models/llm/`, downloaded by its
`scripts/download_models.py`). Any ChatML-format GGUF works (Qwen, many others).

## 5. Part B: what can this machine run?
```bash
PYTHONPATH=src .venv/bin/python -m adaptive.plan                     # profile + plan over project 1's GGUF models
PYTHONPATH=src .venv/bin/python -m adaptive.plan --models /path/to/ggufs --ctx 8192 --min-tok-s 8
```
Prints CPU, cores, SIMD flags, available RAM, GPU and measured memory bandwidth (~2 s), then for every `.gguf` in
the folder: memory needed at that context size, whether it fits, estimated speed, and the choice. Saves
`results/hardware/<cpu>.json`. Run it with other programs closed: a busy CPU lowers the measured bandwidth.
The default model folder is `../01-rag-from-scratch/models/llm/` (see that project's SETUP for the downloads).

Measure the models for real and calibrate the planner's speed estimate (~10 minutes, close other programs):
```bash
PYTHONPATH=src .venv/bin/python -m adaptive.bench_models
```

## 6. Part B: routing evaluation
```bash
bash scripts/get_eval_data.sh                                        # GSM8K test set, 0.7 MB
PYTHONPATH=src .venv/bin/python -m adaptive.route_eval run           # every model × 128 questions (~1 h, resumable)
PYTHONPATH=src .venv/bin/python -m adaptive.route_eval summary       # accuracy and seconds per policy
PYTHONPATH=src .venv/bin/python -m adaptive.route_eval run --limit 1 --tag smoke --only "Qwen3.5-0.8B*"   # 1-minute check
```

## 7. Part B: speculative decoding
```bash
PYTHONPATH=src .venv/bin/python -m adaptive.speculative --draft Qwen3.5-0.8B --target Qwen3.5-2B --k 2 4 6
```
Draft and target must share a tokenizer (same model family). Prints target-alone speed, speculative speed, draft
acceptance rate and whether the output is identical to the target's own greedy output.

## Troubleshooting
| problem | fix |
|---|---|
| `No space left on device` while installing torch | set `TMPDIR` and `PIP_CACHE_DIR` as above |
| `pip install torch` downloads a 2 GB+ CUDA build | use `--index-url https://download.pytorch.org/whl/cpu` |
| `curl: (33) HTTP server doesn't seem to support byte ranges` | download `TinyStoriesV2-GPT4-train.txt` in full, or a mirror that supports ranges |
| `ModuleNotFoundError: llama_cpp` | step 4 |
| `ModuleNotFoundError: tinygpt` / `adaptive` | run from `05-adaptive-llm/` with `PYTHONPATH=src` |
