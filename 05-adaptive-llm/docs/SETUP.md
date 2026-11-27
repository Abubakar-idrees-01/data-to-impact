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

## 4. Part B: what can this machine run?
```bash
PYTHONPATH=src .venv/bin/python -m adaptive.plan                     # profile + plan over project 1's GGUF models
PYTHONPATH=src .venv/bin/python -m adaptive.plan --models /path/to/ggufs --ctx 8192 --min-tok-s 8
```
Prints CPU, cores, SIMD flags, available RAM, GPU and measured memory bandwidth (~2 s), then for every `.gguf` in
the folder: memory needed at that context size, whether it fits, estimated speed, and the choice. Saves
`results/hardware/<cpu>.json`. Run it with other programs closed: a busy CPU lowers the measured bandwidth.
The default model folder is `../01-rag-from-scratch/models/llm/` (see that project's SETUP for the downloads).

## Troubleshooting
| problem | fix |
|---|---|
| `No space left on device` while installing torch | set `TMPDIR` and `PIP_CACHE_DIR` as above |
| `pip install torch` downloads a 2 GB+ CUDA build | use `--index-url https://download.pytorch.org/whl/cpu` |
| `curl: (33) HTTP server doesn't seem to support byte ranges` | download `TinyStoriesV2-GPT4-train.txt` in full, or a mirror that supports ranges |
| `ModuleNotFoundError: tinygpt` / `adaptive` | run from `05-adaptive-llm/` with `PYTHONPATH=src` |
