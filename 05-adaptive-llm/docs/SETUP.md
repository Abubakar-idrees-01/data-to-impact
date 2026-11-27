# Setup

Part A (the tiny GPT) needs Python 3.10+ and PyTorch's CPU build. No GPU is used or needed.

## 1. Install

**Linux / macOS**
```bash
cd 05-adaptive-llm
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu    # ~210 MB download
.venv/bin/pip install numpy pytest
```
On macOS the default PyPI build is fine: `.venv/bin/pip install torch numpy pytest`.

**Windows (PowerShell)**
```powershell
cd 05-adaptive-llm
py -3 -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install numpy pytest
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

## Troubleshooting
| problem | fix |
|---|---|
| `No space left on device` while installing torch | set `TMPDIR` and `PIP_CACHE_DIR` as above |
| `pip install torch` downloads a 2 GB+ CUDA build | use `--index-url https://download.pytorch.org/whl/cpu` |
| `curl: (33) HTTP server doesn't seem to support byte ranges` | download `TinyStoriesV2-GPT4-train.txt` in full, or a mirror that supports ranges |
| `ModuleNotFoundError: tinygpt` | run from `05-adaptive-llm/` with `PYTHONPATH=src` |
