# Setup Guide

The server needs **Python 3.11+**, project 1 (StudyRAG) with an index and its search models, and an MCP-capable
assistant. No GPU, no API key.

## 0. The short way: install from PyPI
If you only want to use the server (not work on its code), skip steps 1–2:
```bash
python3 -m venv ~/.venvs/studyrag && source ~/.venvs/studyrag/bin/activate   # Windows: ...\Scripts\Activate.ps1
pip install studyrag-mcp           # pulls in studyrag (project 1) and the MCP SDK
studyrag sync ~/my-notes           # index your files; the search models (≈150 MB) download on first run
```
Your index is kept in `~/.studyrag` (override with `STUDYRAG_HOME`; the evaluation tools also need
project 2 installed from a clone, `pip install -e 02-llm-evals`, and `STUDYRAG_MCP_EVAL_RESULTS`). In step 3 use
`"command": "/ABSOLUTE/PATH/TO/.venvs/studyrag/bin/studyrag-mcp"` with no `args`. If you set `STUDYRAG_HOME`, pass it
to the server too, with `"env": {"STUDYRAG_HOME": "/path"}`: assistants start servers with a minimal environment.

## 1. Set up StudyRAG first (working from a clone)
Follow [project 1's setup guide](../../01-rag-from-scratch/docs/SETUP.md) far enough to have its models downloaded
(`python scripts/download_models.py`) and your documents indexed (put files in `01-rag-from-scratch/data/my_docs/`,
then click **Sync** in the app, or run `python -m studyrag sync data/my_docs`). The local LLM is **not** needed.

## 2. Install the server
```bash
cd data-to-impact/03-mcp-server
python3 -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt                 # MCP SDK + StudyRAG's search stack (no llama.cpp, no Streamlit)
pip install --no-deps -e .                      # the server itself: `studyrag-mcp` / `python -m studyrag_mcp`
pip install -r requirements-dev.txt && python -m pytest -q    # optional: 16 tests
```
On a slow connection the first install takes a while (≈ 12 minutes at 30–80 KB/s); packages already in pip's cache
from project 1 are reused.

## 3. Connect your assistant
The server speaks MCP over stdio, so the assistant launches it. Use the **absolute path** to this venv's Python.

**Claude Code** (VS Code extension or CLI): create `.mcp.json` in the folder you open (the workspace root):
```json
{
  "mcpServers": {
    "studyrag": {
      "type": "stdio",
      "command": "/ABSOLUTE/PATH/TO/03-mcp-server/.venv/bin/python",
      "args": ["-m", "studyrag_mcp"]
    }
  }
}
```
Reload the window (`Ctrl+Shift+P` → *Developer: Reload Window*), approve the **studyrag** server when asked, and
check it with `/mcp` (8 tools). A template is in [`examples/mcp.json`](../examples/mcp.json).

**Claude Desktop** (Windows/macOS): add the same `"studyrag"` entry under `mcpServers` in
`claude_desktop_config.json` (Settings → Developer → Edit Config), then restart the app. On Windows the command is
`C:\\path\\to\\03-mcp-server\\.venv\\Scripts\\python.exe`.

**Other clients** (Cursor, Windsurf, VS Code's built-in MCP support…) take the same command and arguments.

**Over HTTP** instead of stdio: `studyrag-mcp --http --port 8765`, then connect the client to
`http://127.0.0.1:8765/mcp`.

## 4. Use it
Ask in a new chat:
- *"Using my notes, what is a market space? Cite the slides."*
- *"Quiz me on Lec 2, three questions."* (or pick the **quiz_me** prompt)
- *"Compare the StudyRAG v1 and v2 evaluation runs. Is v2 really better?"*

The first tool call loads the search models (≈ 15 s on a 2015 laptop); later calls take about 1 s.

## Options
| environment variable | default | meaning |
|---|---|---|
| `STUDYRAG_STORAGE_DIR` | `01-rag-from-scratch/storage` | which StudyRAG index to serve (e.g. the public sample index) |
| `STUDYRAG_MCP_EVAL_RESULTS` | `02-llm-evals/results` | where evaluation runs are read from |
| any `STUDYRAG_*` | see project 1 | e.g. `STUDYRAG_TOP_K`, `STUDYRAG_REFUSAL_THRESHOLD` |

Set them in the `"env": {...}` block of the client config.

## Troubleshooting
| problem | fix |
|---|---|
| `/mcp` shows *failed* / "Connection closed" | Run the configured command by hand: `/path/.venv/bin/python -m studyrag_mcp`. "No module named studyrag_mcp" means step 2's `pip install --no-deps -e .` was skipped |
| A document you added isn't found | Sync it in StudyRAG first (app button or `python -m studyrag sync data/my_docs`); the server picks the change up on its next call |
| A document disappeared | It was removed in the app (the ✕ button). Sync the folder again |
| The first answer is slow | Model loading, once per server start (≈ 15 s) |
| Search says `found: false` for something you know is there | Try other wording, or `get_passage` for the slide; relevance is checked against project 1's calibrated threshold (0.2) |
