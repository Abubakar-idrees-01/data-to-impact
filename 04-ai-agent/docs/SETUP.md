# Setup guide

Scout needs **Python 3.11+**, a model API key (Groq's free tier works) and a Tavily key for web search. No GPU.
Optional: project 3's MCP server, for `scout ask --notes`.

## 1. Install
```bash
cd data-to-impact/04-ai-agent
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt           # httpx + the MCP SDK
pip install --no-deps -e .                # the `scout` command
pip install pytest && python -m pytest -q # optional: 40+ tests, no keys or network needed
```

## 2. Keys
Create `04-ai-agent/.env` (gitignored, never committed):
```
SCOUT_API_BASE_URL=https://api.groq.com/openai/v1
SCOUT_API_MODEL=openai/gpt-oss-120b
SCOUT_API_KEY=gsk_...            # console.groq.com -> API Keys (free)
TAVILY_API_KEY=tvly-...          # app.tavily.com (free: ~1,000 searches a month)
```
Type the keys into the file yourself; don't paste them into chats or issues. On Linux/macOS: `chmod 600 .env`.

Any OpenAI-compatible API works. The model must support tool calling: for example OpenAI, OpenRouter, or a local
llama.cpp / Ollama server (`SCOUT_API_BASE_URL=http://localhost:8080/v1`, no key needed).

Optional settings:
| variable | default | |
|---|---|---|
| `SCOUT_FALLBACK_MODEL` | `openai/gpt-oss-20b` | used when the main model's daily quota runs out (empty = stop instead) |
| `SCOUT_CONTEXT_LIMIT` | `6000` | max tokens per request (Groq free tier: 8,000 per minute) |
| `SCOUT_TOPICS` | RAG; LLM evals; MCP; small LLMs on CPU | digest topics, separated by `;` |
| `SCOUT_DIGEST_MAX_STEPS` | `16` | model calls a digest may make before it must stop |
| `SCOUT_DIGEST_DOMAINS` | arxiv.org, github.com, ... | where digests search first (comma-separated) |
| `SCOUT_JOURNAL_DIR` | `<repo>/learning-log` | where notes go (must be inside a git repo) |
| `SCOUT_WORKSPACE` | the repo root | the repo `scout checkup` reads |
| `SCOUT_MCP_COMMAND` | project 3's venv | command that starts an MCP server, for `--notes` |

## 3. Use it
```bash
scout ask "What changed in the latest MCP specification?"      # web research with sources
scout ask --notes "Explain backlinks from my lecture"           # your StudyRAG notes + the web
scout digest                 # research one recent development, show the draft note, ask before committing
scout digest --prepare       # draft only, for later review
scout checkup                # check the repo's docs, propose fixes, ask before committing
scout review                 # approve or decline prepared drafts
scout history                # what it did recently
```
Every command that would change something shows you the exact diff first. **Enter means no.**

## 4. Daily drafts (optional, Linux)
A systemd user timer can prepare a note every morning at 09:30. It **never pushes**; drafts wait for `scout review`.
```bash
bash scripts/install_timer.sh          # install and start the timer
systemctl --user list-timers           # check it
systemctl --user disable --now scout-digest.timer    # remove it
```
macOS/Windows: schedule `scout digest --prepare --quiet` with launchd or Task Scheduler, from this folder.

## 5. Evaluations
```bash
bash scripts/get_data.sh                                  # SimpleQA (MIT) into data/raw/ (gitignored)
setsid nohup bash experiments/pipeline.sh > logs/pipeline.log 2>&1 < /dev/null & disown
setsid nohup bash experiments/pipeline_digest.sh >> logs/pipeline_digest.log 2>&1 < /dev/null & disown   # E4
```
Long runs must be launched detached like this: they take hours on a free tier and must survive the terminal closing.
They resume where they stopped, and wait when the daily quota runs out. Grading uses project 2 (`02-llm-evals/.env`).

## Troubleshooting
| problem | fix |
|---|---|
| `set SCOUT_API_KEY` | create `.env` as in step 2, in the `04-ai-agent` folder |
| `daily token limit reached` | wait for the time shown, or set `SCOUT_FALLBACK_MODEL` to a model with quota left |
| `API 429, waiting ...` lines | normal on a free tier (8,000 tokens/minute); it waits and continues |
| `search failed (401)` | Tavily key missing, wrong, or out of credits |
| `--notes`: `no StudyRAG MCP server found` | set up project 3, or `pip install studyrag-mcp` and set `SCOUT_MCP_COMMAND=studyrag-mcp` |
| `declined automatically, no terminal to ask in` | you ran an approving command without a terminal (e.g. from a timer): use `--prepare` |
| `... changed since this was proposed` | a file changed after the draft was made; run the task again |
| push failed | the commit is kept locally; `git push` when you're online |
