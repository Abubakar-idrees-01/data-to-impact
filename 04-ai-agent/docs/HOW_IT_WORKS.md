# How Scout works

## Architecture
```
                                   you (terminal)
                       scout ask │ digest │ checkup │ review        ▲ approve? [y/N]
                                 ▼                                   │
 ┌──────────────────────────────────────────────────────────────────┴───────────────┐
 │ routines.py   digest · checkup · review          builds the task, collects a ChangeSet │
 │ agent.py      the loop: model → tool calls → results → model ... → answer             │
 │               step limit · token budget (+75% warning) · repeat-call memory · traces   │
 │ context.py    keeps each request under the provider's limit (trims old tool results)   │
 │ llm.py        OpenAI-compatible client: tool calling, 429 waits, daily-quota fallback  │
 │ tools.py      Registry: schema validation · APPROVAL GATE for outward tools · caps     │
 └───────┬───────────────────┬──────────────────────┬────────────────────┬─────────────┘
         │                   │                      │                    │
   web.py (Tavily)     mcp_tools.py           repo.py               changes.py
   web_search          studyrag-mcp           list/read/search      propose_edit
   web_read            (project 3) over       git_log, check_links  write_note (digest)
   SSRF guard,         stdio: search_notes,   git-tracked files     ChangeSet → one diff →
   focused passages    get_passage, ...       only                  approve → write, commit, push
         │                   │
    the internet       your StudyRAG index (project 1)
```

## One digest, end to end
`scout digest` (what you'd see, slightly shortened):

1. **The task is built** ([routines.py](../src/scout/routines.py)): today's date, what "recent" means, your topics,
   the titles of earlier notes (from task memory and `learning-log/README.md`) so it picks something new, and the
   required note format.
2. **Tools for this task**: `web_search` (restricted to the last month and to primary-source sites first),
   `web_read`, and `write_note`, which drafts into a `ChangeSet` and checks the note's citations.
3. **The loop** ([agent.py](../src/scout/agent.py)), each line one model call:
   ```
   → web_search(query='Model Context Protocol specification revision')
   → web_read(source='3', question='what changed in the 2026-07-28 revision')
   → web_search(query='MCP release cycle 2026')
   → web_read(source='7', question='release cadence')
   → write_note(title=..., body=...)        ✗ cite at least two different sources you read
   → web_read(source='2', ...)
   → write_note(title=..., body=...)        ✓ drafted
   ```
   Before each call, [context.py](../src/scout/context.py) checks the request fits (≤ 6,000 tokens on Groq's free
   tier) and trims the oldest page texts if not. Rate-limit replies (429) are waited out; a daily-quota reply
   switches to the fallback model.
4. **The note** is assembled by code, not the model: title, a line saying it was drafted by Scout, the model's body
   with `[n]` citations, and a **Sources** list generated from the pages it actually read. The log's index gets a new
   line at the top.
5. **Approval**: you see one diff (the new note + the index line) and the commit message
   `Learning log: <title>`. `y` writes both files, commits only those two paths, and pushes. Anything else
   discards it. With `--prepare` it's saved to `.scout/pending/` for `scout review` instead.
6. **Memory**: the run is recorded in `.scout/memory.jsonl` (title, sources, tokens, outcome); the full trace is in
   `.scout/traces/`.

## `scout ask --notes`: the agent as an MCP client
With `--notes`, Scout launches project 3's server (`studyrag-mcp`) as a subprocess and speaks MCP to it over stdio:
`initialize`, `tools/list` (8 tools, all marked read-only), then `tools/call` whenever the model uses one. Your notes
are searched locally by StudyRAG; only the passages it returns go to the model. The agent cites notes by their slide
or page and web sources by number, and when the notes don't cover something it says so (in our test it searched,
found relevance 0.0, said the lecture doesn't cover the topic, and answered that part from the web).

## `scout checkup`
A deterministic check runs first (relative markdown links pointing to missing files, ignoring code blocks), and its
findings go into the task. The agent then reads files, verifies each problem against its source of truth (counts the
tests, reads `pyproject.toml`) and proposes at most 3 fixes with `propose_edit`, limited to `*.md` files. Each
proposal needs an exact, unique snippet to replace, so it can't make vague edits. You approve all of them at once.

## Where things live
| | |
|---|---|
| `04-ai-agent/.env` | API keys (gitignored) |
| `learning-log/` (repo root) | the notes you approved |
| `.scout/traces/` | one JSONL file per run (gitignored) |
| `.scout/memory.jsonl` | task history (gitignored) |
| `.scout/pending/` | proposals waiting for `scout review` (gitignored) |
| `experiments/`, `results/` | the evaluations and their saved runs |
