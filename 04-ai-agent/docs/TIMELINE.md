# Scout: Project Timeline

How Scout went from an empty folder to a research agent with an approval gate, four evaluations and a live trace
viewer: what was built, in what order, what went wrong, and how it was solved. Times come from the commit history.

## Where we started
- A scaffolded `04-ai-agent` folder (21 November 2026) and a goal: an agent that does **real, reviewed work** (research
  with citations, a daily learning-log note, doc checkups) where **every outward action needs the owner's approval**.
- One request declined at the start: a bot adding N commits a day to keep the GitHub graph green. That is fake activity
  and against the repo's own rule. The agreed alternative: at most one real, reviewed piece of work a day.
- The same laptop as projects 1–3 (2015 CPU, no GPU), Groq's free tier (8,000 tokens/minute, 200k tokens/day per model)
  and Tavily's free search tier (~1,000 searches/month).

## Phase by phase

### Phase 1: The core (25 Nov, until 15:29)
**What we did:** built from scratch, with no agent framework: an async tool-calling client for any OpenAI-compatible API,
a tool registry that validates arguments against JSON Schema, an approval gate in `Registry.call`, and the agent loop
with a trace of every step.

| Decision | Why |
|---|---|
| Approval enforced in code, default no | A prompt can be overridden by a web page; code can't. |
| No LangChain or similar | Every step visible and explainable in an interview; two dependencies (httpx, mcp). |

### Phase 2: Tools (15:29–15:52)
**What we did:** web search and focused page reading (with an SSRF guard checking every redirect hop), project 3's
StudyRAG tools over MCP, repo tools that only see git-tracked files, change sets (drafted edits, one approval writes,
commits and pushes only those paths), and the routines: digest, checkup, review.

| Difficulty | How we solved it |
|---|---|
| gpt-oss called `web_read` when our tool was `read_page`; Groq rejected the whole reply (`400 tool_use_failed`). | Renamed the tools to what it expects (`web_search` / `web_read`); on that error, retry with a note listing the real names. |
| It cited as `【1†L13-L22】`, its training format, so no `[n]` was found. | Normalised to `[n]` in answers and tool arguments. |

### Phase 3: Live digest runs (15:52–16:25)
**What we did:** ran `scout digest` for real and fixed what broke.

| Difficulty | How we solved it |
|---|---|
| A request of 9,042 tokens hit Groq's 8,000 per-minute limit. | `context.py` trims old tool results to fit a 6,000-token window. |
| It anchored on 2024 and added the year to every search. | Today's date and the meaning of "recent" in the task; searches limited to the last month. |
| It based a note on one SEO blog. | Two different *read* sources required by `write_note`; primary-source domains searched first. |
| The budget ran out before it saved the note. | A warning at 75% of the budget, while it can still act. |
| The daily quota ran out mid-task. | A fallback model (gpt-oss-20b) for real use; experiments wait instead, so a result belongs to one model. |

### Phase 4: Evaluations designed (16:29–16:32)
**What we did:** E1 SimpleQA (closed-book vs agent, official sampling and grader, answer-key sites excluded), E2 a
prompt-injection suite on a fake internet (attempted vs executed, with and without the prompt rule), E3 a planted-error
repo for `checkup` with decoys.

### Phase 5: Docs and demo (16:33–16:45)
Concept notes 1–5, HOW_IT_WORKS, SETUP, a systemd timer that only *prepares* drafts, the live trace viewer, decisions
D1–D11 and the README.

| Difficulty | How we solved it |
|---|---|
| The first E2 fixture's pages called their compound "fictional", so the model refused to answer even the control questions. | Rewritten to read like a real page; the first run kept in `out/discarded/`, not mixed in. |
| A 20b checkup run's report described 4 fixes but proposed 1. What you'd approve wasn't what you read. | The summary shown before approval is now generated from the actual proposals. |

### Phase 6: The long runs (from 25 Nov, 16:45, unattended)
**What happened:** E2 and E3 finished on gpt-oss-20b; E1's closed-book half finished on gpt-oss-120b (40/40).

| Difficulty | How we solved it |
|---|---|
| The SimpleQA agent run hit the daily quota, waited, and was found dead at 23:31 (last log line 19:36). | Relaunched detached; the scripts resume, so nothing was lost. |
| One quota for everything would take days. | Each Groq model has its own quota: digests (E4) run on 20b while 120b does E1–E3, and the judge uses qwen. |
| The project had no `.gitignore` for judge and grader caches. | `.cache/` ignored. |
| The laptop shut down overnight; all three pipelines stopped. | Relaunched the next morning; every saved item was intact. |
| E4 run 4 (gpt-oss-20b) read 5 sources but used all 12 steps, then typed the note as its final answer instead of saving it with `write_note`. | Digest step limit raised to 16 as its own setting (`SCOUT_DIGEST_MAX_STEPS`). E4's 20b results file keeps 12, so one batch has one limit; the 120b round and real use get 16. |

E4, claim support in digest notes, added at 23:39: every snippet and passage the agent saw recorded per source, each
claim judged against it with project 2's validated judge, and a number-swap probe to check the judge on these notes.

*(Results and the rest of this phase are added as the runs finish.)*
