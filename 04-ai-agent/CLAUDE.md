# 04 — AI Agent ("Scout")

## Goal
A personal research and developer agent that does real, reviewed work: researches questions online with citations,
uses the owner's notes through project 3's MCP server, drafts a daily learning-log note, and checks the repo's docs.
Everything outward (files, commits, pushes) needs the owner's approval.

## Hard guardrails (decided with owner)
- **Human approval required for every outward action**, enforced in code (`Registry.call` / `ChangeSet.approve_and_apply`),
  never only in the prompt. Default answer is no. No "approve all" mode for real use.
- **No filler commits.** The owner asked (2026-11-25) for a bot that adds N commits a day to keep the GitHub graph green;
  declined (fake activity, against the repo's own rule). The agreed alternative: one real, reviewed piece of work per
  day (learning-log note or doc fix); on days with nothing worth committing, no commit.
- No mass/auto commenting on others' repos, no LinkedIn scraping/auto-messaging.
- Repo tools only see git-tracked files (private docs and `.env` are invisible, never sent to the provider).

## Decisions (2026-11-25)
- Built from scratch (no LangChain etc.): `agent.py` loop, `tools.py` registry with JSON-Schema validation, `llm.py` async
  httpx client for OpenAI-compatible APIs with tool calling. Deps: httpx, mcp (client for studyrag-mcp).
- Model: Groq `openai/gpt-oss-120b` (reasoning_effort low), fallback `openai/gpt-oss-20b` when the daily quota runs out.
  Keys in `04-ai-agent/.env` (gitignored): SCOUT_API_BASE_URL / SCOUT_API_MODEL / SCOUT_API_KEY / TAVILY_API_KEY.
- Search: Tavily (free tier ~1,000 searches/month). Digests search recent (time_range=month) primary-source domains first.
- Groq free tier: 8,000 tokens/min means one request can't exceed ~8k tokens -> `context.py` trims old tool results
  (context_limit 6000 incl. max_tokens 1536); 200k tokens/day per model.
- Learning log lives in the monorepo at `learning-log/` (configurable SCOUT_JOURNAL_DIR). Notes are labelled as drafted
  by Scout and reviewed by the owner.
- Unattended runs only *prepare* proposals (`--prepare` -> `.scout/pending/`), the owner approves with `scout review`.

## Lessons from live runs (keep for docs)
gpt-oss invents tool names (web_read) -> renamed + retry with feedback; cites as 【n†L..】 -> normalized; anchors on
2024 -> date in task + time_range; cites one SEO blog -> 2+ read sources enforced + primary domains; request too large
(9,042 > 8,000) -> context trimming; budget ran out before saving -> 75% warning; TPD exhausted -> fallback model.

## Status (2026-11-25)
- Done: core loop, web tools, MCP client, repo tools, change sets, digest/checkup/review routines, CLI
  (`scout ask [--notes] | digest | checkup | review | history`), 41 tests (scripted model, no network).
- Next: evaluation (research QA accuracy vs closed-book, prompt-injection suite, digest claim support with evalkit's
  judge), trace viewer page (live demo), docs (README, SETUP, HOW_IT_WORKS, concepts, DECISIONS, RESULTS, TIMELINE,
  OVERVIEW, PDFs), daily scheduling (cron/systemd timer running `scout digest --prepare`).
