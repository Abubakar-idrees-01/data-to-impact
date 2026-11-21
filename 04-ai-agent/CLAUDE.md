# 04 — AI Agent

## Goal
A personal agent that solves real problems for the owner: researches questions online,
reads sources, and produces answers/solutions; helps with GitHub and LinkedIn work.

## Capabilities
- Web research: search → read pages → synthesize answer with sources
- Planning + tool use loop, memory of past tasks, uses 03-mcp-server tools
- GitHub: scan owner's repos for bugs/outdated deps/missing docs and open PRs **for approval**;
  find good-first-issues in projects the owner cares about and draft real contributions
- LinkedIn: draft replies and posts for the owner to review (no autonomous sending)

## Hard guardrails (decided with owner)
- **Human approval required for every outward action** (PRs, comments, messages, posts).
- No automated random commits, no mass/auto commenting on others' repos, no LinkedIn
  scraping/auto-messaging — violates GitHub/LinkedIn terms, risks account bans, and reads as
  spam to recruiters. Visibility comes from genuine, reviewed contributions.

## Evaluation
Task success rate, steps per task, cost/latency — via 02-llm-evals.

## Status
Not started.
