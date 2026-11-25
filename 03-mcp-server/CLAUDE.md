# 03 — MCP Server

## Goal
A Model Context Protocol (MCP) server that lets any MCP client (Claude, IDEs, our own agent)
use our tools. Packaged and published so others can install it.

## Planned tools
- `search_docs` / `ask_docs` — exposes the 01-rag knowledge base
- Local productivity tools (e.g. file search, git repo stats/summaries)
- Resources: list indexed documents; prompts: reusable templates

## Stand-out
- Proper input validation, error handling, tests, and docs
- Published to PyPI; listed in an MCP server directory
- Used by 04-ai-agent as its tool layer

## Decisions (2026-11-23, with owner)
- Build "both": the real server on the official SDK (`mcp` 2.3: `mcp.server.mcpserver.MCPServer`; FastMCP was
  renamed in 2.x, read the installed source before using any API) + a from-scratch JSON-RPC/stdio implementation
  (`scratch.py`) for learning, conformance-tested with the official client.
- Default library: owner's own notes (project 1's storage/), locally; public sample index for tests/screenshots.
- Tools are read-only and call no LLM: the client (Claude) is the model. No API key needed.

## Status (2026-11-25)
- Built: 8 tools (search_notes, list_documents, document_outline, get_passage, check_answer, list_eval_runs,
  eval_run_summary, compare_eval_runs), 2 resources, 3 prompts; path-safe eval access; index auto-reload when
  the app/CLI syncs. 15 tests incl. a stdio end-to-end test with real models (no PYTHONPATH: package must be
  installed, `pip install --no-deps -e .`).
- Connected to the owner's Claude Code via workspace-root `.mcp.json` (gitignored; template in examples/).
- TODO: docs (README, SETUP, HOW_IT_WORKS, concepts incl. "MCP on the wire" from scratch.py, DECISIONS,
  RESULTS e.g. latency, TIMELINE, OVERVIEW, PDFs), demo screenshot/transcript, CI, PyPI publish (ask owner).
