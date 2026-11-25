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

## Status (2026-11-25): complete (publishing open)
- 8 read-only tools, 2 resources, 3 prompts; from-scratch MCP (scratch.py) conformance-tested with the official client;
  index auto-reload; 15 tests (stdio e2e with real models skips in CI). CI: .github/workflows/mcp-server.yml.
- Measured (docs/benchmark.json): handshake 1.8 s, first call 14 s (model load), search 0.94 s, check_answer 1.26 s.
- Docs final: README (real demo session, docs/DEMO.md + images/demo.png), SETUP, HOW_IT_WORKS, 4 concepts,
  DECISIONS D1-D10, RESULTS, OVERVIEW, TIMELINE, 7 PDFs (build with project 1's venv: scripts/build_pdfs.py).
- Owner uses it from Claude Code via workspace-root .mcp.json (gitignored).
- Open (owner's call): PyPI publishing. Blocker: depends on `studyrag` (project 1), which isn't on PyPI.
