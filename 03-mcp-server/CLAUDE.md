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

## Status
Not started.
