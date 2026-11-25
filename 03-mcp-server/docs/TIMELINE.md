# studyrag-mcp: Project Timeline

How the MCP server was built, from the commit history (23–25 November 2026), with the problems met and how
they were solved.

## Where we started
- A StudyRAG library of the owner's lecture slides (project 1) that could only be used through its own app.
- Two decisions with the owner: **build it twice** (official SDK + the protocol from scratch), and serve the
  owner's own notes locally (public sample corpus for tests and screenshots).
- The same 2015 laptop; PyPI downloads at 30–80 KB/s.

## Where we ended
- A working MCP server with 8 read-only tools, 2 resources and 3 prompts, used from the owner's Claude Code.
- An MCP implementation from scratch that the official client accepts.
- 15 tests (protocol, server, end to end over stdio with real models); search in ~1 s; full documentation.

## Phase by phase

### Phase 1: Explore the SDK, build both servers (23 Nov, until 16:38)
**What we did:** installed `mcp` 2.3 (≈ 12 min on this connection), read its source, wrote the from-scratch server
(JSON-RPC over stdio), the service layer over StudyRAG and evalkit, the official-SDK server and 14 tests.

| Difficulty | How we solved it |
|---|---|
| The SDK was a new major version: `FastMCP` had been renamed `MCPServer`, and a new 2026-07-28 protocol added a `server/discover` probe. | Read the installed source instead of relying on memory. Made the scratch server answer `server/discover` with "method not found", which makes new clients fall back to the classic handshake. A conformance test proves it. |
| The full StudyRAG stack includes a 10-minute C++ compile (llama.cpp) the server doesn't need. | Installed only the search stack (embeddings, reranker, OCR, parsers). The server never generates text: the assistant is the model. |
| `check_answer` flagged a true sentence in the first manual test. | Not a bug: the test answer cited the wrong slide. The verifier caught a misattributed citation; documented as a feature. |

### Phase 2: Connect to Claude Code (23 Nov, 16:39–19:13)
| Difficulty | How we solved it |
|---|---|
| Claude Code reported "Connection closed". | The package wasn't installed in its own venv. The end-to-end test had passed because it set `PYTHONPATH`. Installed the package, and the test now launches the server with a clean environment, exactly like Claude Code. |
| The client config needs an absolute path on this laptop. | `.mcp.json` is gitignored; a template lives in `examples/`. |

### Phase 3: Real use (23 Nov, 19:13–19:55)
| Difficulty | How we solved it |
|---|---|
| "What is a market space?" returned unrelated slides with `found: false`. | Three documents (including Lecture 1) had been removed from the index in the StudyRAG app. Re-syncing restored them. The server was honest: it said "not found" instead of inventing an answer. |
| The running server kept serving its old copy of the index. | It now checks the index's timestamp on each call and reloads when the app or CLI syncs (the models stay loaded). Tested with a sync from "another process". |
| `python -m studyrag` failed in project 1's own environment. | StudyRAG was never installed as a package there; fixed, and project 1's setup guide updated. |

### Phase 4: Measure and document (25 Nov)
Latency benchmark over real stdio, a recorded demo session on the public corpus (the check catches a planted
sentence), concept notes, setup guide, decisions, results.

## Lessons
1. **Test the way users run it.** A test that sets up its own environment can hide exactly the bug users hit.
2. **Read the library you install.** Major versions change names and protocols.
3. **A long-running process must notice when its data changes.**
4. **"Not found" is a feature.** The first real failure looked like a bug in search; it was the server correctly
   refusing to answer from notes that weren't there.
