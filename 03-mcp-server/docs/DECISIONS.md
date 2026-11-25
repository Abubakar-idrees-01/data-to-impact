# Design Decisions

### D1. Official SDK for the server, plus a from-scratch implementation
**Decision (with the owner):** "both". The real server uses the official `mcp` SDK (2.3): it covers JSON Schema
validation, transports, the 2026 stateless protocol and edge cases we shouldn't reimplement. `scratch.py`
implements the protocol by hand (≈ 250 lines) to show what goes over the wire, and the official client connecting
to it is a conformance test. The scratch version is for learning, not production.

### D2. Read the installed SDK, not memory
The `mcp` 2.x SDK renamed `FastMCP` to `MCPServer`, added the 2026-07-28 protocol and a `server/discover` probe.
Every API used here was checked against the installed source; the from-scratch server was made to answer
`server/discover` with "method not found" so new clients fall back to the classic handshake.

### D3. No LLM inside the server
**Considered:** an `ask_notes` tool returning StudyRAG's own generated answer.
**Why not:** the caller is already a strong model. Project 1's local 2B model would add ~30 s and write worse
answers. The server adds what the model can't do on its own: search the user's private files (`search_notes`) and
check claims against them (`check_answer`). Result: no API key, and nothing slow in the loop.

### D4. An explicit "not found" signal
`search_notes` returns `found: false` (project 1's calibrated reranker threshold, 0.2) with guidance to say the
notes don't cover the question. Without it, a model handed weak passages tends to answer from general knowledge,
which defeats the point of answering from your own notes. Measured in the end-to-end test: an off-topic question
("boiling point of mercury") on the sample index returns `found: false`.

### D5. Thin MCP layer over plain services
All behaviour lives in `notes.py` and `evals.py`; `server.py` only maps them to MCP. The services are tested with
fakes in milliseconds, and project 4's agent can reuse them directly or through MCP.

### D6. Read-only, bounded, confined
Every tool is annotated read-only; inputs are bounded (`top_k` ≤ 10, answers ≤ 8,000 characters); evaluation runs
are resolved inside one directory, and traversal attempts are refused (tested). Syncing new documents stays a
deliberate user action in the app or CLI.

### D7. Lazy loading and automatic index reload
**Found in use:** the first version loaded StudyRAG at the first call and kept that index forever. After the
owner added a new PDF and re-synced in the app, the running server still served the old index. Now each call
compares the index manifest's modification time and reloads the chunks (the models stay loaded, so it's quick).
Lazy loading keeps the handshake instant (1.8 s including Python start-up) although the models take ≈ 14 s to load.

### D8. Test the server exactly as a client launches it
**Found in use:** the first end-to-end test passed while Claude Code failed with "Connection closed". The test had
passed `PYTHONPATH`, hiding that the package wasn't installed in its own venv. The test now launches
`python -m studyrag_mcp` with a clean environment, the way Claude Code does.

### D9. Expected failures are messages for the model
Unknown documents, out-of-range slides and bad run names raise `ToolError` with a message listing valid options,
so the model can correct its call in the same turn. Unexpected crashes are reported without internals.

### D10. Machine-specific client config stays out of git
`.mcp.json` contains an absolute path to the venv, so it's gitignored; `examples/mcp.json` is the template.
