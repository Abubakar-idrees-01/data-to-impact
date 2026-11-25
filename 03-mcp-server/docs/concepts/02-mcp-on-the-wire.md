# 2. MCP on the wire: the protocol, built from scratch

[`scratch.py`](../../src/studyrag_mcp/scratch.py) implements MCP's core in ~250 lines of standard-library Python,
and the official SDK's client connects to it in our tests. This note follows one real session.

## JSON-RPC 2.0 in one minute
Every message is a JSON object with `"jsonrpc": "2.0"`:
- a **request** has an `id` and a `method` (and usually `params`), and gets exactly one reply with the same `id`;
- a **response** has the `id` and either `result` or `error` (`{"code", "message"}`);
- a **notification** has a `method` but **no `id`**, and gets **no reply**.

On the stdio transport each message is **one line**: no newlines inside, `\n` at the end. stdout carries only
protocol messages; anything else (logs, a stray `print`) goes to stderr, or the client fails to parse the stream.

## A session, line by line
```text
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25",
   "capabilities":{},"clientInfo":{"name":"claude-code","version":"..."}}}
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25",
   "capabilities":{"tools":{"listChanged":false},"resources":{...},"prompts":{...}},
   "serverInfo":{"name":"scratch-demo","version":"0.1.0"},"instructions":"..."}}
→ {"jsonrpc":"2.0","method":"notifications/initialized"}            (no id: no reply)
→ {"jsonrpc":"2.0","id":2,"method":"tools/list"}
← {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"add","description":"Add two integers.",
   "inputSchema":{"type":"object","required":["a","b"],"properties":{"a":{"type":"integer"},...}}}]}}
→ {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add","arguments":{"a":20,"b":22}}}
← {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"sum\": 42}"}],
   "structuredContent":{"sum":42},"isError":false}}
```

## Version negotiation
The client offers the newest version it speaks. If the server supports it, it answers with the same one; if not,
it answers with its own newest, and the client decides whether it can continue. Versions are dated:
2024-11-05, 2025-03-26, 2025-06-18, 2025-11-25, and **2026-07-28**.

2026-07-28 adds a **stateless** mode: instead of the `initialize` handshake, a client can probe with
`server/discover`. The official 2.x client tries that first and, on any error, falls back to the classic
handshake. Our scratch server answers `server/discover` with "method not found" (-32601), which triggers that
fallback, so a 2026 client still works with a 2025-style server.

## Two kinds of failure (important)
| situation | how it's reported | why |
|---|---|---|
| Malformed JSON, unknown method, missing/invalid arguments, unknown tool | **JSON-RPC error** (`-32700`, `-32601`, `-32602`…) | The *request* was wrong; the tool never ran |
| The tool ran and failed (document not found, division by zero) | **Normal result with `"isError": true`** and a message | The *model* should see the failure and correct itself, e.g. retry with another document name |

The official SDK follows the same rule: our server raises `ToolError("no document matches 'Lec 9'. Available:
...")`, and the model reads that text.

## What the SDK adds over our scratch version
Full JSON Schema validation (via pydantic), automatic schemas from Python type hints, Streamable HTTP and SSE
transports, sessions, progress and logging notifications, cancellation, pagination, resource templates and
subscriptions, auth, and the 2026 stateless protocol. That's why the real server uses it, and why the scratch
version exists only to learn from.

## Interview questions
- *Why is a failing tool not a JSON-RPC error?* Protocol errors are for the client software; tool errors are for
  the model, which can recover. Mixing them hides useful feedback from the model.
- *What breaks if a stdio server prints a debug line?* The client reads stdout as protocol messages, gets invalid
  JSON, and the connection fails.
- *How does a new client talk to an old server?* Version negotiation in `initialize`, or the `server/discover`
  probe with fallback.
