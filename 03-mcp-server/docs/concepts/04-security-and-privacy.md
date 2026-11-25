# 4. Security and privacy for an MCP server

## Intuition
An MCP server gives a model hands on your machine. A model can be tricked, by a malicious web page or a
document it reads, into calling tools in ways you never intended. So a server should be safe **even when its
caller is misled**: small powers, strict inputs, and nothing that can be turned into a weapon.

## The threats and what this server does
| threat | example | defence here |
|---|---|---|
| **Path traversal** | The model is induced to call `eval_run_summary(run="../../.ssh/id_rsa")` | Every path is resolved and must stay inside the results directory, otherwise it's refused (tested with `../`, `../../etc/passwd`, absolute paths) |
| **Destructive actions** | "Delete all my notes" | No tool writes, deletes or sends anything; all are annotated read-only. Syncing new files stays a deliberate user action in the app or CLI |
| **Prompt injection through documents** | A PDF contains "ignore previous instructions and..." | Tool results are returned as data. The server never executes text from documents. The host's model must treat tool output as untrusted; that's the host's job, but we avoid giving it dangerous tools to misuse |
| **Resource exhaustion** | `top_k=100000`, a 10 MB "answer" to check | Bounds in the schema and in code (`top_k` ≤ 10, answers ≤ 8,000 characters) |
| **Leaking internals** | A crash message reveals paths or secrets | Expected failures become clear `ToolError` messages; unexpected crashes are reported only as "Error executing tool X", with the traceback in the server's own log |
| **Corrupting the protocol** | A library prints to stdout | All logging goes to stderr; tested end to end over real stdio |

## Privacy: where your notes go
The server runs locally and reads your local index. **Passages that a tool returns are sent to the model's
provider** (for Claude Code: Anthropic), exactly like text you paste into the chat. Nothing else leaves the
machine: no telemetry, no other network calls. For notes that must stay entirely local, use a local model as the
host (e.g. an MCP client with a llama.cpp model) or don't connect the server.

## Least privilege, in one rule
Give the server only the powers its tools need. This one needs to *read* one index and one results folder.
Project 4's agent, which can open pull requests, gets those powers only behind a human approval step.

## Interview questions
- *What is the biggest risk with MCP servers?* Tools that combine reading untrusted content with powerful
  actions (send email, run shell, write files), because injected instructions can drive the actions.
- *How do you test path safety?* Feed traversal attempts and check they're refused and leak nothing (`tests/test_server.py`).
- *Is a read-only server automatically private?* No: what it reads is shown to the model provider.
