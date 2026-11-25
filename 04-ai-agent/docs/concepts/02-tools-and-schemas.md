# 2. Tools: schemas, validation and errors the model can fix

## Intuition
A tool is a Python function with a contract the model can read: a name, a one-line description and a JSON Schema
for its arguments. The model reads the description to decide *when* to use it and the schema to decide *how*.
So tool design is prompt design: a vague description gets vague use.

## One gate for every call
All tool calls go through `Registry.call` ([tools.py](../../src/scout/tools.py)), which:
1. checks the tool exists (else: "no tool named X. Available: ...");
2. parses and **validates the arguments** against the schema: types, required fields, enums, bounds, unknown keys;
3. if the tool is **outward**, asks the human first (concept 3);
4. runs it; an expected failure (`ToolError`) or even a crash becomes a message, not an exception;
5. caps the output (8,000 characters) so one result can't eat the context.

Errors go **back to the model** as the tool's result, worded so it can fix them:
```
error: a must be a integer, got str
error: `old` must appear exactly once in README.md (found 0 times); copy it exactly
error: you cited [7, 8] without reading them; web_read them first or remove those claims
```
In our live digests the model read those messages and corrected itself on the next step.

## Errors the provider catches before we do
Groq validates tool calls server-side. When gpt-oss produced a call to a tool that didn't exist (`web_read`, when
ours was called `read_page`), Groq rejected the whole reply with `400 tool_use_failed`. Sampling again didn't help.
Two fixes: rename the tool to what the model expects (`web_search` / `web_read` as a pair), and on that error retry
with a note saying what was wrong and listing the real tool names. **Name tools the way the model already thinks.**

## MCP tools are just tools
[mcp_tools.py](../../src/scout/mcp_tools.py) connects to an MCP server (project 3's studyrag-mcp), asks for its tool
list, and wraps each one as a `Tool` whose function forwards the call. The schema comes from the server, so nothing is
hard-coded. One safety rule: a tool the server doesn't mark `readOnlyHint` is treated as **outward** and needs approval.

## Interview questions
- *Why validate arguments if the provider already uses the schema?* Providers check syntax, not your rules (bounds,
  existing files, unique matches), and local models may not check at all. And the error message is the model's only
  way to learn what went wrong.
- *A tool fails. Raise or return?* Return a clear message to the model; raise only for bugs in your own loop.
- *How do you design a tool for an LLM?* One clear job, a description that says when to use it, small typed inputs,
  outputs that include what the model needs next (e.g. source numbers to cite), errors that say how to fix the call.
