# 1. The agent loop

## Intuition
A chatbot answers from what it already knows. An **agent** can also *act*: search the web, read a file, call an API.
The trick is simple: the model doesn't run anything itself. It writes a structured request ("call `web_search` with
`{"query": "MCP spec"}`"), our program runs it, and we hand the result back. Repeat until the model says it's done.

That loop, *think → act → observe → repeat*, is the whole idea. People call it ReAct (reason + act) or "tool use".

## How tool calling works
We send the model a list of tools, each with a name, a description and a JSON Schema for its arguments:
```json
{"type": "function", "function": {"name": "web_search", "description": "Search the web...",
  "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
```
Instead of text, the model can reply with a **tool call**:
```json
{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "web_search", "arguments": "{\"query\": \"MCP spec\"}"}}]}
```
We run it and append the result as a `tool` message with the same `id`, then call the model again with the whole
conversation. When it replies with plain text and no tool calls, that's the answer.

## The algorithm (our `agent.py`)
```
messages = [system prompt, task]
for step in 1..max_steps:
    reply = model(messages, tools)
    if reply has no tool calls: return reply.text          # done
    for each call: result = registry.call(name, args)       # validated, approval-gated, capped
                   messages.append(result)
    if tokens used > budget: break
    if 75% of budget used: tell the model to finish        # so it can still save its work
ask the model, with no tools, for a final answer from what it has
```

Three stopping rules keep it from running forever or costing a fortune:
- **max_steps**: model calls per task (8 for questions, 12 for routines);
- **token budget**: 40k per question, 60k per routine; a warning at 75% so the model can wrap up;
- **finalize**: if a limit is hit, one last call *without tools* forces an answer instead of a crash.

Two details from our live runs:
- **The same call twice** (same tool, same arguments) is answered from memory instead of running it again. *But not if
  it failed*: an error can depend on state ("you cited a source you haven't read"), so a retry after fixing it must run.
  Our first version cached errors too and blocked exactly that retry.
- **Every step is traced** as one JSON line (`.scout/traces/*.jsonl`): which model, what it asked for, what came back,
  tokens and seconds. Debugging an agent without traces is guesswork.

## Why build it from scratch?
Frameworks (LangChain, LlamaIndex agents, ...) wrap this loop. Writing it ourselves takes ~150 lines and makes every
behaviour visible: we can show exactly where approval is enforced, how context is trimmed, why it stopped.

## Interview questions
- *What is an AI agent?* A model in a loop that can call tools, observe the results and decide the next step, with
  limits on steps, tokens and what the tools may do.
- *How does the model "call" a function?* It outputs a structured tool call (name + JSON arguments); the application
  executes it and returns the result as a message. The model never runs code itself.
- *How do you stop an agent looping forever?* Step limit, token budget, repeat-call detection, and a forced final
  answer when a limit is reached.
