# 3. Designing tools for a language model

## Intuition
A tool's user is a model. It decides from the **name, description and schema** alone whether to call the tool and
with what arguments, then reads the result as text. So a good MCP tool is designed like an API for a smart
reader who has never seen your code and who believes what it's told.

## The choices in this server ([`server.py`](../../src/studyrag_mcp/server.py))
**1. No LLM inside the server.** The client already is a strong language model. If the server generated answers
too (with project 1's local 2B model), we'd add 30 seconds and a weaker writer. Instead the server does what the
model can't do alone: search the user's private files, and check claims against them.

**2. Retrieval with explicit "found / not found".** `search_notes` returns `found: false` when the best passage is
below project 1's calibrated relevance threshold, plus a `guidance` line: *"say the notes don't cover this rather
than answering from general knowledge"*. A model handed weak passages will otherwise improvise.

**3. Citations designed for the answer.** Each passage carries `n`, a human-readable `citation` ("Digital
Marketing Lec 1 — Slide 4: Market Space") and a `chunk_id`. The model cites `[n]` in its answer and can pass the
`chunk_id`s back to `check_answer`.

**4. A self-check loop.** `check_answer` scores each sentence against the passages it cites (project 1's verifier).
Search, then write, then check, then fix: the model can catch its own unsupported sentence. In testing it flagged
a deliberately invented sentence ("invented in 1850 by Napoleon", support 0.0) and a true sentence cited to the
wrong slide.

**5. Schemas that reject bad calls early.** Type hints plus `pydantic.Field` constraints become the JSON Schema
the model sees (`top_k` between 1 and 10, non-empty strings). Bad arguments fail before any work, with a message.

**6. Errors written for the model.** An unknown document returns *"no document matches 'Lec 9'. Available: Course
Outline, Digital Marketing Lec 1, ..."*, so the model can retry correctly in the same turn.

**7. Read-only annotations.** Every tool is marked `readOnlyHint=True, destructiveHint=False, idempotentHint=True`,
so clients can call them without confirmation.

**8. Instructions.** The server's `instructions` field tells the model the workflow (search → answer only from
passages → cite → check). Hosts include it in the model's context.

## Tools, resources or prompts?
- Things the model should decide to do mid-conversation → **tools** (search, check).
- Data a user might attach to a chat → **resources** (a whole lecture's text).
- Workflows a user starts on purpose → **prompts** ("quiz me", "revise this lecture").

## Interview questions
- *Your tool returns 50 KB of text. Problem?* It fills the model's context and costs tokens. Return the top-k passages
  and let the model ask for more (`get_passage`).
- *How do you stop a model from answering beyond its sources?* Return an explicit "not found" signal and guidance,
  and give it a way to verify claims.
- *Why put constraints in the schema rather than checking in code?* The model sees the schema before calling, so
  it makes fewer bad calls; the code still validates, because schemas are advice to the model, not a guarantee.
