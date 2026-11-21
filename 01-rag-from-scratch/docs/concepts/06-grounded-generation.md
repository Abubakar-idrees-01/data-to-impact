# 6. Grounded generation: prompts, citations and refusal

## The goal
An LLM is a fluent writer with an unreliable memory. In RAG we use it only as the **writer**: the facts
come from the retrieved passages, and the prompt forces it to show where each fact came from.

## The prompt ([`prompts.py`](../../src/studyrag/prompts.py))
```
SYSTEM: You are a study assistant. Answer the student's question using ONLY the numbered sources.
        - After each sentence that uses a source, cite it ... like [1] or [2][3].
        - Use only facts stated in the sources. Do not add outside knowledge.
        - If the sources do not contain the answer, reply with exactly: NOT_FOUND
USER:   Sources:
        [1] (Digital Marketing Lec 1 — Slide 4: Online Market space)
        The market space in marketing is defined as a virtual market place...
        [2] (...)
        Question: What is a market space?
```

Design choices for **small** models (2B parameters follows instructions less reliably than 100B+):
- **Numbered sources with headers** make citing easy and meaningful.
- **A fixed refusal token** (`NOT_FOUND`) is easier for a small model to produce than a polite paragraph,
  and trivial for us to detect. The app then shows a friendly message.
- **Low temperature (0.2)**: we want faithful, not creative.
- **Top 4 passages only**: every extra passage costs prompt-processing time on CPU (see
  [local LLMs](09-local-llms.md)) and gives the model more chances to wander.

## Two layers of refusal
1. **Before the LLM**: the reranker says nothing is relevant, so we refuse instantly (see [reranking](05-reranking.md)).
2. **By the LLM**: the passages are related but don't contain the answer, so the model replies `NOT_FOUND`.

## Streaming
On a CPU the model writes about 5 tokens per second. Showing tokens as they are generated
(`stream=True` → `st.write_stream`) makes a 20-second answer feel responsive. To support this, the QA flow is
split into `prepare()` (retrieve + build prompt) → stream → `finalize()` (parse citations, verify, score
confidence) in [`answer.py`](../../src/studyrag/answer.py).

## Interview questions
- *How do you stop hallucinations?* You can't fully. You reduce them (grounding prompt, low temperature,
  refusal paths) and **detect** what remains (→ [verification](07-verification.md)).
- *Why put sources before the question?* The model reads the evidence with the task already framed by the
  system prompt, and the question sits closest to where it starts writing.
- *Lost in the middle?* LLMs attend most to the start and end of long contexts. With 4 short passages this is
  minor, and ordering by relevance puts the best evidence first.
