# 4. Context, tokens and rate limits

## Intuition
A model has no memory between calls. Each call sends the **whole conversation** again: system prompt, task, every tool
call and every result. So an agent's requests grow with every step, and so does the cost: 8 steps with a 5,000-token
conversation is 40,000 tokens, even if each step only adds a little.

## Three limits we hit, and what we did
**1. One request was too big.** Groq's free tier allows 8,000 tokens per minute, so one request can't exceed that.
After reading four pages, our first live digest sent 9,042 tokens and was refused.
*Fix: trim old tool results* ([context.py](../../src/scout/context.py)). Before each call we estimate the request
size; if it won't fit, the **oldest** tool results are cut to their first 300 characters (for a web page: its number,
title and URL), keeping the newest results whole and never touching the model's own messages. The characters-per-token
ratio is re-measured after every call from the real token count the API reports, so the estimate tracks the tokenizer.
If the provider still says "too large", the limit is lowered and the call retried.

**2. A task ran out of budget one step before saving.** A digest hit its 40k-token budget while fetching a second
source, so the note was never written. *Fix:* a bigger budget for routines (60k), and a warning at 75% so the model
finishes the required step while it still can.

**3. The daily quota ran out.** 200,000 tokens per model per day. Waiting a minute and retrying (fine for per-minute
limits) is pointless for a daily one: our first version wasted 9 minutes doing that. *Fix:* recognise a daily-limit
error, switch to a **fallback model with its own quota** (gpt-oss-20b), or stop at once with a clear message.
Unattended experiments do the opposite: they *wait* for the quota, so every answer in a run comes from one model.

## Spending fewer tokens in the first place
- **Focused reading**: a page is split into passages and only those matching the question are returned (≤ 3,000
  characters), instead of the whole article.
- **Short snippets**: search results carry 500-character snippets; the model reads a page only when it needs to verify.
- **Small reasoning effort**: gpt-oss with `reasoning_effort: low`.

Typical costs we measured: a research question takes 2-3 model calls and 4-7k tokens; a digest 7-11 calls and
21-40k tokens. On the free tier that's roughly 5 digests or 30 questions a day per model.

## Interview questions
- *Why does an agent get more expensive with every step?* Each call resends the whole growing conversation.
- *How do you keep an agent's context small?* Return focused tool outputs, trim or summarise old results, keep the
  newest results and the model's own reasoning, and budget tokens per task.
- *Rate limit vs quota?* A per-minute limit means wait and retry; a daily quota means switch model or stop.
