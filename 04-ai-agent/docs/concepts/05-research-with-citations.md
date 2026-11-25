# 5. Web research with citations you can trust

## Intuition
"Search the web and summarise" is easy. Getting a summary where every claim can be checked is the hard part:
models cite the wrong source, cite sources they never read, and fill gaps from memory.

## Search → read → cite
1. **Search** (Tavily) returns numbered sources with short snippets.
2. **Read** (`web_read`) fetches a page and returns the passages most relevant to the question.
3. **Cite**: every claim gets `[n]`, and the answer ends with the numbered source list.

A `SourceBook` gives each URL **one number per task**: a page found by search and later read keeps the same [n],
so a citation always maps to exactly one URL. URLs are cleaned first (no `#fragment`, no `?ref=` or `utm_` tracking
parameters), so the same page isn't counted twice.

## Rules enforced in code, because prompts weren't enough
Each rule came from a live run that went wrong:
| what the model did | rule now |
|---|---|
| wrote citations as `【1†L13-L22】` (gpt-oss's training format), so our `[n]` parser found none | normalise to `[n]` in answers *and* tool arguments |
| cited sources it had only seen as a snippet | a note may cite only sources it **read** (`write_note` refuses otherwise) |
| based a whole note on one SEO blog | at least **two different** read sources |
| added "2024" to every search, anchored on its training year | the task states today's date and what "recent" means; searches use `time_range=month` |
| relied on secondary write-ups | digests search **primary sources first** (arXiv, GitHub, official docs and blogs), widening to the whole web only if nothing is found, and saying so |
| invented a "Sources" section | the source list is generated from the SourceBook, never from the model's text |

## What still needs a human
Even with these rules, one of our drafts attributed a claim to the wrong source (a release-notes detail cited to a
Go SDK page). Rules can check *that* a source was read, not that it *says* what the sentence claims. That's why
every note is reviewed before it's committed, and why the evaluation measures claim support.

## Interview questions
- *How do you make an LLM's citations trustworthy?* Number sources once per task, only allow citing what was read,
  generate the source list from tool data, prefer primary sources, and verify claim support (human or judge).
- *Why do agents still hallucinate with search?* Search gives them sources, not a guarantee they use them: they can
  misattribute, over-generalise from a snippet, or mix in memory.
