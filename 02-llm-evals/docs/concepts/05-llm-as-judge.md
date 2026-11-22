# 5. LLM-as-judge

## Intuition
Some questions can't be answered by counting words: *"Is every sentence of this answer backed by the
source?"*, *"Does this answer mean the same as the reference?"* A human can answer them, but slowly. An
**LLM judge** is a strong model given a written rubric, the material, and the output to grade, and asked
for a structured verdict.

It's fast and cheap, but it's also a model, so it can be wrong, lenient, strict, or biased. A judge is only a
metric once its **agreement with humans has been measured** (concept 6). Until then, it's an opinion.

## Anatomy of our judge ([`judge.py`](../../src/evalkit/judge.py))
1. **System role:** what kind of grader it is. For faithfulness: *"You never use outside knowledge: a statement
   that is true in the real world but is not in the SOURCE counts as unsupported."* That sentence matters:
   without it, models happily accept true-but-unsourced facts, while human annotators flag them.
2. **Rubric:** precise definitions of each verdict, plus what *not* to flag (faithful paraphrases,
   "the source doesn't say").
3. **Structured output:** JSON, requested through the API's JSON mode and parsed defensively.
4. **Evidence before verdict** (`faithfulness/claims`): the judge must first list each problem with an *exact
   quote* from the response, then give the verdict. Writing evidence first acts like a short chain of thought,
   and it makes every verdict auditable: you can read *why* it said "hallucinated".
5. **Temperature 0 and a disk cache:** the same input always gets the same verdict, and re-running an
   experiment costs no API calls.

| rubric | output | cost |
|---|---|---|
| `faithfulness/claims` | `{"problems": [{"quote", "type", "why"}], "verdict"}` | more output tokens; auditable |
| `faithfulness/direct` | `{"verdict"}` | cheapest; no evidence |
| `correctness` | `{"missing", "contradictions", "verdict": correct / partially_correct / incorrect}` | needs a reference answer |

## Practical problems we hit (and handled in code)
- **Rate limits.** Groq's free tier allows ~8,000 tokens/minute. The client waits as long as the provider asks
  and retries ([`llm.py`](../../src/evalkit/llm.py)).
- **Reasoning models think before answering.** `gpt-oss-120b` spent a 60-token budget on hidden reasoning
  and returned cut-off JSON, which the provider rejected. Fix: a larger budget, and on a JSON rejection
  one retry in plain mode with our own JSON parsing.
- **Unparseable output** is retried once without the cache; after that it's recorded as an error, never
  silently counted as "supported".

## Known judge biases (from the literature; we test some of them in RESULTS)
- **Leniency:** judges miss subtle unsupported details more often than blatant contradictions.
- **Verbosity bias:** longer answers get flagged more (more chances), or praised more (in quality rubrics).
- **Self-preference:** a judge may favour outputs from its own model family.
- **Position bias** in pairwise comparisons (A vs B): the first option wins more often; swap and average.

## Interview questions
- *Why not just ask "is this answer good? 1–10"?* Unanchored scales drift and are hard to validate. Binary or
  3-way decisions with written definitions agree with humans far better and can be measured with kappa.
- *How do you make a judge auditable?* Require quoted evidence before the verdict, and store the raw output.
- *Can the model you're evaluating also be the judge?* It can, but it risks self-preference; prefer a
  stronger or different model and measure agreement either way.
