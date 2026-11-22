# 1. Answer and retrieval metrics: scoring without a judge

## Intuition
Before asking an LLM to grade anything, use the cheap, exact metrics. They cost nothing, never change between
runs, and are easy to explain. Their weakness is that they compare **words, not meaning**.

## Answer metrics ([`metrics/text.py`](../../src/evalkit/metrics/text.py))
All of them first **normalize** both texts the SQuAD way: lowercase, drop punctuation and the articles
"a/an/the", collapse spaces. "The Eiffel-Tower!" becomes "eiffel tower".

| Metric | Question it answers | Example (answer vs reference "Paris") |
|---|---|---|
| **Exact match** | Is the normalized answer identical? | "Paris." → 1 · "In Paris" → 0 |
| **Token F1** | How many words overlap? Precision = shared/answer words, recall = shared/reference words | "The Eiffel tower in Paris" → P = 1/4, R = 1, **F1 = 0.4** |
| **ROUGE-L** | Like F1, but words must appear in the same order (longest common subsequence) | Used for summaries |
| **Key-fact recall** | What share of the reference's key facts appear? A fact counts if ≥ 60% of its content words are present | facts ["paid per sale", "cost per mille"] → 0.5 |

**The trap:** "Affiliates earn a commission on each purchase" and "affiliates are paid per sale" mean the same
thing, but share almost no words. Token metrics say 0. This is why an LLM judge exists, and why we *measure*
how often the cheap metric and the judge disagree.

## Retrieval metrics ([`metrics/retrieval.py`](../../src/evalkit/metrics/retrieval.py))
A retriever returns a **ranked list**. "Relevant" can be a set of ids or a rule ("overlaps the gold page").

| Metric | Formula | Meaning |
|---|---|---|
| **hit@k** | 1 if any relevant item is in the top k | "Did the right passage reach the LLM?" |
| **MRR** | mean of 1 / rank of the first relevant item | Rewards putting it *first*: rank 1 → 1, rank 2 → 0.5, rank 4 → 0.25 |
| **precision@k** | relevant in top k / k | How much of what we show is useful |
| **recall@k** | relevant found in top k / all relevant | How much of what exists we found |
| **nDCG@k** | DCG / ideal DCG, with DCG = Σ gain / log₂(rank + 1) | Graded relevance ("very relevant" vs "somewhat"), discounted by position |

Worked example, ranking `[d3, d1, d7, d2]`, relevant `{d1, d2}`: first hit at rank 2 → RR = 0.5; hit@1 = 0,
hit@2 = 1; recall@2 = 1/2; DCG@4 = 1/log₂3 + 1/log₂5, ideal = 1 + 1/log₂3, so nDCG@4 ≈ 0.65. These exact numbers
are checked in [`tests/test_metrics.py`](../../tests/test_metrics.py).

## Interview questions
- *Hit@k or recall@k for RAG?* Hit@k: the LLM needs one passage with the answer, not all of them.
- *Why is MRR more informative than hit@10?* Two retrievers can both have hit@10 = 1 while one always ranks the
  answer first and the other ninth; MRR separates them.
- *When is token F1 a bad metric?* For long, free-form answers: correct paraphrases score low and verbose answers
  are punished by precision.
