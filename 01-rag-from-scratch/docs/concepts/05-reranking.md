# 5. Reranking with a cross-encoder

## Intuition
Retrieval is a two-stage funnel, like hiring:
1. **Screening** (fast, recall-oriented): BM25 + vectors pull ~20 plausible candidates from thousands.
2. **Interview** (slow, precision-oriented): a stronger model looks at each candidate carefully and re-orders them.

The interviewer is a **cross-encoder**: a transformer that reads `[question] + [chunk]` **together**, so
every word of the question can attend to every word of the chunk. It can tell that *"What is NOT a benefit of
email marketing?"* and a chunk listing email benefits are related but don't directly answer, which a
bi-encoder comparing two separate vectors cannot do.

| | Bi-encoder (embedder) | Cross-encoder (reranker) |
|---|---|---|
| Input | query and chunk separately | the pair together |
| Precompute chunks? | yes | no: must run per (query, chunk) |
| Speed for 10k chunks | ~1 ms (one matrix multiply) | far too slow |
| Accuracy | good | much better |

So: bi-encoder for the whole library, cross-encoder only for the top 20.

## Our model
`Xenova/ms-marco-MiniLM-L-6-v2` (22M parameters, ONNX), trained on Microsoft's MS MARCO search dataset.
It outputs a logit; we apply a sigmoid to get a **relevance probability** in [0, 1]
([`embeddings.py`](../../src/studyrag/embeddings.py)).

## A second job: deciding when to refuse
Because the reranker score is a calibrated-ish relevance probability, it tells us whether *anything* in the
library answers the question. If the best chunk scores below `refusal_threshold`, we refuse **before**
calling the LLM: no hallucination risk, and an instant response. The threshold is chosen from data, using
answerable and unanswerable questions, in `python eval/run_eval.py refusal` (→ [RESULTS.md](../RESULTS.md)).

## A weak spot we found: tables
MS MARCO is made of web passages written as prose. A table row like `Final examination | 40% | 1-5` doesn't
look like an answer to that model. On the owner's course outline, the question *"What is the weightage of the
final exam?"* scored **0.02**, below the refusal threshold, even though retrieval had ranked the right chunk #1.

**Fix: linearize tables at ingest** ([`ingest/cleaning.py`](../../src/studyrag/ingest/cleaning.py),
`linearize_table`). Every row repeats its column headers, turning it into a sentence:

```
Final examination | 40% | 1-5      →   Final examination — Weight: 40%; Primary CLO coverage: 1-5.
```

Same question, same model: **0.02 → 0.80**. We tested three formats and kept the best. Two-column tables, or
tables whose header repeats ("Item | Details | Item | Details"), are key/value lists and become `Key: value.`

The phrasing "How much is the final examination **worth**?" still scored low (0.07 after re-indexing; 0.005 before) even after the fix. A 22M
reranker has blind spots. That's why a refusal also shows the closest passages: if the system is wrong,
the student can still see the answer.

## Interview questions
- *Why not rerank everything?* Cost grows linearly with candidates, about 10–20 ms per pair on CPU.
- *What if the right chunk isn't in the top 20?* The reranker can't fix it, so first-stage recall (hit@20) bounds final quality.
- *Is the sigmoid output a true probability?* Not exactly (MS MARCO calibration ≠ your data), which is why
  we calibrate the threshold on our own questions.
- *What data does a reranker struggle with?* Anything unlike its training data: tables, code and very short
  key/value text. Rewrite it into sentences before indexing, and test with real documents.
