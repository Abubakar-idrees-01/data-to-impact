# 4. Hybrid search with Reciprocal Rank Fusion

## Intuition
BM25 finds exact words, and vector search finds meaning. Each misses what the other catches, so we run
both and merge their results. The problem is that their scores aren't comparable: BM25 gives `12.7`
and cosine gives `0.83`. Adding them, or even normalising them, is fragile.

**Reciprocal Rank Fusion (RRF)** ignores scores and uses only **ranks**:

```
RRF(chunk) = Σ over retrievers   1 / (k + rank)          k = 60
```

## Worked example
| Chunk | BM25 rank | Vector rank | RRF score |
|---|---|---|---|
| A | 1 | 3 | 1/61 + 1/63 = 0.0323 |
| B | 2 | — | 1/62 = 0.0161 |
| C | — | 1 | 1/61 = 0.0164 |
| D | 4 | 2 | 1/64 + 1/62 = 0.0318 |

A and D, found by **both** retrievers, beat B and C, each found by only one. Being "pretty good in both"
beats being "the best in one". That is usually right: agreement between two independent signals is strong evidence.

## Why k = 60?
`k` flattens the curve. With k = 0, rank 1 (score 1.0) would crush rank 2 (0.5). With k = 60 the gap is
1/61 vs 1/62, so a chunk needs support from both lists to stand out. 60 comes from the original paper
(Cormack, Clarke & Büttcher, SIGIR 2009) and is robust in practice.

## Code
[`hybrid.py`](../../src/studyrag/hybrid.py) is 10 lines; [`retrieval.py`](../../src/studyrag/retrieval.py)
takes the top 20 from each retriever, fuses them, and keeps each chunk's individual ranks for the
"How I found this" panel.

## What we measured
The retrieval table in [RESULTS.md](../RESULTS.md) compares BM25 only, vector only, hybrid, and hybrid + reranker.

## Interview questions
- *Why not a weighted sum of normalised scores?* Score distributions differ per query and per retriever;
  min-max normalisation is distorted by outliers. RRF needs no tuning.
- *Downside of RRF?* It throws away score magnitude: a hugely confident #1 counts the same as a barely-#1.
  The reranker afterwards restores fine-grained scoring.
