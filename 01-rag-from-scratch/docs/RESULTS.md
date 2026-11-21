# Results

All numbers were measured on the development laptop: **Intel i5-5200U (2015, 2 cores / 4 threads), no GPU,
15 GB RAM**. Reproduce any table with `python eval/run_eval.py <task> --set sample`. Raw outputs, including
per-question details, are in [`eval/results/`](../eval/results/).

**Datasets**
- **Public** ([`eval/datasets/sample_qa.jsonl`](../eval/datasets/sample_qa.jsonl)): 42 questions over 10 Wikipedia
  articles on digital marketing (~29k words, 250 chunks). 34 are answerable, with gold section labels; 8 are on-topic but **unanswerable**.
- **Private**: questions over the owner's real course material (3 lecture decks with 150 image-only slides,
  plus a course outline). The slides are copyrighted, so only aggregate numbers are published.

**Honest caveats:** the public questions were written by the developer while reading the source sections, so
their wording may be closer to the source text than real student questions. That favours keyword search
somewhat. Small sets (34/8 questions) mean ±1 question moves a rate by ~3 points.

---

## 1. Retrieval: chunking × search method
34 answerable questions; a hit means a chunk overlapping the gold section is in the top k.

| chunking | search | chunks | avg words | hit@1 | hit@3 | hit@5 | MRR@10 |
|---|---|---|---|---|---|---|---|
| fixed | bm25 | 197 | 177 | 0.76 | 0.91 | 0.97 | 0.842 |
| fixed | vector | 197 | 177 | 0.76 | 0.91 | 0.97 | 0.843 |
| fixed | hybrid | 197 | 177 | 0.85 | 0.97 | 0.97 | 0.912 |
| fixed | hybrid + rerank | 197 | 177 | 0.88 | 1.00 | 1.00 | 0.936 |
| structure | bm25 | 250 | 118 | 0.85 | 0.94 | 0.94 | 0.892 |
| structure | vector | 250 | 118 | 0.88 | 0.94 | 0.94 | 0.906 |
| structure | hybrid | 250 | 118 | 0.91 | 0.94 | 0.97 | 0.930 |
| **structure** | **hybrid + rerank** | 250 | 118 | **0.91** | **1.00** | **1.00** | **0.956** |
| semantic | bm25 | 281 | 104 | 0.85 | 0.91 | 0.94 | 0.878 |
| semantic | vector | 281 | 104 | 0.85 | 0.91 | 0.91 | 0.882 |
| semantic | hybrid | 281 | 104 | 0.82 | 0.91 | 0.97 | 0.884 |
| semantic | hybrid + rerank | 281 | 104 | 0.91 | 1.00 | 1.00 | 0.956 |

**Takeaways**
- **Hybrid > either retriever alone** for fixed and structure chunking (MRR +0.07 and +0.03).
- **Reranking is the biggest single win:** hit@3 reaches 1.00 for every chunking strategy.
- **Structure-aware chunking is best without reranking** and ties semantic with it, while being cheaper to build
  (no per-sentence embeddings) and giving cleaner citations, so it is the default.

### Rerank depth (structure + hybrid + rerank)
| candidates reranked | hit@1 | hit@3 | MRR@10 | median latency |
|---|---|---|---|---|
| 20 | 0.91 | 1.00 | 0.956 | 2068 ms |
| **10** (default) | **0.94** | **1.00** | **0.971** | **985 ms** |
| 6 | 0.94 | 1.00 | 0.971 | 603 ms |

The cross-encoder costs ~100 ms per pair on this CPU, so reranking 10 instead of 20 halves latency with no
accuracy loss. 10 keeps more headroom than 6 for harder questions.

## 2. Refusal calibration
Best reranker relevance for each question, then the share kept/refused at each threshold:

| threshold | answerable kept | unanswerable refused |
|---|---|---|
| 0.02 (initial guess) | 1.00 | 0.88 |
| 0.1 | 1.00 | 1.00 |
| **0.2** (chosen) | **1.00** | **1.00** |
| 0.5 | 1.00 | 1.00 |

Answerable questions scored 0.80–1.00; unanswerable ones at most 0.065. The threshold of 0.2 sits in the middle of
a wide gap (on a log scale).

## 3. Claim verifier
68 claim/passage pairs: each reference answer vs its own source (should pass) and vs a different section of the
*same* article (a topically close hard negative, should fail).

| threshold | precision | recall | wrong-source claims flagged |
|---|---|---|---|
| 0.45 | 0.92 | 0.97 | 0.91 |
| **0.55** (default) | **0.97** | **0.97** | **0.97** |
| 0.65 | 0.97 | 0.97 | 0.97 |

## 4. LLM speed (Qwen3.5-2B, Q4_K_M, llama.cpp)
| threads | prompt reading | generation | time to first token (491-token prompt) |
|---|---|---|---|
| **2** (= physical cores, default) | 25.3 tok/s | **5.7 tok/s** | 19.4 s |
| 4 (with hyper-threads) | 24.9 tok/s | 4.7 tok/s | 19.7 s |

Hyper-threads don't help: CPU decoding is memory-bandwidth-bound.

<!-- ANSWERS -->
<!-- MODELS -->
<!-- MCQ -->
<!-- PRIVATE -->
