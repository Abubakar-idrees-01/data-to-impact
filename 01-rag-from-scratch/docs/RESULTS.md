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

| threshold | public: answerable kept | public: unanswerable refused | private: answerable kept | private: unanswerable refused |
|---|---|---|---|---|
| 0.02 (initial guess) | 1.00 | 0.88 | 0.97 | 1.00 |
| 0.1 | 1.00 | 1.00 | 0.97 | 1.00 |
| **0.2** (chosen) | **1.00** | **1.00** | **0.97** | **1.00** |
| 0.5 | 1.00 | 1.00 | 0.97 | 1.00 |

Public answerable questions scored 0.80–1.00 and unanswerable ones at most 0.065. **Overall: 62 of 63 answerable
questions kept, 14 of 14 unanswerable refused.**

**The one miss** is a private question, *"How much is the final examination worth?"*, whose answer is a row in
a grading table. Retrieval ranked the right chunk #1, but the reranker scored it 0.005. Rewriting tables as
sentences ([DECISIONS D18](DECISIONS.md)) lifted it to 0.069, and rephrased as *"weightage of the final exam"*
it scores 0.80. It is still below 0.2. A threshold low enough to keep it would let a public unanswerable question
(0.065) through, so we keep 0.2. The refusal message shows the closest passage, which contains the answer.

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

## 5. End-to-end answers (Qwen3.5-2B, public set)
42 questions through the full pipeline: retrieve → refuse or generate → verify → auto-cite.

| metric | v1 | **v2** (current) | what it measures |
|---|---|---|---|
| answered (answerable) | 1.00 | **1.00** | not wrongly refused |
| refused (unanswerable) | 1.00 | **1.00** | said "not in your documents" |
| key-fact recall | 0.917 | **0.902** | share of the reference answer's key facts present |
| fully correct | 0.853 | **0.824** | every key fact present |
| mostly correct (≥ 50% facts) | 0.971 | **0.971** | |
| faithfulness | 0.927 | **0.951** | answer sentences supported by the retrieved passages |
| citation support | – | **0.957** | model-written citations whose passage supports the sentence |
| citation precision vs gold | 0.706 | **0.863** | cited passages that are the labelled gold section |
| answers with ≥ 1 citation | 1.00 | **1.00** | |
| median time | 37.3 s | **31.9 s** | including retrieval, verification |

**v1 → v2** added auto-citation of verified sentences and a prompt rule to cite every bullet (D15), and widened
two incomplete gold labels. Correctness moved within noise (one question ≈ 3 points); citations and
faithfulness improved. In 53% of answers at least one supported sentence arrived uncited and was cited automatically.

**Two citation metrics, on purpose.** *Citation precision* checks cited passages against our gold labels. It is a
lower bound, because real material often states a fact on several slides and the labels list only one.
*Citation support* needs no labels: it asks whether the passage a sentence cites actually backs that sentence.

## 6. Which local model? (same pipeline, three models)
| | Qwen3.5-0.8B | Qwen2.5-1.5B | **Qwen3.5-2B** (default) |
|---|---|---|---|
| file size (Q4_K_M) | 0.53 GB | 1.12 GB | 1.28 GB |
| prompt reading | 78.5 tok/s | 36.7 tok/s | 25.3 tok/s |
| generation | 18.9 tok/s | 14.2 tok/s | 5.7 tok/s |
| time to first token (491-token prompt) | 6.3 s | 13.4 s | 19.4 s |
| **public set** (34 answerable) | | | |
| key-fact recall | 0.900 | 0.922 | 0.902 |
| fully correct | 0.765 | 0.853 | 0.824 |
| faithfulness | 0.984 | 0.971 | 0.951 |
| citation support | 0.963 | 0.900 | 0.957 |
| median answer time | **14.6 s** | 22.6 s | 31.9 s |
| **private set** (29 answerable, OCR'd slides) | | | |
| key-fact recall | – | 0.905 | **0.941** |
| mostly correct | – | 0.893 | **0.964** |
| citation support | – | 0.800 | **0.978** |
| median answer time | – | 24.8 s | 34.9 s |
| **MCQs** (15 passages) | | | |
| yield / answer-key consistency | – | 0.73 / 0.91 | **1.00** / 0.87 |

**Decision: keep Qwen3.5-2B as the default.** On clean Wikipedia text the three models answer about equally well,
within the noise of 34 questions. The difference shows on the real, OCR'd lecture slides: the 2B model is
"mostly correct" more often (0.96 vs 0.89) and its own citations are far more trustworthy (0.98 vs 0.80; the
1.5B model needed automatic citations in 93% of answers). It also produced an MCQ from every passage. Trustworthy
citations are the point of this project, so we accept ~10 s more per answer.

**Surprise:** Qwen3.5-2B generates 2.5× slower than Qwen2.5-1.5B, far more than the parameter count explains.
Qwen3.5 uses a newer hybrid attention design whose CPU kernels in llama.cpp are less optimized than classic
attention. *Measure; don't assume model size predicts speed.*

**Recommendation by machine:** default 2B; `--llm qwen2.5-1.5b` for ~30% faster answers; `--llm qwen3.5-0.8b`
for weak machines (2× faster, a few more incomplete answers); or the API backend.

## 7. MCQ quality (Qwen3.5-2B, 15 passages from the public set)
| metric | v1 | **v2** |
|---|---|---|
| yield (valid MCQ per passage) | 1.00 | **1.00** |
| grounded (correct option supported by passage) | 1.00 | **0.93** |
| answer-key consistency (model re-answers its own question) | 0.87 | **0.87** |
| raw answer position A / B / C / D | 4 / 5 / 3 / 3 | 4 / 4 / 5 / 2 |
| time per MCQ | 69.6 s | 65.7 s |

**Human review of all 15 v2 questions:** 12 good. 3 flawed: two options paraphrasing each other (2 cases) and
a distractor that is also true. The review found two bugs that automatic metrics missed:
1. Explanations referred to options by letter ("Option C is correct"), but shuffling had moved them. In v2,
   all 15 explanations match the displayed letters ([D20](DECISIONS.md)).
2. Some questions repeated "A) … B) …" inside the stem; v2 had none.

The paraphrased-options flaw is now caught automatically (`_near_duplicate_options`). On the 30 reviewed MCQs it
flags exactly the 2 true duplicates and no valid question. The "grounded" drop from 1.00 to 0.93 is one false flag: the
correct option paraphrased the passage too loosely for the word-overlap check. That check is deliberately strict.

## 8. Private set: the owner's real course material (aggregates only)
3 lecture decks (150 image-only slides, OCR'd) + a course outline (DOCX with tables). 35 questions: 29 answerable,
6 unanswerable. These questions are written by the developer and pending the owner's review.

**Retrieval** (29 answerable):

| chunking | search | hit@1 | hit@3 | MRR@10 |
|---|---|---|---|---|
| fixed | hybrid + rerank | 0.97 | 1.00 | 0.983 |
| structure | hybrid | 0.97 | 1.00 | 0.983 |
| **structure** | **hybrid + rerank** (default) | 0.93 | 1.00 | 0.966 |
| semantic | hybrid + rerank | 0.93 | 1.00 | 0.966 |
| structure | bm25 only | 0.76 | 1.00 | 0.874 |
| structure | vector only | 0.83 | 0.97 | 0.905 |

Hit@3 is 1.00 in every hybrid configuration, so the right slide always reaches the LLM. Here the reranker
slightly *lowers* hit@1 (one question). We keep it because it drives the refusal decision, and the LLM sees the top 4 anyway.

**Answers** (Qwen3.5-2B): key-fact recall **0.94**, mostly correct **0.96**, fully correct 0.86, faithfulness
0.95, citation support **0.98**, refusals **6/6**, one false refusal (the table case above), median 35 s.
OCR'd text answers as well as clean text: the OCR pipeline works.

**Bugs this set found that the public set could not:** the OCR direction classifier (D16), tables read as noise by
the reranker (D18), and table rows losing their meaning ("5" instead of "Week 5"). That last one appeared as
a regression in the eval after the table fix and was caught before release.

