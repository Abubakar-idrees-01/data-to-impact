# Results

All runs are saved in [`results/`](../results/) with per-item details, and can be re-checked with
`python -m evalkit summary <file>`. Judge calls are cached, so `experiments/pipeline.sh` reproduces every number
for free once the cache exists. Intervals are 95% (bootstrap, 1,000 resamples, seeded).

**Status:** dev-set (tuning) results are final. The held-out test run and the StudyRAG grading are running on the
free API tier (≈200k tokens per model per day); their sections are marked *pending*.

---

## 1. Judge vs human annotators: dev set (tuning)
60 RAGTruth answers (20 per task, half hallucinated per human annotators), from RAGTruth's *train* split.

| detector | kappa | 95% CI | recall | precision | F1 | verdicts returned |
|---|---|---|---|---|---|---|
| gpt-oss-120b · quote every problem (700-token budget) | (0.73) | (0.49 – 0.93) | 1.00 | 0.73 | 0.85 | **50%**: excluded |
| **qwen3.8-27b · verdict only** | **0.63** | 0.44 – 0.83 | 0.93 | 0.76 | 0.84 | 100% |
| qwen3.8-27b · quote every problem | 0.31 | 0.15 – 0.49 | 1.00 | 0.60 | 0.75 | 98% |
| word overlap (no LLM, threshold tuned on dev) | 0.30 | 0.10 – 0.51 | 0.90 | 0.60 | 0.72 | 100% |
| new numbers (no LLM) | 0.10 | −0.14 – 0.34 | 0.47 | 0.56 | 0.51 | 100% |
| always "hallucinated" | 0.00 | – | 1.00 | 0.50 | 0.67 | 100% |
| qwen3.8-27b · major/minor problems | *pending* | | | | | |
| gpt-oss-120b · major/minor problems (low effort, 2,000 tokens) | *pending* | | | | | |
| gpt-oss-120b · verdict only (low effort, 2,000 tokens) | *pending* | | | | | |

**Findings**
- **"Always hallucinated" scores F1 0.67 and kappa 0.** On a balanced set F1 rewards a detector that says yes to
  everything. Kappa doesn't. That's why kappa is the headline metric.
- **Asking for a list of problems made Qwen a nitpicker.** It caught every hallucination but flagged 15 of 22
  clean answers, for wording like "tasty Bloody Mary" (the review: "they make a Bloody Mary"). Its kappa, 0.31,
  was no better than counting shared words (0.30). The same model asked for a verdict only: **0.63**.
- **A judge that fails half the time can top the table.** gpt-oss-120b spent its 700-token budget on hidden
  reasoning and returned nothing for 30 of 60 cases; on the 30 it answered, kappa was 0.73. The pre-registered
  coverage rule (≥ 95% verdicts) excluded it; it is re-run with a larger budget and low reasoning effort.
- The intervals are wide (60 cases): the dev set ranks configurations, and the 240-case test set measures the chosen one.

## 2. Judge vs human annotators: held-out test set (*pending*)
240 answers from RAGTruth's *test* split (120 QA, 60 summaries, 60 data-to-text; half hallucinated), judged once
by the configuration chosen by `select_judge.py`. Will report: kappa with CI, recall, precision, confusion
matrix, agreement by task and by the LLM that wrote the answer, and recall on evident vs subtle hallucinations.

## 3. StudyRAG (project 1) v1 → v2: is the improvement real?
42 public questions (34 answerable, 8 unanswerable), the same for both versions; paired tests per item.
Already measured from project 1's own per-item results:

| metric | v1 | v2 | Δ | 95% CI of Δ | p | verdict |
|---|---|---|---|---|---|---|
| citation precision | 0.706 | 0.863 | +0.157 | +0.039 … +0.289 | 0.008 | real improvement |
| key-fact recall | 0.917 | 0.902 | −0.015 | −0.084 … +0.049 | 0.74 | noise (3 better, 4 worse) |
| verified-sentence ratio | 0.941 | 0.960 | +0.020 | −0.034 … +0.071 | 0.46 | noise |
| seconds per answer | 31.9 | 28.5 | −3.4 | −4.5 … −2.2 | < 0.001 | real speed-up |

*Pending:* the same comparison with the LLM judge's **correctness** (vs reference answers) and **faithfulness**
(vs the retrieved passages), and how often the judge agrees with project 1's word-matching key-fact metric and
its sentence verifier.

## 4. Cost
*Pending (reported from the client's accounting): calls, tokens, and time spent waiting on rate limits.*
