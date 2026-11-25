# Results

All runs are saved in [`results/`](../results/) with per-item details; check any of them with
`python -m evalkit summary <file>`. Judge calls are cached, so `experiments/pipeline.sh` reproduces every number
for free once the cache exists. Intervals are 95% (bootstrap, 1,000 resamples, seeded).

**Headline:** on 240 held-out answers, the chosen LLM judge (Qwen3.8-27B, verdict-only rubric) agrees with human
hallucination labels at **kappa 0.625 (95% CI 0.525–0.714)**: it catches **95%** of human-marked hallucinations, and
**75%** of its alarms are real. The best non-LLM baseline reaches kappa 0.233.

---

## 1. Tuning: dev set
60 RAGTruth answers (20 per task, half hallucinated according to human annotators), from RAGTruth's *train* split.
The configuration for the test set was chosen by the rule fixed in advance (highest kappa among configurations
that returned a verdict for ≥ 95% of cases).

| detector | kappa | 95% CI | recall | precision | F1 | verdicts |
|---|---|---|---|---|---|---|
| gpt-oss-120b · quote every problem (700-token budget) | (0.73) | (0.49 – 0.93) | 1.00 | 0.73 | 0.85 | **50%**: excluded |
| **qwen3.8-27b · verdict only** ✓ | **0.63** | 0.44 – 0.83 | 0.93 | 0.76 | 0.84 | 100% |
| gpt-oss-120b · verdict only (low effort, 2,000 tokens) | 0.57 | 0.36 – 0.77 | 0.87 | 0.74 | 0.80 | 100% |
| gpt-oss-120b · major/minor problems (low effort) | 0.53 | 0.31 – 0.73 | 0.80 | 0.75 | 0.77 | 100% |
| qwen3.8-27b · major/minor problems | 0.43 | 0.25 – 0.63 | 1.00 | 0.64 | 0.78 | 100% |
| qwen3.8-27b · quote every problem | 0.31 | 0.14 – 0.49 | 1.00 | 0.60 | 0.75 | 98% |
| word overlap (no LLM, threshold tuned on dev) | 0.30 | 0.10 – 0.51 | 0.90 | 0.60 | 0.72 | 100% |
| new numbers (no LLM) | 0.10 | −0.14 – 0.34 | 0.47 | 0.56 | 0.51 | 100% |
| always "hallucinated" | 0.00 | – | 1.00 | 0.50 | 0.67 | 100% |

**What tuning showed**
- **Asking for evidence made the judges worse.** For both models, the verdict-only rubric beat every rubric that
  asked the judge to list problems. With "quote every problem", Qwen flagged 20 of 29 clean answers, for wording
  like "tasty Bloody Mary" (the review: "they make a Bloody Mary"); kappa 0.31, no better than counting shared
  words. Adding a major/minor severity field helped (0.43) but didn't close the gap. A list format invites the
  model to fill it.
- **A judge that fails half the time can top the table.** gpt-oss-120b spent its 700-token budget on hidden
  reasoning and returned nothing for 30 of 60 cases; on the half it answered, kappa was 0.73. The coverage rule
  excluded it. Re-run with low reasoning effort and a 2,000-token budget, it answered every case and scored 0.57.
- **F1 rewards saying "hallucinated" to everything** (0.67 on a balanced set); kappa doesn't (0). Kappa is the
  headline metric for that reason.

## 2. Held-out test set
240 answers from RAGTruth's *test* split (120 QA, 60 summaries, 60 data-to-text; half hallucinated), judged once
by the chosen configuration.

| detector | kappa | 95% CI | recall | precision | F1 | accuracy |
|---|---|---|---|---|---|---|
| **qwen3.8-27b · verdict only** | **0.625** | **0.525 – 0.714** | **0.95** | **0.75** | **0.84** | **0.81** |
| word overlap (no LLM) | 0.233 | 0.127 – 0.339 | 0.87 | 0.58 | 0.69 | 0.62 |
| new numbers (no LLM) | 0.125 | 0.014 – 0.239 | 0.36 | 0.61 | 0.45 | 0.56 |
| always "hallucinated" | 0.000 | – | 1.00 | 0.50 | 0.67 | 0.50 |

Confusion matrix of the judge: 114 hallucinations caught, **6 missed**, **39 false alarms**, 81 clean answers passed.
The test kappa (0.625) is almost identical to the dev kappa (0.633), so the selection didn't overfit the dev set.

**Where it agrees less**

| slice | n | kappa | recall | precision |
|---|---|---|---|---|
| task: summaries | 60 | 0.80 | 0.90 | 0.90 |
| task: data-to-text (business descriptions) | 60 | 0.60 | 0.93 | 0.74 |
| task: question answering | 120 | 0.55 | 0.98 | 0.69 |
| answers written by Mistral-7B | 43 | 0.76 | 1.00 | 0.92 |
| answers written by GPT-4 | 39 | 0.49 | 0.80 | **0.44** |
| answers written by GPT-3.5 | 27 | 0.43 | 0.60 | 0.50 |

- **The judge's errors are almost all false alarms** (39 vs 6 misses): it is stricter than the annotators.
- **It over-flags fluent answers.** On GPT-4's answers, fewer than half of its alarms were real: strong models
  paraphrase and infer more, and the judge reads that as unsupported. The QA task (short web passages, where
  answers often generalize) shows the same pattern.
- **Mild length effect:** clean answers it wrongly flagged were longer (median 130 words) than those it passed (113).
- Recall on *evident* hallucinations 0.94 (n = 107); on *subtle-only* ones 1.00, but n = 13 is too small to read much into.

## 3. StudyRAG (project 1) graded by the validated judge
42 public questions (34 answerable, 8 unanswerable), the same for both versions, paired tests per item.
Correctness judge: gpt-oss-20b vs the reference answers (not validated against humans; RAGTruth validates the
faithfulness judge). Faithfulness judge: the chosen configuration above, against the passages each answer was
generated from.

| metric | v1 | v2 | Δ | 95% CI of Δ | p | verdict |
|---|---|---|---|---|---|---|
| citation precision | 0.706 | 0.863 | +0.157 | +0.039 … +0.289 | 0.008 | **real improvement** |
| seconds per answer | 31.9 | 28.5 | −3.4 | −4.5 … −2.2 | < 0.001 | **real speed-up** |
| judge: correctness (1 / 0.5 / 0) | 0.926 | 0.912 | −0.015 | −0.088 … +0.044 | 0.83 | noise |
| key-fact recall (word matching) | 0.917 | 0.902 | −0.015 | −0.084 … +0.049 | 0.74 | noise |
| judge: faithful (no unsupported claim) | 0.882 | 0.824 | −0.059 | −0.235 … +0.118 | 0.75 | noise (4 better, 6 worse) |
| verifier: supported-sentence ratio | 0.927 | 0.951 | +0.024 | −0.043 … +0.087 | 0.46 | noise |
| answered when answerable / refused when not | 1.00 / 1.00 | 1.00 / 1.00 | 0 | | | |

**Findings**
- **v2's only real gains are citations and speed.** Both correctness measures agree that answer quality didn't move,
  which confirms project 1's statement "correctness moved within noise" with a second, independent method.
- **The word-matching key-fact metric holds up.** It agrees with the LLM correctness judge on "mostly correct" for
  33 of 34 answers. The exception (s01) scored 0 on key facts while the judge called it partially correct, a
  paraphrase the word matcher missed.
- **The LLM judge is stricter about faithfulness than project 1's verifier** (82% vs 95% for v2). It flags answers
  whose every sentence overlaps a source, which fits its test-set behaviour of over-flagging fluent paraphrase.
  Read StudyRAG's faithfulness as "between 82% and 95%".
- **The CI gate passes** (`gates/studyrag.toml`): all floors cleared, and the faithfulness drop (−0.059, above the
  0.05 tolerance) is not significant (p = 0.75), so `require_significance` lets it through.

## 4. Cost
| model | calls | prompt tokens | completion tokens |
|---|---|---|---|
| qwen/qwen3.8-27b | 452 | 449,873 | 33,275 |
| openai/gpt-oss-120b | 196 | 215,205 | 76,170 |
| openai/gpt-oss-20b | 67 | 26,766 | 6,167 |
| **total** | **715** | **691,844** | **115,612** |

All on Groq's free tier: **$0**, at the price of time. The daily cap (200k tokens per model) and two interruptions
(a session ending, the laptop shutting down) stretched the study over three days; the disk cache meant no call was
ever paid for twice. Note how gpt-oss spends more tokens *thinking* (76k completion tokens for 196 calls) than Qwen
does answering (33k for 452).
