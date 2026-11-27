# 4. Runs, comparisons and regression gates

## Intuition
LLM apps change constantly: a new prompt, a new model, a new chunk size. Each change can fix five questions
and silently break three. Unit tests can't catch that, because the output is free text. What works:
1. **Save every evaluation run** with per-question results, not just the average.
2. **Compare runs question by question** and test whether the difference is real.
3. **Gate**: let CI fail when quality drops, the way it fails when a unit test breaks.

## A run ([`runs.py`](../../src/evalkit/runs.py))
One JSON file per run: `results/<suite>/<name>.json`, holding
- `items`: one record per test case (`id` plus any numeric results: `correct`, `fact_recall`, `seconds`…);
- `metrics`: the mean of each, with a 95% confidence interval (Wilson for pass/fail, bootstrap otherwise);
- `config` (model, prompt version…) and `env` (git commit, whether the code had uncommitted changes, Python,
  CPU count). With these, any number can be traced back to the exact code that produced it.

Any project can produce a run without importing evalkit: write one JSON line per item and run
`evalkit import details.jsonl --suite my-suite --name v2`.

## Comparing ([`compare.py`](../../src/evalkit/compare.py))
`evalkit compare v1.json v2.json` pairs items by `id` and reports for each metric:

| column | meaning |
|---|---|
| Δ and its 95% CI | paired bootstrap over items |
| p | McNemar for pass/fail metrics, paired bootstrap otherwise |
| better / worse items | the ids that changed, so you can read what broke |

Direction matters: for `seconds`, `tokens`, `cost` or `incorrect`, *lower* is better (marked ↓ in the table).
evalkit guesses this from the name (`tokens_per_second` stays higher-is-better) and can be told explicitly, which
ambiguous names need: `attempted` is good on a quiz and bad in a prompt-injection suite.

```bash
evalkit compare base.json new.json --lower-is-better attempted,executed
```

**Real example** (StudyRAG, project 1, v1 → v2 answers on 34 questions):

| metric | Δ | p | verdict |
|---|---|---|---|
| citation precision | +0.157 | 0.008 | real improvement (the auto-citation change) |
| key-fact recall | −0.015 | 0.74 | noise: 3 questions better, 4 worse |
| seconds per answer | −3.4 s | < 0.001 | real speed-up |

Project 1's write-up said "correctness moved within noise". This is the test that backs that sentence.

## The gate ([`gate.py`](../../src/evalkit/gate.py))
Rules in TOML:

```toml
[metrics.key_fact_recall]
min = 0.85                  # never below this
max_drop = 0.03             # never more than 3 points worse than the baseline run
require_significance = true # ...unless a paired test can't tell the drop from noise
```
`evalkit gate new.json --rules gate.toml --baseline old.json` prints PASS/FAIL per metric and exits with code 1
on any failure, which fails the CI job.

**Design tension:** a strict gate on a small set fails on noise (one question flips), and a loose one misses
real regressions. `require_significance` is the compromise: a drop must be both large *and* statistically
supported. Absolute `min` limits still catch disasters immediately.

## Interview questions
- *How do you regression-test an LLM app?* A fixed eval set, saved per-item results per version, a paired
  comparison, and a CI gate on the metrics that matter.
- *Why store per-item results?* Averages hide trade-offs: +5 fixed and −4 broken looks like +1. The item list
  shows what broke.
- *Why record the git commit and a dirty flag?* So a number in a README can be reproduced from the exact code.
