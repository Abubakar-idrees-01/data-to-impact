# 3. Confidence intervals and significance: is the difference real?

## Intuition
Evaluation sets for LLM apps are small: 40 questions, maybe 300. With 40 questions, **one question moves a rate
by 2.5 points**. So "v2 scores 0.85, v1 scored 0.82" may be noise. Two tools answer this:
- a **confidence interval** says how uncertain one number is;
- a **paired test** says whether version B is really better than A *on the same questions*.

## Confidence intervals
**Wilson interval** for a proportion (k successes out of n). The textbook p ± 1.96·√(p(1−p)/n) breaks near 0
and 1: for 10/10 it gives [1, 1], certainty from ten questions. Wilson gives **[0.72, 1.00]**, which is honest.

**Bootstrap** for anything else (F1, kappa, a median latency):
1. Resample the n items *with replacement* (some appear twice, some not at all).
2. Recompute the statistic.
3. Repeat 2,000 times; the 2.5th and 97.5th percentiles are the 95% interval.

It needs no formula, which is why we use it for kappa, where the formula is messy.

## Paired comparison
When two versions answer the **same** questions, compare them question by question. Hard questions are hard
for both, so pairing removes that noise.

**Paired bootstrap:** take per-question score differences (B − A), bootstrap their mean. If the 95% interval
excludes 0, the improvement is unlikely to be luck. p ≈ 2 × the share of resamples on the other side of 0.

**McNemar's test** for pass/fail scores. Only *discordant* questions matter:
- b = questions only A got right, c = questions only B got right.
- If the versions were equally good, each discordant question is a coin flip: b ~ Binomial(b + c, ½).
- Example (in our tests): b = 1, c = 9 → p = 2 · (C(10,0) + C(10,1)) / 2¹⁰ = 22/1024 ≈ **0.021**: a real
  improvement. b = 5, c = 5 → p = 1: no evidence either way, even if 50 other questions changed nothing.

## Code
[`metrics/stats.py`](../../src/evalkit/metrics/stats.py): `wilson_ci`, `bootstrap_ci`, `paired_bootstrap`,
`mcnemar_exact`, plus `pearson`/`spearman` correlations (for judge scores vs human scores).
Everything is seeded, so the same data always gives the same interval.

## Interview questions
- *Why paired tests instead of comparing two means with a t-test?* The two runs share questions; an unpaired
  test treats question difficulty as noise and needs far more data to see the same effect.
- *Your new prompt improves accuracy from 82% to 85% on 40 questions. Ship it?* Check the discordant
  questions: maybe 3 fixed and 2 broken. McNemar p ≈ 1, so there's no evidence; look at what broke.
- *What does a 95% CI mean?* If we repeated the whole experiment many times, about 95% of such intervals
  would contain the true value.
