# 2. Agreement: is the judge as good as a human?

## Intuition
An LLM judge is itself a model, so it can be wrong. Before trusting "the judge says 90% of answers are
faithful", we compare the judge's labels with **human labels on the same items**. The question is: how often
do they agree, and is that more than luck?

## Why accuracy alone lies
Suppose 20% of answers contain a hallucination. A "judge" that always says *no hallucination* is **80%
accurate**, and completely useless. So we look at the problem as a confusion matrix, with
*positive* = "has a hallucination":

| | human: yes | human: no |
|---|---|---|
| **judge: yes** | TP (caught) | FP (false alarm) |
| **judge: no** | FN (missed) | TN |

- **Precision** = TP / (TP + FP): when the judge raises an alarm, how often is it right?
- **Recall** = TP / (TP + FN): of the real hallucinations, how many did it catch?
- **F1**: harmonic mean of the two; low if either is low.
- **Balanced accuracy** = (recall + specificity) / 2: accuracy that weights both classes equally. The
  always-no judge scores 0.5, i.e. chance.

## Cohen's kappa: agreement beyond chance
Two raters who both say "no" 80% of the time will agree often *by chance alone*. Kappa removes that:

  **κ = (p_o − p_e) / (1 − p_e)**

- p_o = observed agreement (share of items where they agree)
- p_e = agreement expected by chance = P(both yes) + P(both no) if they labelled independently

Worked example (in our tests): 50 items; both yes 20, both no 15, disagreements 5 + 10.
p_o = 35/50 = 0.70. Rater A says yes 25/50, B says yes 30/50, so p_e = 0.5·0.6 + 0.5·0.4 = 0.50.
κ = (0.70 − 0.50) / 0.50 = **0.40**.

The always-no judge gets **κ = 0**, which exposes it immediately.

Common reading (Landis & Koch, 1977; a convention, not a law): < 0.2 slight, 0.2–0.4 fair, 0.4–0.6
moderate, 0.6–0.8 substantial, > 0.8 almost perfect. Even two humans labelling hallucinations rarely
exceed ~0.8, so that is the realistic ceiling for a judge.

**MCC** (Matthews correlation) is the correlation between the two yes/no label lists; like kappa, 0 = chance.

## Code
[`metrics/agreement.py`](../../src/evalkit/metrics/agreement.py): `Confusion.from_labels(judge, human)`
gives all of the above; `cohen_kappa(a, b)` handles any number of categories (e.g. scores 1–5).

## Interview questions
- *Your judge is 85% accurate. Is it good?* Depends on the base rate. Report kappa and recall on the
  minority class.
- *Precision or recall for a hallucination detector?* For a safety filter, recall (don't miss
  hallucinations); for an automatic metric on a dashboard, both, so F1 and kappa.
- *How would you know your human labels are reliable?* Have two humans label a subset and compute
  their kappa; that is the ceiling for any judge.
