# 6. Routing and cascades: the smallest model that can answer

## Intuition
"What is the capital of France?" doesn't need the biggest model on the laptop; a multi-step word problem might. On
this machine the 0.8B model answers at ~19 tokens/s and the 2B at ~6, so sending every request to the 2B wastes
two thirds of the user's waiting time on questions the 0.8B gets right anyway.

Two ways to choose, and they complement each other:

| | **router** (decide before answering) | **cascade** (decide after a cheap answer) |
|---|---|---|
| looks at | the prompt's text | the small model's own confidence in its answer |
| cost | ~0 | the small model's answer, plus the big one's when escalating |
| catches | requests that *look* hard (maths, reasoning, code, long) | answers that *came out* uncertain, whatever the prompt looked like |
| misses | hard questions that look easy ("capital of Myanmar?") | confident mistakes |

## The router ([`router.py`](../../src/adaptive/router.py))
Six features of the last user message: word count, how many numbers, an arithmetic expression, word-problem phrases
("how many", "each", "total", "percent"), reasoning words ("why", "explain", "prove", "step by step"), code. Rules:
code, reasoning, a long prompt or a word problem with ≥ 2 numbers → the top model; plain arithmetic → the middle
one; everything else → the smallest. The rules were written before any results, so the evaluation is not fitted to
its own test set.

## The cascade
Answer with the small model, measure its **confidence**, and escalate if it is under a threshold τ:

```
confidence = exp( mean log p(token) )  over the answer's tokens      (the geometric mean probability)
```
Token probabilities come from a llama.cpp logits hook ([`runtime.py`](../../src/adaptive/runtime.py),
`ProbRecorder`): at each step the hook sees the scores for the whole vocabulary and records the probability of the
token that was chosen. For word problems only the final "Answer: …" line counts: a fluent explanation is confident
about its words, not necessarily about its number.

A first look (0.8B): "Paris" 0.88 ✓, "37 × 43 = 1041" 0.39 ✗ (1,591), an invented Tour de France winner 0.42 ✗,
but also a wrong GSM8K answer at 0.99: once the reasoning has gone wrong, the final number follows from it with
confidence. Whether low confidence predicts mistakes often enough is the measurement.

## How it is evaluated ([`route_eval.py`](../../src/adaptive/route_eval.py), [`tasks.py`](../../src/adaptive/tasks.py))
128 questions with exact answers, so no LLM judge: 48 capitals (including commonly confused ones: Canberra, Ankara,
Naypyidaw, Dodoma, Yamoussoukro), 40 arithmetic problems from a fixed seed (half are 2-digit multiplications), and
the first 40 GSM8K test problems. Greedy decoding, so reruns give the same answers.

Every model answers every question **once**; the results (correct?, confidence, seconds) are stored. Every policy
is then simulated exactly from them: a cascade that escalates question *q* costs the small model's seconds on *q*
plus the big model's, and gets the big model's correctness. So thresholds can be swept for free.

Policies compared: each model alone; the rule router; cascades (0.8B → 2B, 0.8B → 1.5B, 1.5B → 2B, through all
three) at τ = 0.3 … 0.95; router + cascade; and the **oracle**, each question sent straight to the smallest model
that gets it right, an upper bound no real policy can reach.

## Results
*(added when `route_eval` has run; it needs an idle CPU, after the Part A training queue)*

## Interview questions
- *Why not always use the biggest model?* On a CPU, speed is inversely proportional to model size; most everyday
  requests don't need the extra quality, and the user waits for every token.
- *What is a model cascade?* Try a cheap model first; accept its answer if a confidence signal is high, otherwise
  pay for a stronger model. FrugalGPT (Chen et al., 2023) does this across API models.
- *Why geometric mean probability, not the mean?* It is exp(mean log-probability): one very unlikely token pulls it
  down sharply, which is what an uncertain answer looks like.
- *How do you evaluate many thresholds without re-running?* Store each model's answer, correctness, confidence and
  time per question once; every policy is a deterministic function of those.
