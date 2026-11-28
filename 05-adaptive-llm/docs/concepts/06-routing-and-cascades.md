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
Three local models (Q4_K_M, 2 threads, thinking off) × 128 questions = 384 answers, generated once in about 40 minutes;
every policy is then simulated from them ([`results/routing/`](../../results/routing/)).

### Each model alone
| model | accuracy | capitals | arithmetic | GSM8K | seconds / question |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.664 | 0.81 | 0.78 | 0.38 | **4.0** |
| Qwen2.5-1.5B | 0.820 | 0.94 | 0.85 | **0.65** | 5.2 |
| Qwen3.5-2B | **0.852** | **0.98** | **0.93** | 0.62 | 8.7 |
| *oracle: the smallest model that gets it right* | *0.922* | | | | *4.8* |

**Bigger is not better here.** The 2B has 6% more parameters than the 1.5B and is 1.6× slower, and it is *worse* on
the GSM8K word problems. Ranking models by size, as both the planner and the rule router do, gets the ladder wrong.

### Policies on all 128 questions
| policy | τ | accuracy | s / question |
|---|---|---|---|
| always 1.5B | | 0.820 | 5.20 |
| always 2B | | 0.852 | 8.66 |
| rule router | | 0.766 | 8.21 |
| cascade 0.8B → 2B | 0.9 | 0.781 | 6.19 |
| cascade 1.5B → 2B | 0.8 | 0.852 | 6.29 |
| cascade through all three | 0.9 | 0.781 | 6.00 |
| rule router + cascade | 0.9 | 0.852 | 8.76 |

These rows are *in-sample*: τ was picked by looking at the same questions. The 1.5B → 2B cascade looks best (the
2B's accuracy for 27% less time), but choosing it after seeing every result would be choosing on the test set, so
it is not in the headline below.

### The headline: τ chosen on the dev half, scored on the held-out half (64 questions)
| policy | τ (from dev) | accuracy [95% CI] | s / question |
|---|---|---|---|
| **always 1.5B** | | **0.812** [0.70, 0.89] | **5.13** |
| always 2B | | 0.797 [0.68, 0.88] | 8.33 |
| always 0.8B | | 0.719 [0.60, 0.81] | 4.24 |
| rule router | | 0.719 [0.60, 0.81] | 7.88 |
| cascade 0.8B → 2B | 0.95 | 0.797 [0.68, 0.88] | 7.09 |
| cascade through all | 0.95 | 0.828 [0.72, 0.90] | 7.95 |
| rule router + cascade | 0.8 | 0.812 [0.70, 0.89] | 8.21 |

Compared question by question with evalkit (project 2; McNemar on the questions that flipped):

| base → new | correct (better / worse) | p | seconds |
|---|---|---|---|
| always 1.5B → always 2B | 0.812 → 0.797 (5 / 6) | 1.00 | 5.1 → 8.3 (p < 0.001) |
| always 1.5B → cascade through all | 0.812 → 0.828 (5 / 4) | 1.00 | 5.1 → 8.0 (p = 0.03) |
| always 1.5B → rule router | 0.812 → 0.719 (3 / 9) | 0.15 | 5.1 → 7.9 (p < 0.001) |
| always 2B → cascade 0.8B → 2B | 0.797 → 0.797 (1 / 1) | 1.00 | 8.3 → 7.1 (p = 0.20) |

**No policy beats "always Qwen2.5-1.5B"**, and every one that comes close is 40–60% slower. With 64 questions,
none of the accuracy differences is significant (even 0.8B vs 1.5B: 12 / 6 flips, p = 0.24). The speed differences
are, so speed decides: on this laptop, run the 1.5B.

### Why the cascades fail: confidence is not correctness on reasoning
| | capitals | arithmetic | GSM8K (step-by-step) |
|---|---|---|---|
| AUROC of confidence for right vs wrong (0.5 = chance) | 0.92–1.00 | 0.85–0.97 | **0.51–0.66** |
| wrong answers with confidence ≥ 0.9 | 1 of 13 | 4 of 18 | **42 of 54** |

For a one-word answer the token probability is a good signal: when the 0.8B was wrong about a capital its confidence
averaged 0.61, against 0.84 when right. After a chain of reasoning it is not. The final number follows from the
model's own working with probability ~0.95, whether that working was right or wrong. **The probability measures
agreement with the text before it, not truth.** So the cascade escalates capitals and sums correctly and lets
confident wrong maths through (0.8B on GSM8K: 0.38 alone, 0.45 at best through a cascade).

The confidence signal does work where it applies: on arithmetic, the cascade from 0.8B to 2B reaches the 2B's
accuracy (0.93) at τ = 0.9. Better signals for reasoning would be self-consistency (sample several solutions and
check that they agree) or a verifier, each costing extra generations. On a CPU that cost is exactly what routing is
meant to save.

### What changes in the engine
- The ladder must be ranked by **measured** quality and speed, not by parameter count. Here the 1.5B is as accurate as the 2B within
  noise and 1.6× faster.
- Escalate on confidence only for short answers. For reasoning, start at the model that handles reasoning best.
- The oracle (0.922 at 4.8 s) shows that a better signal is worth a lot: 7 points above the best policy here, in
  less time than the 1.5B alone.

**For an interview:** "I generated every answer once from three local models, then simulated routing policies
offline, choosing thresholds on a dev half and reporting on the held-out half with McNemar tests. Nothing beat
always using the 1.5B model, because the 2B was slower and no more accurate. Cascades failed on maths: wrong
chain-of-thought answers had confidence above 0.9 in 42 of 54 cases, so token probability is a usable signal for
short answers (AUROC up to 1.0) but close to chance for reasoning (0.51–0.66)."

## Interview questions
- *Why not always use the biggest model?* On a CPU, speed is inversely proportional to model size; most everyday
  requests don't need the extra quality, and the user waits for every token.
- *What is a model cascade?* Try a cheap model first; accept its answer if a confidence signal is high, otherwise
  pay for a stronger model. FrugalGPT (Chen et al., 2023) does this across API models.
- *Why geometric mean probability, not the mean?* It is exp(mean log-probability): one very unlikely token pulls it
  down sharply, which is what an uncertain answer looks like.
- *How do you evaluate many thresholds without re-running?* Store each model's answer, correctness, confidence and
  time per question once; every policy is a deterministic function of those.
