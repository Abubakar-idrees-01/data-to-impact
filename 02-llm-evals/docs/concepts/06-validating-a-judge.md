# 6. Validating an LLM judge (without fooling yourself)

## Intuition
"Our judge agrees with humans 85% of the time" is only meaningful if:
1. the humans' labels are real and independent of the judge;
2. the number wasn't tuned on the same data it's reported on;
3. it's compared with something simpler (is the LLM even needed?);
4. it comes with an error bar.

This note is the recipe we followed, and the mistakes it caught.

## 1. Human labels: RAGTruth
[RAGTruth](https://github.com/ParticleMedia/RAGTruth) (Niu et al., 2024, MIT licence) has ~18,000 answers that six
LLMs (GPT-4, GPT-3.5, Llama-2 7B/13B/70B, Mistral-7B) wrote for three RAG-style tasks: question answering over
web passages, news summarization, and writing a business overview from structured data. Trained annotators
marked every hallucinated span, as either a **conflict** with the source or **baseless** information not in it,
each *evident* or *subtle*. An answer counts as hallucinated if any span is marked.

Their definition matters: a statement that is true in the real world but absent from the source is still
"baseless". Our judge's instructions say exactly that.

## 2. Dev and test splits: never tune on the number you report
We sampled two disjoint sets, balanced 50/50 between hallucinated and clean and seeded so they can be rebuilt
exactly (`datasets/ragtruth_ids.json`):

| set | from RAGTruth's | size | used for |
|---|---|---|---|
| **dev** | train split | 60 (20 per task) | trying prompts, models, settings; fixing bugs |
| **test** | test split | 240 (120 QA, 60 summaries, 60 data-to-text) | one final run of the chosen judge |

Balancing makes the minority class big enough to measure recall and precision. The trade-off: precision on a
balanced set is higher than at a natural, lower hallucination rate. We report kappa and recall, which don't
depend on the balance as much, and say this explicitly.

## 3. A selection rule fixed in advance
After a few dev runs it's tempting to pick "whatever looks best on the test set". So the rule was written into
[`select_judge.py`](../../experiments/select_judge.py) **before** the remaining dev runs finished:
> highest kappa on dev, among configurations that returned a verdict for ≥ 95% of cases; ties within 0.02 go to
> the higher lower confidence bound, then the cheaper configuration.

The 95% coverage condition matters: a judge that fails on half the cases can look great on the half it answered.

## 4. Baselines
| detector | how it works | what it shows |
|---|---|---|
| always "hallucinated" | constant | F1 = 0.67 on a balanced set but kappa = 0: why F1 alone misleads |
| new numbers | flags numbers absent from the source | catches a narrow error type |
| word overlap | flags a sentence when < 55% of its content words appear in the source (threshold tuned on dev) | the bar an LLM must clear |

## 5. What the dev set taught us
- **Asking for evidence made one model a nitpicker.** Qwen3.8-27B with the "quote every problem" rubric caught
  every hallucination but also flagged 15 of 22 clean answers. It objected to things like "tasty Bloody Mary" for
  "they make a Bloody Mary". Kappa 0.31, no better than word overlap. The verdict-only rubric scored **0.63**.
  Lesson: a format that forces a list encourages the model to fill it.
- **Reasoning models can return nothing.** gpt-oss-120b spent its whole token budget thinking and returned empty
  replies for half the cases; its kappa on the other half looked best of all. The coverage rule excluded it.
  Fix: `reasoning_effort: low` and a larger budget, then re-run.
- **A severity field** (major vs minor problems, only major counts) was added to keep the evidence without the
  nitpicking; its dev results are in [RESULTS.md](../RESULTS.md).

## 6. Reading the final number
Report kappa **with its 95% interval**, recall (share of human-marked hallucinations caught), precision (share
of alarms that were real), and where it fails: by task, by which model wrote the answer, and evident vs subtle
hallucinations. A judge with kappa 0.6 is useful for comparing versions of a system; it isn't a replacement
for reading the outputs.

## Interview questions
- *How do you know your LLM judge works?* Agreement with independent human labels on held-out data, with a CI,
  against simple baselines, with the selection decided before looking at the test set.
- *Your judge has 95% recall. Good?* Only with precision: the "always hallucinated" detector has 100% recall.
- *Why might a judge agree less with humans on subtle hallucinations?* They often need world knowledge to see
  that a detail is *new*, and the boundary between "reasonable inference" and "baseless" is fuzzy even for
  annotators.
