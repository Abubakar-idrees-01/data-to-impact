# 11. Evaluating a RAG system

"It works on the three questions I tried" is not evidence. Each component gets its own measurement, so when
something is wrong we know **which part** is wrong.

## The test sets
Each line of a JSONL dataset is a question with a **gold location** (document + slide/page/section numbers that
contain the answer), a reference answer, and key facts:
```json
{"id": "p07", "question": "What is a market space?", "answer": "A virtual market place ...",
 "key_facts": ["virtual market place", "physical boundaries do not apply"],
 "gold": [{"doc": "Digital Marketing Lec 1", "loc": [4]}], "answerable": true, "type": "definition"}
```
- **Public set**: questions over the Wikipedia sample corpus (committed).
- **Private set**: questions over real course slides (kept local, since the slides are copyrighted). Only aggregate numbers are published.
- Both include **unanswerable** questions (on-topic but not covered by the documents) to test refusal.

Labelling gold *locations* rather than chunk IDs makes the labels independent of chunking. A retrieved chunk
counts as relevant if its slide/page range overlaps a gold location, so we can fairly compare chunking strategies.

## Metrics
### Retrieval (no LLM needed, fast)
- **hit@k**: is a relevant chunk among the top k? (k = 1, 3, 5)
- **MRR@10**: mean of 1/rank of the first relevant chunk. It rewards ranking the right chunk *first*.

### Refusal calibration
For each candidate threshold: the share of answerable questions still answered vs the share of unanswerable
questions refused. We pick the best balance.

### Verifier
Precision/recall of the claim checker on supported vs unsupported claim pairs.

### End-to-end answers (needs the LLM)
- **Key-fact recall**: the share of the reference's key facts present in the answer (stemmed word overlap, or
  embedding similarity ≥ 0.85 for paraphrases).
- **Citation precision**: the share of cited sources that are actually gold-relevant.
- **Faithfulness**: the share of answer sentences the verifier marks as supported.
- **Answer rate** on answerable vs **refusal rate** on unanswerable questions.
- **Latency**: median seconds per answer on our CPU.

### MCQs
Yield, grounded rate, answer-key consistency, raw answer-position bias, and seconds per MCQ.

## Why not just use an LLM judge for everything?
On a CPU-only machine, a strong judge model isn't available offline, and a 2B model judging itself shares its
own blind spots. We use transparent automatic metrics and back them with a **human review** of samples.
Project 02 (`02-llm-evals`) will add a validated LLM-as-judge.

## Reproduce
```bash
python eval/run_eval.py retrieval --set sample
python eval/run_eval.py refusal   --set sample
python eval/run_eval.py verifier  --set sample
python eval/run_eval.py answers   --set sample
python eval/run_eval.py mcq       --set sample -n 15
python eval/run_eval.py speed
```
Results are written to `eval/results/` and summarised in [RESULTS.md](../RESULTS.md).
