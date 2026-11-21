# 7. Verification: checking every claim against its source

## Why
A model can cite `[2]` and still say something passage [2] never says. Citations *look* trustworthy, so a
wrong citation is worse than none. So after generation we check each sentence.

## How ([`verify.py`](../../src/studyrag/verify.py))
For each factual sentence of the answer (ignoring short connectors like "Here's a summary:"):

```
support(sentence, passage) = 0.5 · lexical_recall + 0.5 · semantic_match
```

- **lexical_recall**: the share of the sentence's content words (stemmed, no stopwords) that appear in the
  passage. Invented names, numbers and terms drag this down.
- **semantic_match**: the highest cosine similarity between the sentence and any **single sentence** of the
  passage, rescaled from bge's typical range [0.55, 0.90] to [0, 1]. Comparing against single sentences rather
  than the whole passage avoids dilution. This catches faithful paraphrases that use different words.

We take the best score over the sources the sentence cites (or over all sources if it cites none). A sentence
is **supported** if that score ≥ `support_threshold`.

The share of supported sentences, combined with retrieval relevance, gives the **confidence badge**:

| Badge | Rule |
|---|---|
| High | best reranker relevance ≥ 0.5, ≥ 80% of sentences supported, and at least one citation |
| Medium | relevance ≥ 0.1 and ≥ 50% supported |
| Low | anything else |

## Honest limits
This is a fast heuristic, not logical proof. It can't detect a negation flip ("is" vs "is not") that keeps
all the same words, and it may under-score a correct sentence that combines two passages. A Natural Language
Inference (NLI) model would be stronger but adds another model download and is slower on CPU. Its precision
and recall on our data are **measured** in `eval/run_eval.py verifier` (→ [RESULTS.md](../RESULTS.md)), so we
know how much to trust it.

## Interview questions
- *How do you evaluate the verifier?* Build supported/unsupported claim pairs (a reference answer vs its own
  source = positive; vs a different passage from the same document = hard negative) and compute precision/recall per threshold.
- *Why not use the LLM to check itself?* Possible ("LLM-as-judge"), but on CPU it doubles latency, and a small
  model judging its own output shares its blind spots.
