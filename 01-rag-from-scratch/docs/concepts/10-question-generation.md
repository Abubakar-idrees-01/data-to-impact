# 10. Generating MCQs, short questions and flashcards

## Approach: one passage → one item
Instead of "write 10 MCQs about Lecture 2" (long prompt, long output, no traceable sources), we:
1. **Select passages** ([`select_passages`](../../src/studyrag/study/generate.py)): with a topic, the most relevant
   chunks via retrieval; without one, chunks spread evenly across the chosen lectures (so questions cover the whole
   lecture, not just the start). Thin chunks (< 25 words) are skipped.
2. **Generate one item per passage** as grammar-constrained JSON. Every MCQ automatically gets an exact source
   ("Lec 2 — Slide 14"), and each call is short enough for a CPU.
3. **Stream** items to the UI as they're ready.

## Fixing small-model habits
| Problem | Fix |
|---|---|
| Invalid / truncated JSON | grammar-constrained decoding (impossible by construction) |
| Answer position bias (small models often favour "A") | **shuffle options** after generation and remap the answer; the raw position is measured in the eval |
| Explanation says "Option C is correct", but shuffling moved that option to D | remap every letter reference ("Option C", "options A and B", "the answer is B") with the same permutation (`remap_letters`) |
| Options repeated inside the question ("…? A) x B) y") | cut from the stem; after shuffling they would contradict the real options |
| Duplicate questions | embedding similarity to previous questions > 0.9 → dropped; previous questions are also listed in the prompt as "do not repeat" |
| Two identical options | rejected |
| "All of the above" | flagged |
| Two options say the same thing | content-word overlap ≥ 85% between two options (unless one is negated) → rejected and regenerated |
| Two options are both defensible, or the key is wrong | **self-check**: the model answers its own shuffled question from the passage, like a student. If it picks another option, the question is regenerated once; if it still disagrees, it is kept with a ⚠️ warning |
| Correct option not actually in the passage | grounding check (word overlap of the answer + explanation with the passage) → flagged with ⚠️ |

## Quality measurement ([`eval/run_eval.py mcq`](../../eval/run_eval.py))
- **Yield**: valid MCQs per passage attempted.
- **Grounded rate**: the correct option is supported by its passage.
- **Answer-key consistency**: the model is shown the passage and its *own* question and must answer it. If it
  picks a different option than the key, the question is ambiguous or the key is wrong. This started as an
  eval metric; once the human review showed it was catching real flaws, it became part of generation.
- **Raw position distribution**: how often the model originally put the answer at A/B/C/D (shows why shuffling matters).
- A human review of a sample (see [RESULTS.md](../RESULTS.md)).

## Grading short answers
[`quiz.py`](../../src/studyrag/study/quiz.py) asks the LLM to grade the student's answer as correct, partially correct
or incorrect against the model answer **and** the source passage, with brief feedback. If the LLM call fails,
it falls back to keyword overlap with the model answer, so grading never breaks.

## Export
MCQs → Markdown (answers hidden in collapsible blocks) or JSON; flashcards → **Anki** CSV (`front;back`), so they
plug into a spaced-repetition app.
