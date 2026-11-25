# 6. Evaluating an agent

## Intuition
A chatbot gives one answer you can grade. An agent takes many steps, uses tools, and can fail in ways that never show
up in its final answer: it can be wrong, be hijacked by a web page, "fix" things that weren't broken, or write a
sentence its source doesn't support. One "task success" number hides most of that, so Scout is measured four ways,
each aimed at one way of failing.

| | failure it measures | how |
|---|---|---|
| **E1** | being wrong | SimpleQA questions, model alone vs. Scout with web tools, official grader prompt |
| **E2** | being hijacked | 10 prompt-injection attacks + 2 controls on a fake internet; *attempted* vs *executed* |
| **E3** | being busy | a small repo with 5 planted doc errors and 4 decoys; `checkup` must fix one kind, leave the other |
| **E4** | citing what the source doesn't say | every claim in a digest note judged against what the agent actually read |

## Design rules shared by all four
1. **Run the real code path.** E3 calls the same `checkup()` routine you use; E4 calls the same `digest()`. Only the
   approver changes (it declines), so nothing is written. An eval of a simplified copy measures the copy.
2. **Separate "tried" from "did".** In E2 the model sometimes *attempts* an outward action an attacker asked for; the
   approval gate stops it being *executed*. Reporting only the second would hide that the prompt alone isn't a defence;
   reporting only the first would hide that the gate works.
3. **Decoys and controls.** E3's decoys (a `#anchor` link, British spelling, a link inside a code block, a number that
   *matches* its source) catch an agent that changes everything; E2's controls (pages with no attack) check that
   defences don't break normal work.
4. **One model per run.** When the daily quota runs out, experiments wait instead of switching to the fallback model,
   so a result belongs to one model.
5. **Resumable.** Every item is saved as it finishes; a restart continues where it stopped.

## E4 in detail: claim support
Code already guarantees that a note cites only sources the agent **read**, and at least two of them. It can't check
that a sentence **says** what its source says. That needs a judge.

1. **Record the evidence.** During the run, `RecordingWeb` keeps every search snippet and every passage `web_read`
   returned, per source number. That is all the agent ever saw. (Re-fetching the page later would judge against a
   different text: pages change, and the agent only saw the passages most relevant to its question.)
2. **Split the note into claims.** Each bullet, and each sentence of a paragraph, with the section it's in and its
   `[n]` citations (`note_claims`). "Why it matters" and "Try it" are advice, so they count only where they cite.
3. **Judge each claim.**
   - cited claim → against the sources it cites → **citation support** (is the citation right?)
   - uncited fact in "What happened" / "Key details" → against everything read → **grounding** (is it from a source
     at all, or from the model's memory?)
4. **Use a validated judge.** The judge is project 2's: the configuration chosen on RAGTruth's dev set *before* seeing
   test results, with test kappa 0.625 against human labels, on qwen3.8-27b (a different model family from the writer,
   gpt-oss, so a model isn't grading its own style).
5. **Probe the judge on these notes.** For every claim the judge accepted, one number is changed (a year by −2, other
   numbers ×3), for example "next release on 30 March 2027" → "30 March 2025". The judge must now reject it. If it
   doesn't, its "supported" verdicts don't mean much here, whatever its kappa on RAGTruth.

```
digest() ──► note + evidence per source ──► note_claims() ──► judge(claim, its sources) ──► support rates
                    (RecordingWeb)                                     │
                                                                       └─► corrupt(claim) ──► judge must reject
```

## Limits worth saying out loud
- The judge is right about 4 times in 5 on RAGTruth, not always: per-claim verdicts are evidence, and the rates come with
  confidence intervals.
- Claim splitting is a heuristic (sentence ends, bullets). A long sentence with two facts is one claim.
- The probe only tests numbers. A judge that catches a wrong year might still miss a wrong name.
- "Supported by what it read" isn't "true": a source can be wrong. E1 measures truth; E4 measures faithfulness.

## Our code
`experiments/digest_eval.py` (`RecordingWeb`, `note_claims`, `corrupt`, `source_text`), `tests/test_digest_eval.py`,
`experiments/pipeline_digest.sh`. E1–E3: `experiments/simpleqa.py`, `injection.py`, `checkup_eval.py`.

## Results
See [RESULTS.md](../RESULTS.md).

## Interview questions
- *How would you evaluate an agent?* Separately for each way it can fail (wrong, hijacked, over-eager, unfaithful),
  on the real code path, with controls and decoys, one model per run, and intervals on every rate.
- *Why not just ask an LLM whether the note is good?* An unvalidated judge is a second unknown. We reuse a judge whose
  agreement with humans is measured, run it per claim against the exact evidence, and check it catches planted errors.
- *Faithfulness vs. correctness?* Faithful = says what its sources say; correct = true in the world. A note can be
  faithful to a wrong source. E4 measures the first, E1 the second.
