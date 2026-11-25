# evalkit: Project Timeline

How evalkit went from an empty folder to a validated evaluation toolkit: what was built, in what order, what went
wrong, and how it was solved. Dates and times come from the commit history (22 November 2026 onwards).

## Where we started
- An empty `02-llm-evals` folder with a goal: a shared evaluation harness for every project in the portfolio, whose
  standout feature would be **checking the LLM judge against human labels**.
- Two decisions taken with the owner: the judge runs on **Groq's free tier** (any OpenAI-compatible API), and the
  human labels come from a **public dataset** rather than hand-labelling.
- The same laptop as project 1: 2015 CPU, no GPU, slow PyPI downloads.

## Phase by phase

### Phase 1: Foundations (until 14:14)
**What we did:** chose RAGTruth (MIT, human span-level hallucination labels, 36 MB). Wrote the test-case format
and every metric: text, retrieval, agreement (kappa, MCC) and statistics (Wilson and bootstrap intervals,
paired bootstrap, McNemar), each tested against hand-computed values.

| Difficulty | How we solved it |
|---|---|
| PyPI is very slow on this connection. | No third-party dependencies at all: statistics, HTTP, TOML and charts with the standard library. |
| The root `.gitignore` rule `data/raw/` only matched the repository root, so the 36 MB dataset was pushed. | Rule changed to `**/data/raw/` for every project; files untracked. (They remain in one old commit's history.) |

### Phase 2: Runs, comparisons, gates (14:14–14:17)
**What we did:** runs as JSON files with per-item results; `evalkit compare` with paired tests; `evalkit gate`
for CI; a command-line tool. First real use: StudyRAG v1 → v2.

| Difficulty | How we solved it |
|---|---|
| The comparison called slower answers "better" (it assumed higher is always better). | Direction-aware comparison: metrics like `seconds` or `cost` are recognised by name, or declared. |

**First finding:** StudyRAG's citation improvement is statistically real (p = 0.008); its key-fact change is noise
(p = 0.74). Project 1 had said "within noise"; now it's tested.

### Phase 3: The LLM judge (14:17–14:19)
**What we did:** an OpenAI-compatible client in ~150 lines of standard library, with a disk cache and rate-limit
handling; faithfulness and correctness rubrics; non-LLM baselines.

| Difficulty | How we solved it |
|---|---|
| gpt-oss-120b returned cut-off JSON that Groq rejected. | It's a reasoning model and was thinking past its budget: bigger budget, and on a JSON rejection, a retry in plain mode. |

### Phase 4: Validating the judge (14:24–16:27, then running unattended)
**What we did:** a balanced, seeded RAGTruth sample, with 60 dev cases for tuning and 240 test cases kept untouched,
plus three baselines. Then four judge configurations on the dev set.

| Problem found by checking the dev results | Fix |
|---|---|
| **The evidence rubric made Qwen a nitpicker:** 20 of 29 clean answers flagged, for wording like "tasty" or "highlight". Kappa 0.31, no better than counting words. | A verdict-only rubric (0.63), and a severity rubric where only *major* problems count. |
| **gpt-oss returned nothing for half the cases:** its hidden reasoning used the whole 700-token budget. Its kappa on the other half looked best (0.73). | A coverage rule (≥ 95% verdicts) in the pre-registered selection; re-run with low reasoning effort and a 2,000-token budget; empty replies now raise a clear error. |
| `pkill -f <pattern>` killed its own shell, because the pattern appeared in its own command line. | Stop processes by PID only. |
| **Groq's free daily quota** (200k tokens per model per day) ran out with ~750k tokens of work left. | With the owner: stay free. Each model has its own quota, so two work in parallel; the client waits as long as Groq asks; `pipeline.sh` runs every step unattended. Cached calls make restarts free. |
| The DNS dropped once mid-run. | Network errors retry with backoff; a case that still fails is counted as an error, never as a verdict. |

### Phase 5: Docs and report (from 16:35, while the study runs)
README, setup guide, architecture, six concept notes, decisions D1–D13, results (dev final), the HTML report and
this timeline.

### Still running
The selected judge on the 240 held-out cases, and the grading of StudyRAG's answers. *(This section is updated
when they finish.)*

## Lessons so far
1. **Validate the evaluator.** Two judge setups that looked reasonable were no better than word counting, or failed
   half the time; only comparison with human labels showed it.
2. **Coverage is a metric.** A judge's score on the cases it answered says nothing about the ones it didn't.
3. **Decide the selection rule before you see the results.**
4. **Free tiers are fine for a study if the code is patient:** cache everything, wait as asked, make restarts free.
