# Design Decisions

Each entry records what we chose, what else we considered, and why. "Measured" means the choice came from our
own numbers in [RESULTS.md](RESULTS.md).

### D1. Python standard library only
**Considered:** pandas, numpy, scipy, matplotlib, openai, pydantic, ragas, deepeval.
**Why:** an evaluation library is installed into *other* projects' environments, where every dependency can
conflict. Here PyPI downloads ran at 30–80 KB/s, which turns a scipy install into an afternoon. Everything we
need (bootstrap, McNemar, kappa, HTTP, TOML, SVG) is a few dozen lines, and writing it is the point of a
"from scratch" portfolio. Installing evalkit adds zero packages.

### D2. Per-item results in every run, not just averages
**Why:** comparisons between versions need item-level pairing (paired tests are far more sensitive than
comparing two averages), and every regression should be traceable to the exact questions that broke.

### D3. Paired tests: McNemar for pass/fail, paired bootstrap otherwise
**Why:** two versions of a system answer the same questions. Pairing removes question difficulty from the
comparison. McNemar is exact for small counts. The bootstrap handles any metric without a formula.
*Measured on StudyRAG v1 → v2:* citation precision +0.16 (p = 0.008, real), key-fact recall −0.015 (p = 0.74,
noise). Project 1's claim that "correctness moved within noise" is now a tested statement.

### D4. Gate rules with an optional significance requirement
**Why:** on small sets, a strict "no drop allowed" gate fails on one flipped question; a loose gate misses real
regressions. `require_significance` fails a drop only when it is both larger than the tolerance and statistically
supported; absolute `min`/`max` limits still catch disasters immediately.

### D5. Judge over any OpenAI-compatible API, with a disk cache
**Considered:** a local judge (project 1's Qwen3.5-2B on CPU), provider SDKs.
**Why:** a 2B model on a 2015 CPU is slow (~30 s per item) and a much weaker judge. The OpenAI-compatible protocol
covers Groq, OpenAI, OpenRouter and local servers with one ~150-line client. The cache makes every experiment
re-runnable for free and bit-for-bit reproducible (temperature 0 + same request = same stored answer).

### D6. RAGTruth as the source of human labels
**Considered:** labelling StudyRAG answers ourselves; other public sets.
**Decision (with the owner):** public labels only. RAGTruth fits best: real RAG-style tasks, answers from six
different LLMs, span-level human annotations of both contradictions and unsupported additions, MIT licence.
Raw files are downloaded, not committed; the sampled ids are committed so the sample can be rebuilt exactly.

### D7. Balanced dev/test samples; dev from RAGTruth's train split
**Why:** a separate dev set lets us change prompts and fix bugs without contaminating the reported number.
Balancing (50% hallucinated) gives enough positives to measure recall with a usable interval. The cost is that
precision is optimistic compared with a natural hallucination rate; we report kappa and recall first, and say so.

### D8. A judge-selection rule fixed before the results
**Why:** choosing the configuration after looking at test results inflates the reported number. The rule
(highest dev kappa among configurations that answered ≥ 95% of cases) lives in `select_judge.py` and was
committed before the remaining dev runs finished.

### D9. Coverage requirement of 95%
**Measured:** gpt-oss-120b (700-token budget) returned verdicts for only 30 of 60 dev cases, because it spent its
budget reasoning, and scored the highest kappa (0.73) *on the half it answered*. Without a coverage rule, the
selection would have picked a judge that fails half the time.

### D10. Severity-rated problems instead of "list every problem" — measured
**Measured:** with the "quote every problem" rubric, Qwen3.8-27B flagged 20 of 29 clean dev answers (kappa 0.31,
no better than word overlap at 0.30), mostly for wording ("tasty" for a dish a reviewer liked). Verdict-only
reached 0.63. The severity rubric keeps the quoted evidence but only *major* problems count. **Outcome:** severity
helped (0.31 → 0.43 for Qwen) but verdict-only won for both models (Qwen 0.63, gpt-oss 0.57 vs 0.53), so D8's rule
chose Qwen verdict-only. On the held-out test it scored 0.625. The cost: its verdicts come without quoted evidence.

### D11. Reasoning models: larger budget, low effort, explicit empty-reply error
**Measured:** with a 700-token budget gpt-oss-120b returned empty content on 26 of 60 cases and truncated JSON on 4.
Fix: `reasoning_effort: low`, `max_tokens 2000`, and an explicit error when a reply is empty because the budget ran
out, so a failure can never be parsed as "no problems found".

### D12. Free tier, run unattended over two days
**Decision (with the owner):** stay on the free tier. Groq allows ~200k tokens per model per day, and quotas are
per model, so two models work in parallel. The client waits as long as the provider asks (up to 15 minutes per
wait) instead of failing, and `pipeline.sh` chains every step, so the study finishes by itself.

### D13. Static HTML report as the live demo
**Considered:** a Streamlit dashboard.
**Why:** an evaluation report is read, not interacted with. One self-contained HTML file (inline SVG, no
JavaScript) can be hosted free on GitHub Pages, never goes to sleep like a free app server, and can be attached
to an email. Every chart shows its values and confidence intervals as text too.

### D14. A separate model for StudyRAG correctness grading
**Why:** free-tier quotas are per model. Grading StudyRAG's answers for correctness on gpt-oss-20b, in parallel with the
judge study on the other two models, saved most of a day. Correctness grading isn't covered by the RAGTruth
validation either way, which RESULTS states.
