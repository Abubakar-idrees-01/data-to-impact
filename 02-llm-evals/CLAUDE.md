# 02 — LLM Evals ("evalkit")

## Goal
A reusable evaluation harness that every other project in this repo plugs into, so each
claim in a README ("RAG is 85% faithful", "router saves 60% latency") is backed by numbers.

## Decisions (2026-11-22, with owner)
- Package `evalkit` in `src/evalkit`, **Python standard library only** (PyPI is very slow here; also makes it
  trivially installable into other projects: `pip install -e ../02-llm-evals`). pytest is the only dev dep.
- **Judge model:** Groq via any OpenAI-compatible API (owner's key in `.env`, gitignored; same variable names
  as project 1: STUDYRAG_API_* are NOT reused, use EVALKIT_API_BASE_URL / EVALKIT_API_MODEL / EVALKIT_API_KEY).
  Local OpenAI-compatible server (llama.cpp / Ollama) is the offline fallback. Groq free tier: 8k tokens/min,
  ~1k output tokens/min on qwen3.8-27b, 1k requests/day per model -> judge calls are cached on disk + retried.
- **Human labels:** public dataset only: **RAGTruth** (MIT, human span-level hallucination labels on RAG
  responses; QA / Summary / Data2txt; 6 generator models). Raw files in `data/raw/ragtruth/` (gitignored);
  commit only sampled ids + the script that rebuilds the sample.
- Results go in `results/` (root .gitignore ignores `runs/`).
- Live demo = the static HTML report (GitHub Pages), no server needed.

## Plan (phases; docs written with each phase)
1. Dataset format + metrics (text, retrieval, agreement, stats with CIs) ✅
2. Runs (one JSON per run), compare (paired bootstrap, McNemar, flipped items), gate (TOML thresholds), CLI ✅
3. LLM client (urllib, retries, disk cache) + judge rubrics (faithfulness, correctness) + lexical baseline ✅
4. Judge validation on RAGTruth: dev set for prompt work, held-out test set for the final numbers;
   compare with non-LLM baselines; bias analysis by task / generator model / answer length
5. StudyRAG as first consumer: import project 1 runs, v1 vs v2 significance, judge vs key-fact heuristic
6. HTML report (inline SVG charts) + GitHub Pages + CI workflow (tests + gate)
7. README, SETUP, HOW_IT_WORKS, concepts, DECISIONS, RESULTS, TIMELINE, OVERVIEW, PDFs, screenshots

## Consumers
01-rag (retrieval + answers), 04-ai-agent (task success), 05-adaptive-llm (quality vs speed).

## Status (2026-11-25): complete
- Judge study finished: chosen judge (qwen3.8-27b, verdict-only) test kappa 0.625 [0.525, 0.714], recall 0.95,
  precision 0.75 (dev 0.633); word-overlap baseline 0.233. StudyRAG v1->v2: citation precision +0.157 (p=0.008) and
  -3.4 s real; judge correctness/faithfulness within noise. Cost 715 calls, ~807k tokens, $0.
- Docs final: README, SETUP, HOW_IT_WORKS, 6 concepts, DECISIONS D1-D14, RESULTS, OVERVIEW, TIMELINE, 7 PDFs,
  report site/index.html (+ docs/images/report.png). 36 tests. CI: .github/workflows/evalkit.yml (tests + gate).
- Long runs: ALWAYS launch detached (`setsid nohup bash experiments/pipeline.sh > logs/pipeline.log 2>&1 < /dev/null & disown`);
  Groq TPD is a rolling 24 h window, quotas are per model.
- Open (owner's call): publish site/ on GitHub Pages; strip the accidentally pushed RAGTruth files from git history.
