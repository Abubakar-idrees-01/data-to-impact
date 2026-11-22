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

## Status
Phases 1-3 done (33 tests). Groq key copied into 02-llm-evals/.env (EVALKIT_*; gitignored).
