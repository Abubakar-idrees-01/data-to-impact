# 02 — LLM Evals

## Goal
A reusable evaluation harness that every other project in this repo plugs into, so each
claim in a README ("RAG is 85% faithful", "router saves 60% latency") is backed by numbers.

## Scope
- Dataset format for test cases (question, expected answer, reference docs, tags)
- Metrics: exact match / F1, retrieval recall@k & MRR, faithfulness / groundedness,
  LLM-as-judge with rubric, latency, tokens/sec, memory use
- **Judge validation**: measure agreement between LLM judge and human labels (rare in portfolios)
- Regression tracking: compare runs across versions, fail CI if quality drops
- HTML/Markdown report generation with charts

## Consumers
01-rag (retrieval + answers), 04-ai-agent (task success), 05-adaptive-llm (quality vs speed).

## Status
Not started.
