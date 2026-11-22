#!/usr/bin/env bash
# The whole judge study, unattended. Free-tier friendly: each model has its own daily token quota, so the two
# models work in parallel, and every call waits out rate limits instead of failing. Re-running is cheap:
# finished calls come from .cache/llm.
#   bash experiments/pipeline.sh > logs/pipeline.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs
GPT="--model openai/gpt-oss-120b --effort low --max-tokens 2000"

# 1. three models, three separate daily quotas, in parallel:
#    gpt-oss-120b and qwen: remaining dev configurations | gpt-oss-20b: StudyRAG correctness grading
( $PY experiments/judge_validation.py --split dev --rubric faithfulness/severity $GPT
  $PY experiments/judge_validation.py --split dev --rubric faithfulness/direct $GPT ) > logs/dev_gptoss.log 2>&1 &
$PY experiments/judge_validation.py --split dev --rubric faithfulness/severity --model qwen/qwen3.8-27b \
  > logs/dev_qwen.log 2>&1 &
$PY experiments/studyrag_eval.py --correctness-only > logs/studyrag_correctness.log 2>&1 &
wait

# 2. choose the judge by the pre-registered rule, then one run on the held-out test split
$PY experiments/select_judge.py
$PY experiments/judge_validation.py --split test --selected results/judge-ragtruth-dev/selected.json
$PY experiments/judge_validation.py --split test --baselines

# 3. StudyRAG graded with the selected faithfulness judge (correctness comes from the cache)
$PY experiments/studyrag_eval.py
echo "PIPELINE DONE $(date)"
