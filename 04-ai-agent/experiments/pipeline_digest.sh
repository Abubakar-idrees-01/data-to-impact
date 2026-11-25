#!/usr/bin/env bash
# E4 digest claim support. 20b now (own daily quota, runs beside the 120b pipeline), 120b after it. Launch detached:
#   setsid nohup bash experiments/pipeline_digest.sh >> logs/pipeline_digest.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "PIPELINE-DIGEST START $(date)"
$PY experiments/digest_eval.py run --runs 8 --model openai/gpt-oss-20b
$PY experiments/digest_eval.py judge --model openai/gpt-oss-20b
until grep -q "PIPELINE-120B-NEXT DONE" logs/pipeline_120b_next.log 2>/dev/null; do sleep 300; done
$PY experiments/digest_eval.py run --runs 8 --model openai/gpt-oss-120b
$PY experiments/digest_eval.py judge --model openai/gpt-oss-120b
echo "PIPELINE-DIGEST DONE $(date)"
