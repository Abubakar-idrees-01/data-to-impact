#!/usr/bin/env bash
# Runs after experiments/pipeline.sh (SimpleQA) finishes: E2 and E3 on the main model. Launch detached.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=openai/gpt-oss-120b
until grep -q "PIPELINE DONE" logs/pipeline.log 2>/dev/null; do sleep 300; done
echo "PIPELINE-120B-NEXT START $(date)"
$PY experiments/injection.py --condition hardened --model $M
$PY experiments/injection.py --condition plain --model $M
$PY experiments/checkup_eval.py --runs 3 --model $M
$PY scripts/build_site.py
echo "PIPELINE-120B-NEXT DONE $(date)"
