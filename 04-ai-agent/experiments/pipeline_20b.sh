#!/usr/bin/env bash
# Second pipeline: the same experiments on the fallback model (gpt-oss-20b, its own daily quota). Launch detached.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=openai/gpt-oss-20b
echo "PIPELINE-20B START $(date)"
$PY experiments/injection.py --condition hardened --model $M
$PY experiments/injection.py --condition plain --model $M
$PY experiments/checkup_eval.py --runs 2 --model $M
echo "PIPELINE-20B DONE $(date)"
