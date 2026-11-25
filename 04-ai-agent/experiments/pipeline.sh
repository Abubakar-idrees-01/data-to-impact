#!/usr/bin/env bash
# Unattended evaluation pipeline. Launch DETACHED so it survives the terminal/session:
#   setsid nohup bash experiments/pipeline.sh > logs/pipeline.log 2>&1 < /dev/null & disown
# Every step resumes where it stopped, so rerunning after an interruption loses nothing.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "PIPELINE START $(date)"
$PY experiments/simpleqa.py run --condition closed --n 40
$PY experiments/simpleqa.py run --condition agent --n 40
$PY experiments/simpleqa.py grade
echo "PIPELINE DONE $(date)"
