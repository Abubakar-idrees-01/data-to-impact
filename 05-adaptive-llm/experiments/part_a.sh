#!/usr/bin/env bash
# Part A experiments, one after another (two cores: parallel runs would only slow each other down).
# Every run resumes from its checkpoint, so rerunning this script after an interruption loses at most one eval
# interval. Launch detached:
#   setsid nohup bash experiments/part_a.sh >> logs/part_a.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH=src
COMMON="--steps 1500 --batch 32 --ctx 256 --d 256 --n-layer 6 --n-head 8 --eval-every 250"
echo "PART-A START $(date)"
while pgrep -f "tinygpt.train --name dense-d256-L6" >/dev/null; do sleep 60; done     # the baseline, if running
$PY -m tinygpt.train --name dense-d256-L6 $COMMON                                    # (resumes / finishes it)
$PY -m tinygpt.train --name moe-4x-top1 $COMMON --n-experts 4 --top-k 1 --expert-mult 4
$PY -m tinygpt.train --name moe-8x-top2 $COMMON --n-experts 8 --top-k 2 --expert-mult 2
$PY -m tinygpt.train --name exit-d256-L6 $COMMON --exit-weight 1.0
$PY -m tinygpt.early_exit --name dense-d256-L6
$PY -m tinygpt.early_exit --name exit-d256-L6
echo "PART-A DONE $(date)"
