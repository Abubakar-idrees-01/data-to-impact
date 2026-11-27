#!/usr/bin/env bash
# Part B measurements, run after the Part A training queue (they need an idle CPU: a busy core changes every number).
# Every step is resumable or cheap to repeat. Launch detached:
#   setsid nohup bash experiments/part_b.sh >> logs/part_b.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH=src
echo "PART-B START $(date)"
until grep -q "PART-A DONE" logs/part_a.log 2>/dev/null; do
  pgrep -f part_a.sh >/dev/null || { echo "part_a.sh is not running and not done: stopping"; exit 1; }
  sleep 120
done
$PY -m adaptive.bench_models                                          # speeds, RSS, cache reuse, calibration
$PY -m adaptive.plan                                                  # the plan, now calibrated
$PY -m adaptive.route_eval run                                        # every model × 128 questions
$PY -m adaptive.route_eval export                                     # held-out headline policies -> evalkit runs
$PY -m adaptive.speculative --draft Qwen3.5-0.8B --target Qwen3.5-2B --k 2 4 6
$PY -m adaptive.speculative --draft qwen2.5-0.5b --target qwen2.5-1.5b --k 2 4 6
echo "PART-B DONE $(date)"
