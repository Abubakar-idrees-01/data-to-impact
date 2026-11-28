#!/usr/bin/env bash
# Runs after part_a.sh, before the Part B measurements:
# 1. retrain moe-4x-top1 with the top-1 gating fix (the first run, kept as models/moe-4x-top1-bug, had a router that
#    got no gradient from the language-model loss: see docs/concepts/03-mixture-of-experts.md);
# 2. write the final sample stories of every Part A model (same prompt, seed and settings as the dense samples);
# 3. hand over to the Part B queue.
# Launch detached:
#   setsid nohup bash experiments/part_a_rerun.sh >> logs/part_a.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH=src
COMMON="--steps 1500 --batch 32 --ctx 256 --d 256 --n-layer 6 --n-head 8 --eval-every 250"
until grep -q "PART-A DONE" logs/part_a.log 2>/dev/null; do
  pgrep -f "experiments/part_a.sh" >/dev/null || { echo "part_a.sh is not running and not done: stopping"; exit 1; }
  sleep 120
done
echo "PART-A RERUN START $(date)"
$PY -m tinygpt.train --name moe-4x-top1 $COMMON --n-experts 4 --top-k 1 --expert-mult 4 || exit 1
for m in moe-4x-top1-bug moe-4x-top1 moe-8x-top2 exit-d256-L6; do
  $PY -m tinygpt.sample --name "$m" --n 3 > "results/samples/$m-final.txt"
done
echo "PART-A RERUN DONE $(date)"
exec bash experiments/part_b.sh >> logs/part_b.log 2>&1 < /dev/null
