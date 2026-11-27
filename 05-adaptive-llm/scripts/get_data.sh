#!/usr/bin/env bash
# TinyStories V2 (stories written by GPT-4; CDLA-Sharing-1.0) from Hugging Face, into data/raw/ (gitignored).
# Only the first TRAIN_MB of the 2.2 GB train file is fetched (HTTP Range): a CPU can't use more in a few hours.
# The last story in that slice is cut mid-way; data.py drops it.
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN_MB=${TRAIN_MB:-200}
BASE=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
mkdir -p data/raw
curl -fL --retry 5 -C - -o data/raw/valid.txt "$BASE/TinyStoriesV2-GPT4-valid.txt"
curl -fL --retry 5 -r 0-$((TRAIN_MB * 1024 * 1024 - 1)) -o data/raw/train.txt "$BASE/TinyStoriesV2-GPT4-train.txt"
ls -l data/raw
