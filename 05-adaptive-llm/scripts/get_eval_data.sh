#!/usr/bin/env bash
# Part B routing evaluation: the GSM8K test set (grade-school maths word problems; OpenAI, MIT licence), 1,319
# problems, 0.7 MB, into data/raw/ (gitignored). The other task families are generated in adaptive/tasks.py.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
curl -fL --retry 5 -o data/raw/gsm8k_test.jsonl \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
wc -l data/raw/gsm8k_test.jsonl
