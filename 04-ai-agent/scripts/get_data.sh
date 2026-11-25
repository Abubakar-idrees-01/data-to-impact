#!/usr/bin/env bash
# Downloads the public evaluation data into data/raw/ (gitignored). SimpleQA: OpenAI, MIT licence.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
curl -sSfL -o data/raw/simple_qa_test_set.csv https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv
curl -sSfL -o data/raw/simpleqa_eval.py https://raw.githubusercontent.com/openai/simple-evals/main/simpleqa_eval.py
echo "ok: $(wc -l < data/raw/simple_qa_test_set.csv) lines"
