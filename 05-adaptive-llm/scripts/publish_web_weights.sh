#!/usr/bin/env bash
# Upload the browser demo's exported weights (site/models/, site/tokenizer.json) as files of a GitHub Release.
# The weights stay out of git; the Pages workflow (.github/workflows/pages.yml) downloads them when it deploys
# site/ to https://abubakar-idrees-01.github.io/data-to-impact/tinygpt/.
#   PYTHONPATH=src .venv/bin/python -m tinygpt.export_web && bash scripts/publish_web_weights.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="tinygpt-web-v1"
FILES=(site/models/*.json site/models/*.bin site/tokenizer.json)     # *.json includes index.json
if ! gh release view "$TAG" >/dev/null 2>&1; then
  gh release create "$TAG" --title "Tiny GPT browser demo: int8 weights" --notes \
"Weights for the in-browser demo of project 5 (05-adaptive-llm/site): dense, 8-expert MoE and early-exit 6-layer GPTs
trained from scratch on TinyStories, exported by \`python -m tinygpt.export_web\` as int8 with one float32 scale per
row (validation loss +0.0001 to +0.0004 vs float32). Format: see src/tinygpt/export_web.py. Not meant to be used
directly: the GitHub Pages workflow downloads these files when it deploys the demo."
fi
gh release upload "$TAG" "${FILES[@]}" --clobber
gh release view "$TAG" --json assets --jq '.assets[] | "\(.name)\t\(.size)"'
