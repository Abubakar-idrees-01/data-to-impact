# Setup Guide

evalkit itself needs only **Python 3.11+**: no third-party packages, no GPU. An API key is needed only for the
LLM judge.

## 1. Get the code and create an environment
```bash
git clone https://github.com/Abubakar-idrees-01/data-to-impact.git
cd data-to-impact/02-llm-evals
python3 -m venv .venv
source .venv/bin/activate              # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt    # only pytest, for the tests
python -m pytest -q                    # 36 tests, ~3 s, no network
```
On Windows, use `py -3.12 -m venv .venv`. If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 2. Try the command line (no API key needed)
```bash
export PYTHONPATH=src                  # or: pip install -e .   (then the `evalkit` command exists)
python -m evalkit summary results/judge-ragtruth-dev/baseline-lexical.json
python -m evalkit compare results/studyrag-answers/v1.json results/studyrag-answers/v2.json
python -m evalkit gate results/studyrag-answers/v2.json --rules gates/studyrag.toml --baseline results/studyrag-answers/v1.json
```

## 3. Configure the LLM judge (optional)
Any OpenAI-compatible endpoint works: Groq, OpenAI, OpenRouter, or a local llama.cpp / Ollama server. Create a
file `.env` in `02-llm-evals/` (it is gitignored):
```
EVALKIT_API_BASE_URL=https://api.groq.com/openai/v1
EVALKIT_API_MODEL=qwen/qwen3.8-27b
EVALKIT_API_KEY=gsk_...
```
Providers retire models, so list what your key can use first:
```bash
curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $EVALKIT_API_KEY" | grep -o '"id":"[^"]*"'
```
**Offline alternative:** run a local OpenAI-compatible server (e.g. `llama-server -m model.gguf --port 8080`) and
set `EVALKIT_API_BASE_URL=http://localhost:8080/v1`. A small local model is a much weaker judge; validate it first.

Quick check:
```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from evalkit.llm import ChatClient
from evalkit.judge import Judge
j = Judge(ChatClient.from_env(), "faithfulness/direct")
print(j(task="Summarize.", source="Acme was founded in 1998 in Lyon.", response="Acme was founded in Paris.").label)  # True
EOF
```

## 4. Reproduce the judge validation
```bash
# RAGTruth (MIT licence, ~36 MB): downloaded, never committed
mkdir -p data/raw/ragtruth && cd data/raw/ragtruth
for f in response.jsonl source_info.jsonl; do
  curl -LO https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/$f
done
cd ../../..
python experiments/ragtruth.py                              # rebuilds the exact dev/test samples
python experiments/judge_validation.py --split dev --baselines   # no API needed
bash experiments/pipeline.sh > logs/pipeline.log 2>&1       # everything else, unattended
python experiments/build_report.py                          # -> site/index.html
```
**Time and quota:** the full study uses about 1M tokens. On Groq's free tier (≈200k tokens per model per day) it
takes about two days. The pipeline waits out the limits by itself, and every finished call is cached in `.cache/llm`, so
stopping and restarting loses nothing. Grading StudyRAG also needs `python experiments/studyrag_export.py` run with
project 1's environment (see that file).

## 5. Use evalkit in another project
```bash
pip install -e ../02-llm-evals          # from the other project's environment; no dependencies come with it
```
```python
from evalkit.runs import Run
from evalkit.metrics.text import token_f1
items = [{"id": q.id, "f1": token_f1(answer(q.input), q.expected)} for q in cases]
Run.build("my-suite", "v3", items).save("results")        # then: evalkit compare / gate
```
Or skip the import entirely: write one JSON line per item and run `evalkit import details.jsonl --suite S --name v3`.

## 6. Continuous integration
[`.github/workflows/evalkit.yml`](../../.github/workflows/evalkit.yml) runs the tests and the quality gate on
every push that touches this folder. It needs no API key: it checks committed runs.

## Troubleshooting
| Problem | Fix |
|---|---|
| `set EVALKIT_API_BASE_URL` | Create `.env` (step 3) in the folder you run from, or export the variables |
| `API error 404 … model … does not exist` | The provider retired the model: list models (step 3) and change `EVALKIT_API_MODEL` |
| Log shows `API 429 … tokens per day (TPD)` | Free daily quota used up: the experiment waits and continues by itself; or use another model (quotas are per model) |
| `empty reply: the … token budget ran out` | A reasoning model spent its budget thinking: pass `--max-tokens 2000 --effort low` |
| `unparseable judge output` | Counted as an error, never as a verdict; re-run the experiment (cached calls are free, failed ones retry) |
| `tomllib` not found | Python is older than 3.11; use 3.11+ |
