# Results

Measured on the development laptop: Intel i5-5200U (2015, 2 cores / 4 threads), no GPU.

## 1. Tests
15 tests, ~9 s (`python -m pytest -q`):

| level | what runs | count |
|---|---|---|
| From-scratch protocol | raw JSON-RPC (handshake, version negotiation, validation, errors, stdio framing) and the **official SDK client against the scratch server** over a real subprocess | 6 |
| Server (in-process) | every tool, resource and prompt through the official client, with fake models; path-traversal attempts; index reload after an external sync | 8 |
| End to end | the installed server launched over stdio exactly like Claude Code does, with the real embedding and reranker models on the public sample index | 1 |

## 2. Latency
Over real stdio, public sample index (10 Wikipedia articles, 250 chunks), 10 queries
([`scripts/benchmark.py`](../scripts/benchmark.py), raw numbers in [`benchmark.json`](benchmark.json)):

| step | time |
|---|---|
| start Python, connect, handshake | 1.8 s |
| `tools/list` | 9 ms |
| first tool call (loads embedding + reranker models, once) | 14.1 s |
| `search_notes` (hybrid search + reranking), median / max | 0.94 s / 1.21 s |
| `check_answer` (per-sentence verification), median | 1.26 s |
| `list_documents`, median | 5 ms |

Searching is dominated by the cross-encoder reranker (≈ 100 ms per candidate on this CPU; see project 1's D13).

## 3. Behaviour on the owner's real notes
Checked by hand through the server, on 6 documents (3 lecture decks with OCR'd slides, 2 PDFs, a course outline):

| check | result |
|---|---|
| "what is a market space" | Lec 1, Slides 4, 3 and 5 (the definition slides), `found: true`, relevance 1.0 |
| an off-topic question | `found: false` (relevance below 0.2), so the model is told the notes don't cover it |
| `check_answer`, true sentence cited to the right slide | supported (0.745) |
| `check_answer`, invented sentence ("invented in 1850 by Napoleon") | unsupported (0.0) |
| `check_answer`, true sentence cited to the *wrong* slide | unsupported: it correctly catches a misattributed citation |
| `get_passage` for slide 999 | clear error listing the valid range (1–23) |
| `eval_run_summary("../../etc/passwd")` | refused, nothing read |

## 4. Problems found while using it
| problem | cause | fix |
|---|---|---|
| Claude Code: "Connection closed" | The package wasn't installed in its venv; the e2e test passed `PYTHONPATH` and hid it | Install step documented; test launches with a clean environment (D8) |
| A document "disappeared" from search | It had been removed from the index in the StudyRAG app; the running server also never reloaded | Re-sync restored it; the server now reloads the index when it changes (D7) |
| `python -m studyrag` failed in project 1's own venv | StudyRAG was never installed as a package there | Install step added to project 1's setup guide |
