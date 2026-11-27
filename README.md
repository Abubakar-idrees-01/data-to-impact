# Data to Impact — Data Science & AI Portfolio

End-to-end Data Science and AI projects focused on solving real-world problems:
from problem framing and data collection to modeling, evaluation, and deployment.
Everything is designed to run on modest, CPU-only hardware.

**🌐 Live demo:** [evalkit report: can an LLM judge spot hallucinations like a human?](https://abubakar-idrees-01.github.io/data-to-impact/)

## Projects

| # | Project | Problem | Key ideas | Status |
|---|---------|---------|-----------|--------|
| 1 | [StudyRAG: RAG from Scratch](01-rag-from-scratch) | Trustworthy study assistant over your own course material | Hand-built BM25 + vector search, hybrid retrieval, OCR, verified citations, MCQ generation | ✅ v1 done: 100% top-3 retrieval, 14/14 refusals, 0.94 key-fact recall on real slides; on PyPI: `pip install studyrag` |
| 2 | [evalkit: LLM Evals](02-llm-evals) | "Is my AI system actually good?" | Metrics with confidence intervals, paired version comparison, CI quality gates, LLM judge validated against human labels | ✅ done: judge agrees with human hallucination labels at kappa 0.625 on 240 held-out answers (best non-LLM baseline 0.233) |
| 3 | [studyrag-mcp: MCP Server](03-mcp-server) | Give AI assistants access to private knowledge and tools | Model Context Protocol (official SDK + built from scratch), cited search, answer fact-checking, safe read-only tools | ✅ done: used from Claude Code; search ~1 s on a 2015 CPU; 16 tests incl. real stdio end to end; on PyPI: `pip install studyrag-mcp` |
| 4 | [Scout: AI Agent](04-ai-agent) | Automate research and dev chores safely | Agent loop from scratch, tool calling, MCP client, approval gate enforced in code, prompt-injection defence, context trimming | 🚧 working, evaluations running: [live trace viewer](https://abubakar-idrees-01.github.io/data-to-impact/scout/); injection attacks tried 2/10, executed 0/10 |
| 5 | [Adaptive LLM](05-adaptive-llm) | Fast local chat on weak hardware | Tiny GPT with MoE + early exit, hardware-aware routing, speculative decoding | 🚧 in progress (A2: baseline GPT trained) |

The projects build on each other: the agent uses the MCP server, the MCP server exposes the RAG
system, and every project is measured with the shared evals harness.

## Structure

Each project lives in its own folder with:

```
project-name/
├── README.md        # problem, approach, results, how to run
├── data/            # small samples only (raw data is gitignored)
├── notebooks/       # exploration & analysis
├── src/             # reusable code
├── models/          # saved models (gitignored)
└── requirements.txt
```
