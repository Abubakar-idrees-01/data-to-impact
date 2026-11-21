# Data to Impact — Data Science & AI Portfolio

End-to-end Data Science and AI projects focused on solving real-world problems:
from problem framing and data collection to modeling, evaluation, and deployment.
Everything is designed to run on modest, CPU-only hardware.

## Projects

| # | Project | Problem | Key ideas | Status |
|---|---------|---------|-----------|--------|
| 1 | [StudyRAG: RAG from Scratch](01-rag-from-scratch) | Trustworthy study assistant over your own course material | Hand-built BM25 + vector search, hybrid retrieval, OCR, verified citations, MCQ generation | ✅ v1 done: 100% top-3 retrieval, 14/14 refusals, 0.94 key-fact recall on real slides |
| 2 | [LLM Evals](02-llm-evals) | "Is my AI system actually good?" | Faithfulness, retrieval metrics, validated LLM-as-judge, regression tracking | 🚧 planned |
| 3 | [MCP Server](03-mcp-server) | Give AI assistants access to private knowledge and tools | Model Context Protocol, published package | 🚧 planned |
| 4 | [AI Agent](04-ai-agent) | Automate research and dev chores safely | Tool use, planning, memory, human-in-the-loop | 🚧 planned |
| 5 | [Adaptive LLM](05-adaptive-llm) | Fast local chat on weak hardware | Tiny GPT with MoE + early exit, hardware-aware routing, speculative decoding | 🚧 planned |

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
