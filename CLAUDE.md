# Data to Impact — portfolio monorepo

Data Science & AI portfolio. Each numbered folder is an independent project with its own
CLAUDE.md (project context), README.md (public write-up), and requirements.

## Owner's machine (design for this, but every project must adapt to any hardware)
- Intel i5-5200U, 2 cores / 4 threads, AVX2, **no GPU**, 15 GB RAM, limited disk
- Python 3.12, Linux; repo lives on an NTFS drive (core.fileMode=false is set)

## Projects (numbered by build order; dependencies flow downward)
1. `01-rag-from-scratch` — document Q&A that answers only from uploaded docs, with citations
2. `02-llm-evals` — shared evaluation harness used by every other project
3. `03-mcp-server` — MCP server exposing the RAG + utility tools to any AI client
4. `04-ai-agent` — personal research/dev agent with human approval on outward actions
5. `05-adaptive-llm` — tiny GPT from scratch (MoE + early exit) + hardware-adaptive inference engine

## Workflow rules
- Commit in small, meaningful steps and **push to origin/main after each working step**
  (owner wants a steady GitHub contribution graph). Never make filler/fake commits.
- One virtualenv per project (`<project>/.venv`), pinned `requirements.txt`.
- Everything must run on CPU. Large data and model weights stay out of git.
- Model weights / HF cache go on this drive (project `models/`, gitignored), not `~` (root has ~8 GB free).

## Documentation standard (every project — owner learns from these docs)
- `README.md`: problem, live demo link, screenshots/GIF, features, results with numbers, quickstart.
- `docs/SETUP.md`: step-by-step install for Linux/Windows/macOS, model downloads, troubleshooting.
- `docs/HOW_IT_WORKS.md`: architecture + diagram, data flow end to end.
- `docs/concepts/*.md`: one beginner-friendly learning note per core concept, written so the owner
  can explain it in an interview (intuition → math/algorithm → our code → results).
- `docs/DECISIONS.md`: why X over Y, backed by our measurements.
- `docs/images/`: screenshots (captured headlessly with Chrome) and demo GIF.
- Write the docs for a phase in the same step as its code, not at the end.
