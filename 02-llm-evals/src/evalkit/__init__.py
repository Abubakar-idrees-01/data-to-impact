"""evalkit: a small, dependency-free toolkit for evaluating LLM systems.

- dataset   test cases in JSONL
- metrics   text, retrieval, agreement (judge vs human) and statistics with confidence intervals
- runs      every evaluation run saved as one JSON file, so versions can be compared later
- compare   paired comparison of two runs, with significance tests
- gate      fail CI when a metric regresses
- llm/judge LLM-as-judge over any OpenAI-compatible API, cached and rate-limit aware
- report    a self-contained HTML report with inline SVG charts
"""

__version__ = "0.1.0"
