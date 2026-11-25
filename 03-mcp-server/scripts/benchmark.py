"""Latency of the real server over stdio, the way an AI client experiences it.

    python scripts/benchmark.py            # uses project 1's PUBLIC sample index (Wikipedia articles)

Measures: process start + handshake, the first tool call (loads the embedding and reranker models), then
repeated search_notes / check_answer / list_documents calls. Writes docs/benchmark.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_INDEX = ROOT.parent / "01-rag-from-scratch" / "eval" / ".cache" / "sample-structure"
QUERIES = ["How are affiliates paid?", "What is search engine optimization?", "What is email segmentation?",
           "How does pay per click advertising work?", "What is content marketing?", "What is a backlink?",
           "What is influencer marketing?", "What does a marketing mix include?", "What is social media marketing?",
           "How is the open rate of an email measured?"]


def ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


async def run() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"STUDYRAG_STORAGE_DIR": str(SAMPLE_INDEX)}
    params = StdioServerParameters(command=sys.executable, args=["-m", "studyrag_mcp"], env=env)
    t0 = time.perf_counter()
    async with Client(params, read_timeout_seconds=300) as c:
        t_connect = time.perf_counter() - t0
        t = time.perf_counter()
        await c.list_tools()
        t_list_tools = time.perf_counter() - t
        t = time.perf_counter()
        await c.call_tool("list_documents", {})
        t_first = time.perf_counter() - t
        search, check, docs = [], [], []
        for q in QUERIES:
            t = time.perf_counter()
            r = await c.call_tool("search_notes", {"query": q, "top_k": 4})
            search.append(time.perf_counter() - t)
            ids = [p["chunk_id"] for p in r.structured_content["passages"]]
            answer = " ".join(p["text"].split(". ")[0] + f" [{n}]." for n, p in
                              enumerate(r.structured_content["passages"][:2], 1))
            t = time.perf_counter()
            await c.call_tool("check_answer", {"answer": answer, "chunk_ids": ids})
            check.append(time.perf_counter() - t)
            t = time.perf_counter()
            await c.call_tool("list_documents", {})
            docs.append(time.perf_counter() - t)
    return {"machine": f"{platform.processor() or platform.machine()}, {os.cpu_count()} threads",
            "index": "project 1 public sample corpus (10 Wikipedia articles, 250 chunks)",
            "start_and_handshake_ms": ms(t_connect), "list_tools_ms": ms(t_list_tools),
            "first_call_loads_models_ms": ms(t_first),
            "search_notes_median_ms": ms(statistics.median(search)), "search_notes_max_ms": ms(max(search)),
            "check_answer_median_ms": ms(statistics.median(check)),
            "list_documents_median_ms": ms(statistics.median(docs)), "n_queries": len(QUERIES)}


if __name__ == "__main__":
    result = asyncio.run(run())
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "benchmark.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
