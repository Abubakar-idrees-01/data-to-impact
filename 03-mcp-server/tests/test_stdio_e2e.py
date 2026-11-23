"""End to end: launch the real server the way Claude Code does (a subprocess over stdio), on the PUBLIC sample
index built by project 1's evaluation (Wikipedia articles), with the real embedding and reranker models.

Skipped when that index or the models aren't present (e.g. in CI)."""

import asyncio
import os
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

RAG = Path(__file__).resolve().parents[2] / "01-rag-from-scratch"
SAMPLE_INDEX = RAG / "eval" / ".cache" / "sample-structure"
MODELS = RAG / "models" / "fastembed"

pytestmark = pytest.mark.skipif(not (SAMPLE_INDEX / "manifest.json").exists() or not MODELS.exists(),
                                reason="needs project 1's sample index and downloaded models")


def test_real_server_over_stdio():
    async def run():
        # No PYTHONPATH: the server must start the way a real client launches it (the package must be installed;
        # an earlier version of this test passed sys.path along and hid exactly that bug).
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"STUDYRAG_STORAGE_DIR": str(SAMPLE_INDEX)}
        params = StdioServerParameters(command=sys.executable, args=["-m", "studyrag_mcp"], env=env)
        async with Client(params, read_timeout_seconds=180) as c:
            init_tools = await c.list_tools()
            hit = await c.call_tool("search_notes", {"query": "How are affiliates paid?", "top_k": 3})
            off = await c.call_tool("search_notes", {"query": "What is the boiling point of mercury?"})
            return init_tools, hit, off
    tools, hit, off = asyncio.run(run())
    assert len(tools.tools) == 8
    s = hit.structured_content
    assert s["found"] and s["best_relevance"] > 0.5 and "Affiliate" in s["passages"][0]["citation"]
    assert off.structured_content["found"] is False            # calibrated refusal threshold, from project 1
