"""Record a real MCP session for the docs: the installed server over stdio, on project 1's PUBLIC sample index.

    python scripts/demo.py     # writes docs/DEMO.md and docs/images/demo.html (screenshot it for demo.png)

The draft answer below was written by Claude (in Claude Code) from the passages that search_notes returns for
this question, with one invented sentence added on purpose so the check has something to catch.
Every tool result shown is the server's real output.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_INDEX = ROOT.parent / "01-rag-from-scratch" / "eval" / ".cache" / "sample-structure"
QUESTION = "How are affiliates paid in affiliate marketing?"
DRAFT = ("Affiliates receive a commission for each visit, signup or sale they generate for a merchant [2]. "
         "The commission is usually a percentage of the price of the product, but it can also be a flat rate "
         "per referral [2]. Eighty percent of affiliate programs use revenue sharing or pay per sale, and nineteen "
         "percent use cost per action [3]. Most affiliates are also paid a fixed monthly salary by the merchant [1].")
OFF_TOPIC = "What is the boiling point of mercury?"


async def record() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"STUDYRAG_STORAGE_DIR": str(SAMPLE_INDEX)}
    async with Client(StdioServerParameters(command=sys.executable, args=["-m", "studyrag_mcp"], env=env),
                      read_timeout_seconds=300) as c:
        search = (await c.call_tool("search_notes", {"query": QUESTION, "top_k": 3})).structured_content
        ids = [p["chunk_id"] for p in search["passages"]]
        check = (await c.call_tool("check_answer", {"answer": DRAFT, "chunk_ids": ids})).structured_content
        off = (await c.call_tool("search_notes", {"query": OFF_TOPIC, "top_k": 2})).structured_content
    return {"search": search, "check": check, "off": off}


def markdown(rec: dict) -> str:
    s, ch, off = rec["search"], rec["check"], rec["off"]
    lines = ["# Demo: a real session", "",
             "The installed server over stdio, on StudyRAG's **public sample corpus** (Wikipedia articles). "
             "Tool results are real; the draft answer was written by Claude from the returned passages, with one "
             "invented sentence added on purpose. Re-create with `python scripts/demo.py`.", "",
             f"**User:** {QUESTION}", "", f"**→ `search_notes`**: found = {s['found']}, best relevance = "
             f"{s['best_relevance']}", ""]
    for p in s["passages"]:
        lines.append(f"- **[{p['n']}] {p['citation']}** (relevance {p['relevance']}): "
                     f"{p['text'].split(chr(10), 1)[-1][:220].strip()}…")
    lines += ["", "**Draft answer:**", "", f"> {DRAFT}", "", f"**→ `check_answer`**: supported ratio = "
              f"{ch['supported_ratio']}", ""]
    for x in ch["sentences"]:
        lines.append(f"- {'✅' if x['supported'] else '❌'} ({x['support']:.2f}) {x['sentence']}")
    final = DRAFT.rsplit(" Most affiliates", 1)[0]
    lines += ["", "**Final answer** (the unsupported sentence removed):", "", f"> {final}", "",
              f"**User:** {OFF_TOPIC}", "", f"**→ `search_notes`**: found = {off['found']}, best relevance = "
              f"{off['best_relevance']}: *\"{off['guidance']}\"*", ""]
    return "\n".join(lines)


def page(rec: dict) -> str:
    s, ch = rec["search"], rec["check"]
    esc = html.escape
    rows = "".join(f"<div class='p'><b>[{p['n']}] {esc(p['citation'])}</b> <span class='m'>relevance "
                   f"{p['relevance']}</span><br>{esc(p['text'].split(chr(10), 1)[-1][:200].strip())}…</div>"
                   for p in s["passages"])
    checks = "".join(f"<div class='c {'ok' if x['supported'] else 'no'}'>{'✓' if x['supported'] else '✗'} "
                     f"<span class='m'>{x['support']:.2f}</span> {esc(x['sentence'])}</div>" for x in ch["sentences"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#f6f6f4;font:15px/1.55 "Noto Sans",system-ui,sans-serif;color:#1b1b1a}}
main{{max-width:880px;margin:0 auto;padding:22px}}.u{{background:#2a78d6;color:#fff;padding:10px 14px;border-radius:12px;
display:inline-block;margin:6px 0 12px}}.t{{font:13px "DejaVu Sans Mono",monospace;color:#52514e;margin:10px 0 6px}}
.box{{background:#fff;border:1px solid #e3e2dc;border-radius:10px;padding:10px 14px;margin:6px 0 14px}}
.p{{margin:6px 0;font-size:13.5px}}.m{{color:#8a8984;font-size:12px}}.c{{margin:4px 0;font-size:13.5px}}
.ok{{color:#0b7a3e}}.no{{color:#c2372f}}h1{{font-size:18px;margin:0 0 10px}}</style></head><body><main>
<h1>studyrag-mcp · a real tool session (public sample corpus)</h1>
<div class="u">{esc(QUESTION)}</div>
<div class="t">→ search_notes(query, top_k=3) · found: {s['found']} · best relevance {s['best_relevance']}</div>
<div class="box">{rows}</div>
<div class="t">→ check_answer(draft, chunk_ids) · supported ratio {ch['supported_ratio']}</div>
<div class="box">{checks}</div>
<div class="t">The model removes the unsupported sentence before answering.</div></main></body></html>"""


if __name__ == "__main__":
    rec = asyncio.run(record())
    (ROOT / "docs" / "DEMO.md").write_text(markdown(rec), encoding="utf-8")
    (ROOT / "docs" / "images").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "images" / "demo.html").write_text(page(rec), encoding="utf-8")
    print(json.dumps({"found": rec["search"]["found"], "supported_ratio": rec["check"]["supported_ratio"],
                      "unsupported": rec["check"]["unsupported"], "off_topic_found": rec["off"]["found"]}, indent=1))
