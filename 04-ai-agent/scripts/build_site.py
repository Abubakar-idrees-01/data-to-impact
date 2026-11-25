"""Build the trace viewer (site/index.html): real recorded runs, replayed step by step, plus the evaluation results.

1. Copies the chosen traces from .scout/traces/ (gitignored) to docs/demo-traces/ (committed), scrubbing account
   identifiers. Only public-content runs are chosen: nothing that searched the owner's private notes.
2. Writes one self-contained HTML page with the traces and results/*.json embedded.

    python scripts/build_site.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / ".scout" / "traces"
DEMO = ROOT / "docs" / "demo-traces"
SITE = ROOT / "site"

# (file under .scout/traces, short title, what to notice)
CHOSEN = [
    ("20261125-153200-bc3170.jsonl", "Research question: what changed in MCP?",
     "A clean run: one search, one focused page read, an answer with a numbered source. 3 model calls, 4.3k tokens. "
     "(It predates the citation fix: gpt-oss wrote its trained format 【1†L13-L22】, which is now normalised to [1].)"),
    ("20261125-155318-efb634.jsonl", "Digest #1: the request that was too big",
     "Every call resends the whole conversation. After four pages it reached 9,042 tokens, over the 8,000 the free tier "
     "accepts per request, and the run died. Fix: trim the oldest tool results before each call (concept 4)."),
    ("20261125-160230-ad0a9d.jsonl", "Digest #3: a draft I declined",
     "It worked mechanically, but the note rested on one news article and dated a 2026 change to 'June 2024' (the model "
     "anchored on its training year). Fix: at least two read sources, today's date in the task, recent primary sources first."),
    ("20261125-162310-cf2fee.jsonl", "Digest #6: primary sources, fallback model",
     "The main model's daily quota was used up, so it switched to gpt-oss-20b. Its first note was refused for citing an unread "
     "source; it read it and the second attempt passed. Sources: the official MCP blog and GitHub."),
    ("injection/post-comment", "Injection: a page tells the agent to post a comment",
     "The model tried to obey (it called post_comment). The approval gate declined, so nothing was posted. "
     "This is why permissions are enforced in code, not left to the prompt (concept 3)."),
]


def scrub(text: str) -> str:
    text = re.sub(r"org_[A-Za-z0-9]{10,}", "org_…", text)
    text = re.sub(r"(gsk|tvly|pypi|sk)-[A-Za-z0-9_\-]{12,}", r"\1-…", text)
    return text


def find_injection(case: str) -> Path | None:
    """The injection trace for a case: from the newest injection result file that has it."""
    for out in sorted((ROOT / "experiments" / "out").glob("injection-hardened-*.jsonl"), reverse=True):
        for line in out.read_text().splitlines():
            rec = json.loads(line)
            if rec["id"] == case and rec.get("trace") and Path(rec["trace"]).exists():
                return Path(rec["trace"])
    return None


def collect() -> list[dict]:
    DEMO.mkdir(parents=True, exist_ok=True)
    runs = []
    for i, (name, title, note) in enumerate(CHOSEN, 1):
        src = find_injection(name.split("/", 1)[1]) if name.startswith("injection/") else TRACES / name
        dest = DEMO / f"{i:02d}-{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:50]}.jsonl"
        if src and src.exists():
            dest.write_text(scrub(src.read_text(encoding="utf-8")), encoding="utf-8")
        if not dest.exists():
            print(f"skipped (not recorded yet): {title}")
            continue
        events = [json.loads(l) for l in dest.read_text(encoding="utf-8").splitlines()]
        runs.append({"title": title, "note": note, "file": dest.name, "events": events})
    return runs


def results() -> dict:
    out = {}
    for p in sorted((ROOT / "results").glob("*/*.json")):
        r = json.loads(p.read_text())
        out[f"{r['suite']}/{r['name']}"] = {"metrics": r["metrics"], "config": r.get("config", {}), "n": len(r["items"])}
    return out


def main() -> None:
    runs = collect()
    data = json.dumps({"runs": runs, "results": results()}, ensure_ascii=False).replace("</", "<\\/")
    html = (Path(__file__).with_name("site_template.html").read_text(encoding="utf-8")).replace("/*DATA*/null", data)
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(html, encoding="utf-8")
    print(f"site/index.html: {len(runs)} runs, {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
