"""E4. Are the learning-log notes true to their sources? Claim support in `scout digest` notes.

The digest routine already refuses notes that cite sources the agent didn't read (or fewer than two). This checks
the part code can't: does each sentence actually say what its sources say?

1. `run`: the digest routine runs exactly as in real use (topic rotates; the approver declines, so nothing is
   written). Every search snippet and every passage web_read returned is recorded per source: that is all the
   evidence the agent had.
2. `judge`: each note is split into claims (bullets and sentences). Every claim is checked by evalkit's judge, the
   configuration selected and validated on RAGTruth (faithfulness/direct on qwen3.8-27b, a different model family
   from the writer; test kappa 0.625 against human labels):
     - a claim with citations is checked against the sources it cites   -> citation support
     - an uncited claim in "What happened" / "Key details" is checked against everything the agent read
                                                                           -> grounding
   "Why it matters" and "Try it" are advice, not facts about the sources: judged only where they cite something.
3. Sensitivity probe: in claims the judge accepted, one number is changed (a year by -2, other numbers x3)
   and the judge must now reject the claim. If it doesn't, its "supported" verdicts on these notes mean little.

    python experiments/digest_eval.py run --runs 8 --model openai/gpt-oss-20b
    python experiments/digest_eval.py judge --model openai/gpt-oss-20b

Runs resume where they stopped and wait for the daily quota instead of switching models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "02-llm-evals" / "src"))

from scout.approval import DenyAll                   # noqa: E402
from scout.config import Settings                    # noqa: E402
from scout.routines import digest                    # noqa: E402
from scout.web import Web                            # noqa: E402

OUT = ROOT / "experiments" / "out"
RESULTS = ROOT / "results"
EVALKIT = ROOT.parent / "02-llm-evals"
SELECTED = EVALKIT / "results" / "judge-ragtruth-dev" / "selected.json"
FACT_SECTIONS = ("what happened", "key details")
MAX_SOURCE_CHARS = 6000


# ------------------------------------------------------------------ recording what the agent saw
class RecordingWeb(Web):
    """Web tools that remember every snippet and passage they returned, per source number."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.snippets: dict[int, str] = {}
        self.passages: dict[int, list[str]] = {}

    async def search(self, query: str, max_results: int = 5) -> dict:
        out = await super().search(query, max_results)
        for r in out.get("results", []):
            self.snippets.setdefault(r["n"], r["snippet"])
        return out

    async def read_page(self, source: str, question: str = "") -> dict:
        out = await super().read_page(source, question)
        seen = self.passages.setdefault(out["n"], [])
        if out["text"] not in seen:
            seen.append(out["text"])
        return out

    def evidence(self) -> list[dict]:
        return [{"n": s.n, "url": s.url, "title": s.title, "read": s.read, "snippet": self.snippets.get(s.n, ""),
                 "passages": self.passages.get(s.n, [])} for s in self.book.sources]


# ------------------------------------------------------------------ notes -> claims
_CITE = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+")
_SENT_END = re.compile(r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<!\bal\.)(?<=[.!?])\s+(?=[A-Z0-9\"“(*`\[])")


def cites_of(text: str) -> list[int]:
    return sorted({int(n) for g in _CITE.findall(text) for n in re.split(r"\s*,\s*", g)})


def strip_cites(text: str) -> str:
    return re.sub(r"\s+([.,;:])", r"\1", _CITE.sub("", text)).strip()


def sentences(paragraph: str) -> list[str]:
    """Split on sentence ends; a citation right after a full stop ("... done. [2] Next") stays with its sentence."""
    parts = _SENT_END.split(paragraph.strip())
    out: list[str] = []
    for p in parts:
        m = re.match(r"^((?:\[\d+(?:\s*,\s*\d+)*\]\s*)+)(.*)$", p, re.S)
        if m and out:
            out[-1] += " " + m.group(1).strip()
            p = m.group(2)
        if p.strip():
            out.append(p.strip())
    return out


def note_claims(md: str) -> list[dict]:
    """Claims in a note: every bullet, and every sentence of a paragraph, with the section it's in."""
    body = re.split(r"\n## Sources\b", md, maxsplit=1)[0]
    claims, section = [], ""
    for block in re.split(r"\n\s*\n", body):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        if lines[0].startswith("# ") or (lines[0].startswith("*") and "drafted by Scout" in lines[0]):
            continue
        if lines[0].startswith("##"):
            section = lines[0].lstrip("#").strip().lower()
            lines = lines[1:]
        units: list[str] = []
        para: list[str] = []
        for l in lines:
            if _BULLET.match(l):
                if para:
                    units += sentences(" ".join(para))
                    para = []
                units.append(_BULLET.sub("", l).strip())
            elif units and not para and l.startswith("  "):      # a bullet's continuation line
                units[-1] += " " + l.strip()
            else:
                para.append(l.strip())
        if para:
            units += sentences(" ".join(para))
        for u in units:
            text = strip_cites(u).strip("*_ ")
            if len(text.split()) < 4:
                continue
            cites = cites_of(u)
            factual = any(section.startswith(s) for s in FACT_SECTIONS)
            if cites or factual:
                claims.append({"section": section, "text": text, "cites": cites})
    return claims


# ------------------------------------------------------------------ sensitivity probe
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")


def corrupt(text: str) -> str | None:
    """Change one number: a year by -2 (else the first number x3). None if the claim has no number."""
    nums = list(_NUM.finditer(text))
    if not nums:
        return None
    years = [m for m in nums if re.fullmatch(r"(19|20)\d\d", m.group(1))]
    m = years[0] if years else nums[0]
    v = m.group(1)
    if years:
        new = str(int(v) - 2)
    elif "." in v:
        whole, frac = v.split(".")
        new = f"{int(whole) * 3 + 1}.{frac}"
    else:
        new = str(int(v) * 3 if int(v) > 1 else int(v) + 7)
    return text[:m.start(1)] + new + text[m.end(1):]


def source_text(evidence: list[dict], nums: list[int] | None) -> str:
    """The evidence the agent had for these sources (all read sources if nums is None)."""
    picked = [e for e in evidence if (e["n"] in nums if nums is not None else e["read"])]
    parts = []
    for e in picked:
        seen = "\n\n[...]\n\n".join(e["passages"]) or "(not read; only the search snippet was seen)"
        block = f"[{e['n']}] {e['title']} ({e['url']})\nSearch snippet: {e['snippet']}\nPage text read:\n{seen}"
        parts.append(block[:MAX_SOURCE_CHARS])
    return "\n\n=====\n\n".join(parts)


# ------------------------------------------------------------------ run digests
async def run(a) -> None:
    from simpleqa import wait_for_quota
    s = Settings.from_env()
    s.api_model, s.fallback_model = a.model, None
    s.state_dir = ROOT / ".scout" / "eval-digest"           # own memory: later runs avoid earlier runs' topics
    model = a.model.split("/")[-1]
    out = OUT / f"digest-{model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    done = len(old)
    # one step limit per results file: a resumed file keeps the limit it started with (12 before it was a setting)
    if old:
        s.digest_max_steps = old[0].get("max_steps", 12)
    elif a.max_steps:
        s.digest_max_steps = a.max_steps
    print(f"digest with {a.model}: {done}/{a.runs} already done -> {out.name} (step limit {s.digest_max_steps})",
          flush=True)
    for i in range(done + 1, a.runs + 1):
        topic = s.topics[(i - 1) % len(s.topics)]
        while True:
            web = RecordingWeb(s.tavily_key, page_chars=s.page_chars, time_range="month",
                               domains=s.digest_domains or None)
            t0 = time.time()
            r = await digest(s, DenyAll(), topic=topic, web=web, push=False, today=date.today())
            if r.run.stopped != "error":
                break
            if not await wait_for_quota(Exception(r.run.answer)):
                print(f"  run{i}: model error, stopping: {r.run.answer[:200]}")
                return
        note = next((c.new_text for c in (r.changes.changes if r.changes else [])
                     if c.path.endswith(".md") and not c.path.endswith("README.md")), "")
        rec = {"id": f"run{i}", "topic": topic, "model": a.model, "max_steps": s.digest_max_steps,
               "date": date.today().isoformat(),
               "written": bool(note), "title": r.title, "note": note, "evidence": web.evidence(),
               "sources_read": sum(e["read"] for e in web.evidence()), "searches": web.searches, "reads": web.reads,
               "tokens": r.run.total_tokens, "steps": r.run.steps, "seconds": round(time.time() - t0, 1),
               "stopped": r.run.stopped, "answer": r.run.answer[:500], "trace": r.run.trace_path}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  run{i} [{topic}]: {'note' if note else 'NO NOTE'} {r.title[:60]!r}  read {rec['sources_read']}  "
              f"{rec['tokens']:,} tok {rec['seconds']:.0f}s", flush=True)


# ------------------------------------------------------------------ judge the notes
def judge(a) -> None:
    from evalkit.judge import Judge
    from evalkit.llm import ChatClient
    from evalkit.runs import Run
    sel = json.loads(SELECTED.read_text())
    client = ChatClient.from_env(dotenv=EVALKIT / ".env", cache_dir=ROOT / ".cache" / "judge", model=sel["model"],
                                 extra_body=sel.get("extra_body"), max_wait=900, max_retries=1000)
    j = Judge(client, sel["rubric"], max_tokens=sel.get("max_tokens"))
    model = a.model.split("/")[-1]
    recs = [json.loads(l) for l in (OUT / f"digest-{model}.jsonl").read_text().splitlines()]
    claims_out = OUT / f"digest-claims-{model}.jsonl"
    items, rows = [], []
    for rec in recs:
        item = {"id": rec["id"], "topic": rec["topic"], "written": rec["written"], "title": rec["title"],
                "tokens": rec["tokens"], "sources_read": rec["sources_read"]}
        if not rec["written"]:
            items.append(item)
            print(f"  {rec['id']}: no note written")
            continue
        task = f'Write one statement for a research note titled "{rec["title"]}", using only the SOURCE.'
        verdicts = []
        for k, c in enumerate(note_claims(rec["note"]), 1):
            nums = c["cites"] or None
            src = source_text(rec["evidence"], nums)
            v = j(task=task, source=src, response=c["text"])
            row = {"run": rec["id"], "claim": k, **c, "kind": "cited" if c["cites"] else "uncited",
                   "supported": None if v.label is None else not v.label, "error": v.error}
            if row["supported"]:
                bad = corrupt(c["text"])
                if bad:
                    pv = j(task=task, source=src, response=bad)
                    row["probe"], row["probe_caught"] = bad, None if pv.label is None else bool(pv.label)
            verdicts.append(row)
        rows += verdicts

        def rate(rs):
            rs = [r for r in rs if r["supported"] is not None]
            return round(sum(r["supported"] for r in rs) / len(rs), 3) if rs else None
        cited = [r for r in verdicts if r["kind"] == "cited"]
        uncited = [r for r in verdicts if r["kind"] == "uncited"]
        probes = [r for r in verdicts if r.get("probe_caught") is not None]
        item |= {"claims": len(verdicts), "cited_claims": len(cited), "uncited_claims": len(uncited),
                 "claim_support": rate(verdicts), "citation_support": rate(cited), "grounding_uncited": rate(uncited),
                 "all_supported": all(r["supported"] for r in verdicts),
                 "probes": len(probes), "probe_caught": rate([{"supported": r["probe_caught"]} for r in probes]),
                 "unsupported": [r["text"] for r in verdicts if r["supported"] is False]}
        items.append(item)
        print(f"  {rec['id']}: {len(verdicts)} claims, support {item['claim_support']}, cited {item['citation_support']}, "
              f"uncited {item['grounding_uncited']}, probes caught {item['probe_caught']} ({len(probes)})", flush=True)
    claims_out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    # pooled over claims too: per-note means weight a 4-claim note like a 12-claim one
    pool = [r for r in rows if r["supported"] is not None]
    probes = [r for r in rows if r.get("probe_caught") is not None]
    run = Run.build("scout-digest", model, items,
                    metrics=["written", "claim_support", "citation_support", "grounding_uncited", "all_supported",
                             "probe_caught", "claims", "sources_read", "tokens"],
                    config={"writer": a.model, "judge": sel["name"], "judge_kappa_dev": sel["kappa"],
                            "claims_judged": len(pool),
                            "claims_supported_pooled": round(sum(r["supported"] for r in pool) / len(pool), 3) if pool else None,
                            "probes_pooled": len(probes),
                            "probe_caught_pooled": round(sum(r["probe_caught"] for r in probes) / len(probes), 3) if probes else None,
                            "judge_calls": client.usage.calls, "judge_cache_hits": client.usage.cache_hits,
                            "judge_tokens": client.usage.prompt_tokens + client.usage.completion_tokens})
    print("saved", run.save(RESULTS), "| claims", len(pool), "pooled support", run.config["claims_supported_pooled"],
          "| probes caught", run.config["probe_caught_pooled"], f"({len(probes)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", type=int, default=8)
    r.add_argument("--model", default="openai/gpt-oss-20b")
    r.add_argument("--max-steps", type=int, help="digest step limit for a new results file (default: the setting)")
    g = sub.add_parser("judge")
    g.add_argument("--model", default="openai/gpt-oss-20b")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    asyncio.run(run(a)) if a.cmd == "run" else judge(a)


if __name__ == "__main__":
    main()
