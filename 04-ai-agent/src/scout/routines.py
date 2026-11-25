"""Routines: the real jobs the agent does, each ending in a ChangeSet you approve (or not).

- `digest`: research one recent development in your topics and draft a learning-log note, citing only sources
  the agent actually read. Labelled as drafted by the agent.
- `checkup`: look for concrete problems in the repo's docs (broken links, numbers that contradict each other,
  stale status lines) and propose small fixes, each with a reason.
- `review`: show changes prepared earlier by an unattended run, and apply the ones you approve.

Nothing here commits on a schedule by itself: an unattended run only *prepares* a proposal (`prepare=True`).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .agent import Agent, RunResult, normalize_citations
from .approval import Approver
from .changes import ChangeSet, edit_tools, slugify
from .config import Settings
from .llm import LLM
from .memory import Memory
from .repo import Repo, git
from .tools import Registry, Tool, ToolError
from .web import Web

LOG_INDEX_HEADER = """# Learning log

Short notes on recent developments in the areas this portfolio covers. Each note is **drafted by Scout**
(the research agent in [04-ai-agent](../04-ai-agent)) from the sources it read, then reviewed by me before it's
committed. Sources are listed at the end of every note.

"""


@dataclass
class RoutineResult:
    kind: str
    status: str                       # "applied" | "declined" | "prepared" | "nothing" | "failed"
    title: str = ""
    detail: dict | None = None
    run: RunResult | None = None
    pending_path: str | None = None
    changes: ChangeSet | None = None      # what was proposed (whether or not it was approved)


def _agent(settings: Settings, registry: Registry, llm: LLM | None, on_event, max_steps: int | None = None,
           token_budget: int | None = None) -> Agent:
    return Agent(llm or LLM.from_settings(settings), registry, max_steps=max_steps or settings.max_steps,
                 token_budget=token_budget or settings.token_budget, max_tokens_per_call=settings.max_tokens_per_call,
                 traces_dir=settings.traces_dir, on_event=on_event,
                 context_limit=settings.context_limit)


def repo_root_of(path: Path) -> Path:
    p = path if path.exists() else path.parent
    while not p.exists():
        p = p.parent
    return Path(git(p, "rev-parse", "--show-toplevel").strip())


def notify(title: str, body: str) -> None:
    """A desktop notification if `notify-send` exists (Linux); silently nothing otherwise."""
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "-a", "Scout", title, body], check=False, capture_output=True)


async def _finish(cs: ChangeSet, settings: Settings, approver: Approver, prepare: bool, push: bool) -> tuple[str, dict]:
    if not cs.changes:
        return "nothing", {}
    if prepare:
        path = cs.save(settings.state_dir / "pending")
        notify("Scout has a proposal for you", f"{cs.message}\nRun: scout review")
        return "prepared", {"pending": str(path)}
    out = await cs.approve_and_apply(approver, push=push)
    return out.pop("status"), out


# ------------------------------------------------------------------ digest
DIGEST_TASK = """Write today's learning-log note for me.

Today is {today}. "Recent" means published after {since}. Don't put a year in search queries unless a source
gave it to you, and take every date in the note from a source, never from memory.

1. Pick ONE specific, recent development in one of these areas: {topics}.
   Not something earlier notes covered. Earlier notes: {recent}.
2. Research it: search, then read at least two different sources (web_read) to check the facts. Searches look at
   primary sources first (papers, code, official docs and blogs); prefer those over secondary write-ups.
   Pick something concrete: a paper, a release, a spec change, a tool, not a general trend.
3. Call write_note exactly once, with a specific title (not "Recent advances in RAG") and a markdown body
   with these sections:
   ## What happened   (2-4 sentences)
   ## Why it matters for my projects   (I build RAG, LLM evaluation, MCP servers and small LLMs on a CPU laptop)
   ## Key details   (3-6 bullet points, each with its [n] citation)
   ## Try it   (one small concrete thing I could try, if there is one)
   Under 350 words. Cite only sources you read. Don't add a Sources section: it's added automatically.
4. Then reply with one line: the title."""


def digest_tools(cs: ChangeSet, web: Web, journal_dir: Path, today: date, repo_root: Path) -> list[Tool]:
    rel_dir = journal_dir.resolve().relative_to(repo_root).as_posix()

    def write_note(title: str, body: str) -> dict:
        body = normalize_citations(body)
        if any(c.path.endswith(".md") and c.path != f"{rel_dir}/README.md" for c in cs.changes):
            raise ToolError("a note was already written in this task; reply with its title")
        nums = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
        unknown = sorted(n for n in nums if web.book.get(n) is None)
        if unknown:
            raise ToolError(f"source(s) {unknown} don't exist; cite only numbers from your searches")
        unread = sorted(n for n in nums if not web.book.get(n).read)
        if unread:
            raise ToolError(f"you cited {unread} without reading them; web_read them first or remove those claims")
        if not nums:
            raise ToolError("the note has no citations; cite the sources you read as [n]")
        if len(nums) < 2:
            raise ToolError("cite at least two different sources you read (a second source confirms the facts); "
                            "search for and read another one, ideally a primary source")
        body = re.sub(r"\n#+\s*Sources\b.*", "", body.strip(), flags=re.S | re.I)   # we add our own list
        sources = "\n".join(f"{s.n}. [{s.title or s.url}]({s.url})" for s in web.book.sources if s.n in nums)
        title = title.strip().strip("#").strip()
        name = f"{today.isoformat()}-{slugify(title)}.md"
        text = (f"# {title}\n\n*{today.isoformat()} · drafted by Scout (my research agent) from the sources below, "
                f"reviewed by me*\n\n{body}\n\n## Sources\n{sources}\n")
        cs.put(f"{rel_dir}/{name}", text, "learning-log note")
        index_path = journal_dir / "README.md"
        index = cs.current(f"{rel_dir}/README.md") or LOG_INDEX_HEADER
        line = f"- {today.isoformat()} · [{title}]({name})\n"
        if "\n- " in index:                                    # newest first: insert before the first entry
            i = index.index("\n- ") + 1
            index = index[:i] + line + index[i:]
        else:
            index = index.rstrip("\n") + "\n\n" + line
        cs.put(f"{rel_dir}/README.md", index, "index")
        cs.message = f"Learning log: {title}"
        return {"saved_as": f"{rel_dir}/{name}", "note": "drafted; the user reviews it before it's committed"}

    return [Tool("write_note", "Save today's learning-log note (drafted; the user approves before it's committed).",
                 {"type": "object", "properties": {"title": {"type": "string", "minLength": 8},
                                                   "body": {"type": "string", "minLength": 200}},
                  "required": ["title", "body"]}, write_note)]


async def digest(settings: Settings, approver: Approver, *, topic: str | None = None, prepare: bool = False,
                 push: bool = True, llm: LLM | None = None, web: Web | None = None, on_event=None,
                 today: date | None = None) -> RoutineResult:
    today = today or date.today()
    memory = Memory(settings.memory_path)
    repo_root = repo_root_of(settings.journal_dir)
    web = web or Web(settings.tavily_key, page_chars=settings.page_chars, time_range="month",
                     domains=settings.digest_domains or None)
    cs = ChangeSet(str(repo_root), kind="digest")
    registry = Registry(approver).add(*web.tools(), *digest_tools(cs, web, settings.journal_dir, today, repo_root))
    recent = [e["title"] for e in memory.recent("digest", 30) if e.get("title")]
    # also the titles already in the log (memory may be fresh on a new machine)
    index = settings.journal_dir / "README.md"
    if index.exists():
        recent += re.findall(r"^- [\d-]+ · \[([^\]]+)\]", index.read_text(encoding="utf-8"), flags=re.M)
    since = date.fromordinal(today.toordinal() - 92).strftime("%B %Y")
    task = DIGEST_TASK.format(today=today.strftime("%d %B %Y"), since=since,
                              topics="; ".join([topic] if topic else settings.topics),
                              recent="; ".join(dict.fromkeys(recent)) or "none yet")
    run = await _agent(settings, registry, llm, on_event, max_steps=max(settings.max_steps, 12),
                       token_budget=max(settings.token_budget, 60_000)).run(task)
    title = cs.message.removeprefix("Learning log: ")
    if not cs.changes:
        memory.record("digest", status="failed", reason=run.answer[:300], tokens=run.total_tokens)
        return RoutineResult("digest", "failed", run=run, detail={"answer": run.answer})
    cs.summary = f"Today's learning-log note: {title}  ({run.tool_calls} tool calls, {run.total_tokens:,} tokens)"
    status, detail = await _finish(cs, settings, approver, prepare, push)
    memory.record("digest", status=status, title=title, tokens=run.total_tokens, trace=run.trace_path,
                  sources=[s.url for s in web.book.sources if s.read])
    return RoutineResult("digest", status, title, detail, run, detail.get("pending"), cs)


# ------------------------------------------------------------------ checkup
CHECKUP_TASK = """Do a careful checkup of this repository's documentation and propose small fixes.

Automatic checks already found these broken relative links: {links}

Look for concrete, verifiable problems in the markdown docs:
- the broken links above (find the right target with list_files, or remove the link);
- numbers that contradict each other between files (test counts, results, versions): check the source
  of truth (e.g. count the tests, read the results file) before changing anything;
- status lines that are out of date, and obvious typos.
{focus}
Rules: propose at most {max_fixes} fixes with propose_edit, only ones you've verified by reading the files.
Don't change code, results or meaning. If you find nothing worth fixing, say so: that's a fine outcome.
Finish with a short report: what you checked, each fix and why, and anything you noticed but didn't fix."""


async def checkup(settings: Settings, approver: Approver, *, focus: str = "", max_fixes: int = 3, prepare: bool = False,
                  push: bool = True, llm: LLM | None = None, on_event=None) -> RoutineResult:
    repo = Repo(settings.workspace)
    memory = Memory(settings.memory_path)
    cs = ChangeSet(str(repo.root), kind="checkup")
    registry = Registry(approver).add(*repo.tools(), *edit_tools(cs, repo, allowed="*.md"))
    links = repo.check_links()["broken"]
    task = CHECKUP_TASK.format(links=", ".join(f"{b['file']}:{b['line']} -> {b['link']}" for b in links) or "none",
                               focus=f"Focus on: {focus}\n" if focus else "", max_fixes=max_fixes)
    run = await _agent(settings, registry, llm, on_event, max_steps=max(settings.max_steps, 12),
                       token_budget=max(settings.token_budget, 60_000)).run(task)
    if cs.changes:
        reasons = [c.reason for c in cs.changes if c.reason]
        first = reasons[0] if reasons else "small fixes"
        cs.message = f"Docs checkup: {first[0].lower() + first[1:]}" + (f" (+{len(reasons) - 1} more)" if len(reasons) > 1 else "")
        cs.message = cs.message[:100]
        cs.summary = run.answer
    status, detail = await _finish(cs, settings, approver, prepare, push)
    memory.record("checkup", status=status, title=cs.message, tokens=run.total_tokens, trace=run.trace_path,
                  files=[c.path for c in cs.changes], links_broken=len(links))
    return RoutineResult("checkup", status, cs.message, {**detail, "report": run.answer}, run, detail.get("pending"), cs)


# ------------------------------------------------------------------ review
async def review(settings: Settings, approver: Approver, *, push: bool = True) -> list[RoutineResult]:
    folder = settings.state_dir / "pending"
    out = []
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        cs = ChangeSet.load(path)
        try:
            res = await cs.approve_and_apply(approver, push=push)
            status = res.pop("status")
        except ToolError as e:
            status, res = "failed", {"error": str(e)}
        done = folder.parent / "reviewed"
        done.mkdir(parents=True, exist_ok=True)
        path.rename(done / f"{path.stem}-{status}.json")
        Memory(settings.memory_path).record(f"review:{cs.kind}", status=status, title=cs.message)
        out.append(RoutineResult(cs.kind, status, cs.message, res))
    return out
