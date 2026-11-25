"""Command line: `scout ask "question"` and friends. Run `scout --help`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import textwrap

from .agent import Agent
from .approval import ConsoleApprover
from .config import Settings
from .llm import LLM
from .mcp_tools import McpTools
from .memory import Memory
from . import routines
from .tools import Registry
from .web import Web


def progress(event: dict) -> None:
    """One short line per step on stderr, so you can watch the agent work."""
    t = event["type"]
    if t == "llm" and event["tool_calls"]:
        for c in event["tool_calls"]:
            args = c["arguments"]
            try:
                args = ", ".join(f"{k}={v!r}" for k, v in json.loads(args).items())
            except (ValueError, AttributeError):
                pass
            print(f"  → {c['name']}({textwrap.shorten(args, 90)})", file=sys.stderr)
    elif t == "tool" and event["is_error"]:
        print(f"    ✗ {textwrap.shorten(event['result'], 100)}", file=sys.stderr)
    elif t == "error":
        print(f"  ✗ {event['message']}", file=sys.stderr)


def research_registry(settings: Settings, approver) -> tuple[Registry, Web]:
    web = Web(settings.tavily_key, page_chars=settings.page_chars)
    return Registry(approver).add(*web.tools()), web


def print_result(result, web: Web | None) -> None:
    print("\n" + result.answer.strip() + "\n")
    if web:
        cited = web.book.cited(result.answer)
        if cited:
            print("Sources")
            for s in cited:
                print(f"  [{s.n}] {s.title or s.url}\n      {s.url}")
    print(f"\n({result.steps} model calls, {result.tool_calls} tool calls, {result.total_tokens:,} tokens, "
          f"{result.seconds:.0f} s, stopped: {result.stopped})", file=sys.stderr)


NOTES_CONTEXT = ("You can also search the user's own study notes (lecture slides, PDFs) with search_notes. For questions "
                 "about their courses, search the notes first and prefer them; use the web for what the notes don't "
                 "cover. Cite notes by their citation in parentheses, e.g. (Lec 2 — Slide 14), and web sources as [n].")


def make_agent(settings: Settings, registry: Registry, quiet: bool = False) -> Agent:
    return Agent(LLM.from_settings(settings), registry, max_steps=settings.max_steps,
                 token_budget=settings.token_budget, max_tokens_per_call=settings.max_tokens_per_call,
                 traces_dir=settings.traces_dir, on_event=None if quiet else progress,
                 context_limit=settings.context_limit)


async def cmd_ask(args, settings: Settings) -> int:
    registry, web = research_registry(settings, ConsoleApprover())
    question = " ".join(args.question)
    if args.notes:
        if not settings.mcp_command:
            raise RuntimeError("no StudyRAG MCP server found: install studyrag-mcp or set SCOUT_MCP_COMMAND")
        async with McpTools(settings.mcp_command, include={"search_notes", "get_passage", "list_documents"}) as mcp:
            registry.add(*mcp.tools())
            result = await make_agent(settings, registry, args.quiet).run(question, context=NOTES_CONTEXT)
    else:
        result = await make_agent(settings, registry, args.quiet).run(question)
    print_result(result, web)
    return 0 if result.stopped != "error" else 1


def show_routine(r) -> int:
    run = r.run
    if run:
        print(f"\n({run.steps} model calls, {run.tool_calls} tool calls, {run.total_tokens:,} tokens, {run.seconds:.0f} s)",
              file=sys.stderr)
    d = r.detail or {}
    if r.kind == "checkup" and d.get("report"):
        print("\n" + d["report"].strip())
    msg = {"applied": f"✓ committed {d.get('commit', '')}" + (f" and pushed to {d['pushed']}" if d.get("pushed") else ""),
           "declined": "declined: nothing was changed",
           "prepared": f"prepared for review (run `scout review`): {d.get('pending')}",
           "nothing": "nothing to change",
           "failed": "the agent didn't produce a result" + (f": {d.get('answer', '')[:300]}" if d.get("answer") else "")}
    print(f"\n{r.kind}: {r.title or ''}\n{msg.get(r.status, r.status)}")
    if d.get("push_error"):
        print("⚠ " + d["push_error"])
    return 0 if r.status != "failed" else 1


async def cmd_digest(args, settings: Settings) -> int:
    r = await routines.digest(settings, ConsoleApprover(), topic=args.topic, prepare=args.prepare, push=not args.no_push,
                              on_event=None if args.quiet else progress)
    return show_routine(r)


async def cmd_checkup(args, settings: Settings) -> int:
    r = await routines.checkup(settings, ConsoleApprover(), focus=args.focus or "", prepare=args.prepare,
                               push=not args.no_push, on_event=None if args.quiet else progress)
    return show_routine(r)


async def cmd_review(args, settings: Settings) -> int:
    results = await routines.review(settings, ConsoleApprover(), push=not args.no_push)
    if not results:
        print("nothing waiting for review")
    for r in results:
        show_routine(r)
    return 0


async def cmd_history(args, settings: Settings) -> int:
    for e in Memory(settings.memory_path).recent(None, args.n):
        print(f"{e['date'][:16]}  {e['kind']:<16} {e.get('status', ''):<9} {e.get('title') or e.get('reason', '')[:70]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scout", description="A research and developer agent with human approval.")
    ap.add_argument("--model", help="override SCOUT_API_MODEL")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask", help="research a question on the web and answer with sources")
    p.add_argument("question", nargs="+")
    p.add_argument("-q", "--quiet", action="store_true", help="don't show the steps")
    p.add_argument("--notes", action="store_true", help="also search your StudyRAG notes (project 3's MCP server)")
    for name, help_ in (("digest", "research one recent development and draft a learning-log note"),
                        ("checkup", "check the repo's docs and propose small verified fixes")):
        p = sub.add_parser(name, help=help_)
        if name == "digest":
            p.add_argument("--topic", help="research this topic instead of the configured ones")
        else:
            p.add_argument("--focus", help="what to look at, e.g. '03-mcp-server docs'")
        p.add_argument("--prepare", action="store_true", help="don't ask now: save the proposal for `scout review`")
        p.add_argument("--no-push", action="store_true", help="commit after approval, but don't push")
        p.add_argument("-q", "--quiet", action="store_true")
    p = sub.add_parser("review", help="approve or decline proposals prepared earlier")
    p.add_argument("--no-push", action="store_true")
    p = sub.add_parser("history", help="what the agent did recently")
    p.add_argument("-n", type=int, default=20)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="  %(message)s")
    settings = Settings.from_env()
    if args.model:
        settings.api_model = args.model
    handler = {"ask": cmd_ask, "digest": cmd_digest, "checkup": cmd_checkup, "review": cmd_review,
               "history": cmd_history}[args.cmd]
    try:
        return asyncio.run(handler(args, settings))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as e:                       # config problems etc.: one clear line, not a traceback
        if args.verbose:
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1
