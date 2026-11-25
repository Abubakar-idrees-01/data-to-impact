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


async def cmd_ask(args, settings: Settings) -> int:
    registry, web = research_registry(settings, ConsoleApprover())
    agent = Agent(LLM.from_settings(settings), registry, max_steps=settings.max_steps,
                  token_budget=settings.token_budget, max_tokens_per_call=settings.max_tokens_per_call,
                  traces_dir=settings.traces_dir, on_event=None if args.quiet else progress)
    result = await agent.run(" ".join(args.question))
    print_result(result, web)
    return 0 if result.stopped != "error" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scout", description="A research and developer agent with human approval.")
    ap.add_argument("--model", help="override SCOUT_API_MODEL")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask", help="research a question on the web and answer with sources")
    p.add_argument("question", nargs="+")
    p.add_argument("-q", "--quiet", action="store_true", help="don't show the steps")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="  %(message)s")
    settings = Settings.from_env()
    if args.model:
        settings.api_model = args.model
    handler = {"ask": cmd_ask}[args.cmd]
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
