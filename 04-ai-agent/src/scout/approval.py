"""Human approval for outward actions.

The agent never decides alone to change anything outside itself. Before an outward tool runs, an `Approver`
is shown an `Action` (what will happen, plus the exact diff/preview) and answers yes or no.

- `ConsoleApprover`: asks you in the terminal. The default answer is **no**: pressing Enter declines.
- `DenyAll`: declines everything (evaluations, dry runs, and the default if nothing is configured).
- `Scripted`: answers from a list (tests only).
There is deliberately no "approve everything" mode for real use.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Protocol, TextIO


@dataclass
class Action:
    tool: str
    summary: str                 # one line: "commit 1 file to data-to-impact and push to origin/main"
    details: str = ""            # the exact change: a diff, the text of a note, the command


class Approver(Protocol):
    async def approve(self, action: Action) -> bool: ...


@dataclass
class DenyAll:
    seen: list[Action] = field(default_factory=list)

    async def approve(self, action: Action) -> bool:
        self.seen.append(action)
        return False


@dataclass
class Scripted:
    answers: list[bool]
    seen: list[Action] = field(default_factory=list)

    async def approve(self, action: Action) -> bool:
        self.seen.append(action)
        return self.answers.pop(0) if self.answers else False


class ConsoleApprover:
    def __init__(self, stdin: TextIO | None = None, stdout: TextIO | None = None, max_detail_lines: int = 80):
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self.max_detail_lines = max_detail_lines

    def _ask(self, action: Action) -> bool:
        out = self.stdout
        bar = "─" * 72
        out.write(f"\n{bar}\n⚠  APPROVAL NEEDED: {action.summary}\n{bar}\n")
        if action.details:
            lines = action.details.splitlines()
            out.write("\n".join(lines[: self.max_detail_lines]) + "\n")
            if len(lines) > self.max_detail_lines:
                out.write(f"... ({len(lines) - self.max_detail_lines} more lines)\n")
            out.write(bar + "\n")
        out.write("Approve? [y/N] ")
        out.flush()
        answer = self.stdin.readline()
        return answer.strip().lower() in {"y", "yes"}

    async def approve(self, action: Action) -> bool:
        if not self.stdin.isatty() and self.stdin is sys.stdin:
            self.stdout.write(f"\n(declined automatically, no terminal to ask in: {action.summary})\n")
            return False
        return await asyncio.to_thread(self._ask, action)
