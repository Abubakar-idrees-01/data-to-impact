"""Task memory: one JSON line per routine run (what was done, when, with what outcome).

Used so the agent doesn't repeat itself (a digest won't cover a topic an earlier note covered) and so you
can see the history (`scout history`). Local only (`.scout/`, gitignored).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class Memory:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, kind: str, **fields) -> dict:
        entry = {"date": datetime.now().isoformat(timespec="seconds"), "kind": kind, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def recent(self, kind: str | None = None, n: int = 30) -> list[dict]:
        items = [e for e in self.all() if kind is None or e.get("kind") == kind]
        return items[-n:]
