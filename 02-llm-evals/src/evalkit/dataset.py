"""Test cases: one JSON object per line.

A case holds everything any metric might need; each field is optional except `id` and `input`:

    {"id": "q1", "input": "When was X founded?", "expected": "In 1998.",
     "key_facts": ["1998"], "context": ["passage the answer must come from"],
     "gold": ["doc3#p2"], "answerable": true, "tags": {"task": "qa"}}

Common aliases are accepted so existing datasets load unchanged: question -> input, answer -> expected,
passages/contexts -> context.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

ALIASES = {"question": "input", "prompt": "input", "answer": "expected", "reference": "expected",
           "passages": "context", "contexts": "context"}


@dataclass
class Case:
    id: str
    input: str
    expected: str | None = None
    key_facts: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    gold: list[Any] = field(default_factory=list)       # relevant ids / locations, for retrieval metrics
    answerable: bool = True
    tags: dict[str, str] = field(default_factory=dict)  # for slicing results: task, difficulty, source...
    meta: dict[str, Any] = field(default_factory=dict)  # anything else (kept, never interpreted)

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        d = {ALIASES.get(k, k): v for k, v in d.items()}
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in known}
        d = {k: v for k, v in d.items() if k in known}
        if "id" not in d or "input" not in d:
            raise ValueError(f"case needs 'id' and 'input' (or 'question'): {sorted(d)}")
        if isinstance(d.get("context"), str):
            d["context"] = [d["context"]]
        case = cls(**d)
        case.id = str(case.id)
        case.meta = {**extra, **case.meta}
        return case

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {}) or k in ("id", "input")}


def load_cases(path: str | Path) -> list[Case]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    cases.append(Case.from_dict(json.loads(line)))
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    raise ValueError(f"{path}:{n}: {e}") from e
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"{path}: duplicate ids {dupes[:5]}")
    return cases


def save_cases(cases: Iterable[Case], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
