"""An evaluation run = per-item results + aggregate metrics + how it was produced, saved as one JSON file.

Keeping per-item results (not just averages) is what makes later comparisons possible: two versions can be
compared question by question, and every regression can be traced to the exact items that broke.

    results/<suite>/<name>.json
    {"suite": "studyrag-answers", "name": "v2", "created": "...", "config": {...}, "env": {...},
     "metrics": {"key_fact_recall": {"value": 0.90, "ci_low": 0.84, "ci_high": 0.95, "n": 34}},
     "items": [{"id": "s01", "key_fact_recall": 1.0, ...}, ...]}
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .metrics.stats import bootstrap_ci, wilson_ci


def numeric(value: Any) -> float | None:
    """bool/int/float -> float; anything else (text, None, lists) -> None."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def aggregate(items: Iterable[dict], metrics: Iterable[str]) -> dict[str, dict[str, float]]:
    """Mean of each metric over the items that have it, with a 95% confidence interval.

    Pass/fail metrics (all values 0 or 1) use the Wilson interval, others the bootstrap.
    """
    items = list(items)
    out = {}
    for m in metrics:
        vals = [v for v in (numeric(it.get(m)) for it in items) if v is not None]
        if not vals:
            continue
        if all(v in (0.0, 1.0) for v in vals):
            lo, hi = wilson_ci(int(sum(vals)), len(vals))
        else:
            lo, hi = bootstrap_ci(vals)
        out[m] = {"value": mean(vals), "ci_low": lo, "ci_high": hi, "n": len(vals)}
    return out


def environment(cwd: str | Path | None = None) -> dict[str, Any]:
    """Where and from which code a run was produced (no personal data: no user or host names)."""
    env = {"python": platform.python_version(), "platform": platform.platform(terse=True),
           "machine": platform.machine(), "cpu_count": os.cpu_count()}
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip()
        commit = git("rev-parse", "--short", "HEAD")
        if commit:
            env["git_commit"] = commit
            env["git_dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        pass
    return env


@dataclass
class Run:
    suite: str
    name: str
    items: list[dict]
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @classmethod
    def build(cls, suite: str, name: str, items: list[dict], metrics: Iterable[str] | None = None,
              config: dict | None = None, extra_metrics: dict[str, float] | None = None) -> "Run":
        """Create a run; metrics default to every numeric field found in the items."""
        for it in items:
            if "id" not in it:
                raise ValueError("every item needs an 'id' so runs can be compared item by item")
        if metrics is None:
            seen: dict[str, None] = {}
            for it in items:
                for k, v in it.items():
                    if k != "id" and numeric(v) is not None:
                        seen.setdefault(k)
            metrics = list(seen)
        agg = aggregate(items, metrics)
        for k, v in (extra_metrics or {}).items():     # run-level numbers that aren't item means (e.g. kappa)
            agg[k] = {"value": float(v), "n": len(items)}
        return cls(suite=suite, name=name, items=items, metrics=agg, config=config or {}, env=environment())

    def value(self, metric: str) -> float | None:
        m = self.metrics.get(metric)
        return m["value"] if m else None

    def item_values(self, metric: str) -> dict[str, float]:
        return {str(it["id"]): v for it in self.items if (v := numeric(it.get(metric))) is not None}

    def save(self, directory: str | Path = "results") -> Path:
        path = Path(directory) / self.suite / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Run":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**d)

    def summary_table(self) -> str:
        rows = ["| metric | value | 95% CI | n |", "|---|---|---|---|"]
        for k, m in self.metrics.items():
            ci = f"{m['ci_low']:.3f} – {m['ci_high']:.3f}" if "ci_low" in m else "–"
            rows.append(f"| {k} | {m['value']:.3f} | {ci} | {m['n']} |")
        return "\n".join(rows)


def load_jsonl_items(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

