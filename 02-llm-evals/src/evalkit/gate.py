"""Regression gate: fail (exit code 1) when a run breaks its quality rules. Meant for CI.

Rules live in a TOML file, one table per metric:

    [metrics.key_fact_recall]
    min = 0.85                 # absolute floor
    max_drop = 0.03            # vs the baseline run, in metric units
    require_significance = true  # only fail a drop that a paired test says is real (p < 0.05)

    [metrics.median_seconds]
    higher_is_better = false   # optional: names with "seconds", "latency", "cost"... are detected
    max = 40
    max_drop = 5               # "drop" = getting worse, i.e. +5 s here

Absolute limits catch "too bad"; baseline limits catch "worse than before". With small eval sets a
1-question wobble can look like a drop, which is what `require_significance` is for.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .compare import compare, higher_is_better
from .runs import Run


@dataclass
class Check:
    metric: str
    ok: bool
    message: str


def load_rules(path: str | Path) -> dict[str, dict]:
    with open(path, "rb") as f:
        return tomllib.load(f).get("metrics", {})


def check(run: Run, rules: dict[str, dict], baseline: Run | None = None) -> list[Check]:
    results = []
    lower = {m for m, r in rules.items() if not r.get("higher_is_better", higher_is_better(m))}
    cmp = compare(baseline, run, list(rules), lower_is_better=lower) if baseline else None
    for metric, rule in rules.items():
        value = run.value(metric)
        if value is None:
            results.append(Check(metric, False, "missing from run"))
            continue
        higher = rule.get("higher_is_better", higher_is_better(metric))
        problems = []
        if "min" in rule and value < rule["min"]:
            problems.append(f"{value:.3f} < min {rule['min']}")
        if "max" in rule and value > rule["max"]:
            problems.append(f"{value:.3f} > max {rule['max']}")
        if baseline is not None and "max_drop" in rule:
            base_value = baseline.value(metric)
            diff = cmp.get(metric) if cmp else None
            if base_value is not None:
                worse_by = (base_value - value) if higher else (value - base_value)
                significant = diff.significant if diff else True
                if worse_by > rule["max_drop"] and (significant or not rule.get("require_significance", False)):
                    p = f", p={diff.p:.3f}" if diff else ""
                    problems.append(f"worse than baseline by {worse_by:.3f} (> {rule['max_drop']}{p})")
        results.append(Check(metric, not problems, "; ".join(problems) or f"{value:.3f} ok"))
    return results
