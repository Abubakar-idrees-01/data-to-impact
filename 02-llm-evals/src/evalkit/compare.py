"""Compare two runs of the same suite item by item: did the new version really improve?

For each metric present in both runs, over the items both runs share:
- delta of the means, with a paired-bootstrap 95% CI and p-value;
- for pass/fail metrics, McNemar's exact test on the items that flipped;
- the ids of the items that got better and worse, so a regression can be inspected, not just counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics.stats import mcnemar_exact, paired_bootstrap
from .runs import Run

# Metrics where a smaller value is better, recognised by name unless the caller says otherwise.
# Ambiguous names (e.g. "attempted": good on a quiz, bad in a prompt-injection suite) need an explicit override.
LOWER_IS_BETTER_HINTS = ("second", "latency", "_ms", "time", "error", "cost", "loss", "tokens", "memory", "steps",
                         "incorrect", "hallucinat", "unsupported", "_broken", "wrong", "leak", "harm")
# Checked first: rates per unit of time are better when higher, whatever else the name contains.
HIGHER_IS_BETTER_HINTS = ("per_second", "per_sec", "_per_min", "throughput", "speedup", "tok_s")


def higher_is_better(metric: str) -> bool:
    name = metric.lower()
    if any(h in name for h in HIGHER_IS_BETTER_HINTS):
        return True
    return not any(h in name for h in LOWER_IS_BETTER_HINTS)


@dataclass
class MetricDiff:
    metric: str
    n: int                       # items present in both runs
    base: float
    new: float
    delta: float
    ci_low: float
    ci_high: float
    p: float                     # McNemar for pass/fail metrics, paired bootstrap otherwise
    test: str
    higher_is_better: bool = True
    improved: list[str] = field(default_factory=list)
    regressed: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return self.p < 0.05


@dataclass
class Comparison:
    base: str
    new: str
    diffs: list[MetricDiff]
    only_in_base: list[str]
    only_in_new: list[str]

    def get(self, metric: str) -> MetricDiff | None:
        return next((d for d in self.diffs if d.metric == metric), None)

    def to_markdown(self, show_items: int = 8) -> str:
        lines = [f"### {self.base} → {self.new}", "",
                 "| metric | n | base | new | Δ | 95% CI of Δ | p | test | better / worse items |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for d in self.diffs:
            star = " **\\***" if d.significant else ""
            name = d.metric if d.higher_is_better else f"{d.metric} ↓"
            lines.append(f"| {name} | {d.n} | {d.base:.3f} | {d.new:.3f} | {d.delta:+.3f}{star} | "
                         f"{d.ci_low:+.3f} … {d.ci_high:+.3f} | {d.p:.3f} | {d.test} | "
                         f"{len(d.improved)} / {len(d.regressed)} |")
        lines += ["", "\\* p < 0.05. ↓ lower is better. Paired over items present in both runs."]
        for d in self.diffs:
            if d.regressed and show_items:
                lines.append(f"- **{d.metric}** got worse on: {', '.join(d.regressed[:show_items])}"
                             + (" …" if len(d.regressed) > show_items else ""))
        if self.only_in_base or self.only_in_new:
            lines.append(f"- Not compared: {len(self.only_in_base)} items only in base, "
                         f"{len(self.only_in_new)} only in new.")
        return "\n".join(lines)


def compare(base: Run, new: Run, metrics: list[str] | None = None, eps: float = 1e-9,
            lower_is_better: set[str] | None = None) -> Comparison:
    """lower_is_better: metric names where smaller is better (default: guessed from the name, e.g. "seconds")."""
    names = metrics or [m for m in base.metrics if m in new.metrics]
    base_ids = {str(it["id"]) for it in base.items}
    new_ids = {str(it["id"]) for it in new.items}
    diffs = []
    for m in names:
        a, b = base.item_values(m), new.item_values(m)
        ids = sorted(set(a) & set(b))
        if not ids:
            continue
        xa, xb = [a[i] for i in ids], [b[i] for i in ids]
        boot = paired_bootstrap(xa, xb)
        hib = (m not in lower_is_better) if lower_is_better is not None else higher_is_better(m)
        up = [i for i in ids if b[i] > a[i] + eps]
        down = [i for i in ids if b[i] < a[i] - eps]
        improved, regressed = (up, down) if hib else (down, up)
        binary = all(v in (0.0, 1.0) for v in xa + xb)
        if binary:
            p, test = mcnemar_exact(len(regressed), len(improved)), "McNemar"
        else:
            p, test = boot["p"], "paired bootstrap"
        diffs.append(MetricDiff(metric=m, n=len(ids), base=sum(xa) / len(xa), new=sum(xb) / len(xb),
                                delta=boot["delta"], ci_low=boot["ci_low"], ci_high=boot["ci_high"],
                                p=p, test=test, higher_is_better=hib, improved=improved, regressed=regressed))
    return Comparison(base=f"{base.suite}/{base.name}", new=f"{new.suite}/{new.name}", diffs=diffs,
                      only_in_base=sorted(base_ids - new_ids), only_in_new=sorted(new_ids - base_ids))
