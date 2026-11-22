"""Pick the judge configuration for the held-out test set, by a rule fixed before seeing test data.

Rule (written down before the dev runs finished):
1. Only LLM-judge runs on the dev split, with verdicts on at least 95% of cases (a judge that often fails to
   answer is not usable, whatever its kappa on the rest).
2. Highest Cohen's kappa with the human labels.
3. Tie (kappa within 0.02): the higher lower bound of the 95% CI, then the cheaper configuration (fewer tokens).

    python experiments/select_judge.py      # prints the table, writes results/judge-ragtruth-dev/selected.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalkit.runs import Run  # noqa: E402

DEV = ROOT / "results" / "judge-ragtruth-dev"


def candidates() -> list[dict]:
    rows = []
    for path in sorted(DEV.glob("*.json")):
        if path.name == "selected.json" or path.stem.startswith("baseline"):
            continue
        run = Run.load(path)
        m = {k: v["value"] for k, v in run.metrics.items()}
        n = len(run.items)
        usage = run.config.get("usage", {})
        rows.append({"name": run.name, "rubric": run.config["rubric"].split("@")[0], "model": run.config["model"],
                     "extra_body": run.config.get("extra_body"), "max_tokens": run.config.get("max_tokens"),
                     "kappa": m.get("kappa", 0.0), "kappa_ci_low": m.get("kappa_ci_low", 0.0),
                     "f1": m.get("f1", 0.0), "coverage": 1 - m.get("n_errors", 0) / n, "n": n,
                     "tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)})
    return rows


def select(rows: list[dict]) -> dict | None:
    usable = [r for r in rows if r["coverage"] >= 0.95]
    if not usable:
        return None
    best = max(r["kappa"] for r in usable)
    tied = [r for r in usable if best - r["kappa"] <= 0.02]
    return sorted(tied, key=lambda r: (-r["kappa_ci_low"], r["tokens"] or float("inf")))[0]


def main() -> None:
    rows = candidates()
    print(f"{'configuration':<62} {'kappa':>6} {'CI low':>7} {'F1':>5} {'coverage':>9}")
    for r in sorted(rows, key=lambda r: -r["kappa"]):
        print(f"{r['name']:<62} {r['kappa']:6.3f} {r['kappa_ci_low']:7.3f} {r['f1']:5.2f} {r['coverage']:9.0%}")
    chosen = select(rows)
    if not chosen:
        sys.exit("no usable configuration yet")
    (DEV / "selected.json").write_text(json.dumps(chosen | {"rule": __doc__.split("Rule")[1].split("\n\n")[0]},
                                                  indent=1))
    print(f"\nselected: {chosen['name']}")


if __name__ == "__main__":
    main()
