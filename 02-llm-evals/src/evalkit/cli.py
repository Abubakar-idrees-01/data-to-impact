"""Command line: evalkit <command> ...

    evalkit import  details.jsonl --suite studyrag-answers --name v2 [--metrics a,b]
    evalkit summary results/studyrag-answers/v2.json
    evalkit compare results/studyrag-answers/v1.json results/studyrag-answers/v2.json [--lower-is-better a,b]
    evalkit gate    results/studyrag-answers/v2.json --rules gate.toml [--baseline v1.json]
"""

from __future__ import annotations

import argparse
import sys

from .compare import compare, higher_is_better
from .gate import check, load_rules
from .runs import Run, load_jsonl_items


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evalkit", description="Evaluate LLM systems.")
    sub = p.add_subparsers(dest="cmd", required=True)

    imp = sub.add_parser("import", help="turn a per-item JSONL file into a run")
    imp.add_argument("jsonl")
    imp.add_argument("--suite", required=True)
    imp.add_argument("--name", required=True)
    imp.add_argument("--metrics", help="comma-separated; default: every numeric field")
    imp.add_argument("--out", default="results")

    sm = sub.add_parser("summary", help="metrics of a run with 95%% confidence intervals")
    sm.add_argument("run")

    cp = sub.add_parser("compare", help="paired comparison of two runs")
    cp.add_argument("base")
    cp.add_argument("new")
    cp.add_argument("--metrics", help="comma-separated; default: all shared metrics")
    cp.add_argument("--lower-is-better", default="", help="comma-separated metrics where smaller is better "
                    "(added to those guessed from the name, e.g. seconds, tokens, incorrect)")
    cp.add_argument("--higher-is-better", default="", help="comma-separated metrics to treat as higher-is-better "
                    "even if the name suggests otherwise")

    gt = sub.add_parser("gate", help="exit 1 if the run breaks the rules (for CI)")
    gt.add_argument("run")
    gt.add_argument("--rules", required=True)
    gt.add_argument("--baseline")

    a = p.parse_args(argv)
    if a.cmd == "import":
        metrics = a.metrics.split(",") if a.metrics else None
        run = Run.build(a.suite, a.name, load_jsonl_items(a.jsonl), metrics, config={"source": a.jsonl})
        print(f"saved {run.save(a.out)}\n\n{run.summary_table()}")
    elif a.cmd == "summary":
        run = Run.load(a.run)
        print(f"{run.suite}/{run.name} ({len(run.items)} items, {run.created})\n\n{run.summary_table()}")
    elif a.cmd == "compare":
        metrics = a.metrics.split(",") if a.metrics else None
        base, new = Run.load(a.base), Run.load(a.new)
        lower = {m for m in set(base.metrics) | set(new.metrics) if not higher_is_better(m)}
        lower = (lower | set(filter(None, a.lower_is_better.split(",")))) - set(filter(None, a.higher_is_better.split(",")))
        print(compare(base, new, metrics, lower_is_better=lower).to_markdown())
    elif a.cmd == "gate":
        results = check(Run.load(a.run), load_rules(a.rules), Run.load(a.baseline) if a.baseline else None)
        for r in results:
            print(f"{'PASS' if r.ok else 'FAIL'}  {r.metric}: {r.message}")
        return 0 if all(r.ok for r in results) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
