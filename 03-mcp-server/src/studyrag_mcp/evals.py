"""Read-only access to evalkit runs (project 2): list them, summarize one, compare two.

Paths are always resolved inside one results directory: a tool argument like "../../.ssh/id_rsa" is
rejected, so an AI client can never use these tools to read arbitrary files.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_RESULTS = Path(__file__).resolve().parents[3] / "02-llm-evals" / "results"


class EvalsError(ValueError):
    pass


class Evals:
    def __init__(self, results_dir: str | Path | None = None):
        self.root = Path(results_dir or os.environ.get("STUDYRAG_MCP_EVAL_RESULTS") or DEFAULT_RESULTS).resolve()

    def _path(self, run: str) -> Path:
        name = run.strip().removesuffix(".json")
        if not name:
            raise EvalsError("run is empty; use list_eval_runs to see the available runs")
        p = (self.root / f"{name}.json").resolve()
        if self.root not in p.parents:
            raise EvalsError("run must be a name like 'suite/run' inside the results directory")
        if not p.exists():
            raise EvalsError(f"no run {name!r}; use list_eval_runs to see the available runs")
        return p

    def list_runs(self) -> dict:
        if not self.root.exists():
            return {"results_dir": None, "runs": [],
                    "note": "no evalkit results directory found; set STUDYRAG_MCP_EVAL_RESULTS to one"}
        runs = sorted(str(p.relative_to(self.root).with_suffix("")) for p in self.root.glob("*/*.json")
                      if p.name != "selected.json")
        return {"results_dir": str(self.root), "count": len(runs), "runs": runs}

    @staticmethod
    def _evalkit():
        try:
            import evalkit.compare
            import evalkit.runs
            return evalkit
        except ImportError:
            raise EvalsError("evaluation tools need evalkit (project 2): pip install -e path/to/02-llm-evals") from None

    def summary(self, run: str) -> dict:
        Run = self._evalkit().runs.Run
        r = Run.load(self._path(run))
        return {"run": f"{r.suite}/{r.name}", "items": len(r.items), "created": r.created,
                "config": r.config, "metrics": r.metrics, "table": r.summary_table()}

    def compare(self, base: str, new: str, metrics: list[str] | None = None) -> dict:
        ek = self._evalkit()
        compare, Run = ek.compare.compare, ek.runs.Run
        c = compare(Run.load(self._path(base)), Run.load(self._path(new)), metrics or None)
        if not c.diffs:
            raise EvalsError("the two runs share no metrics/items to compare")
        return {"base": c.base, "new": c.new, "markdown": c.to_markdown(),
                "metrics": [{"metric": d.metric, "n": d.n, "base": d.base, "new": d.new, "delta": d.delta,
                             "ci": [d.ci_low, d.ci_high], "p": d.p, "test": d.test, "significant": d.significant,
                             "improved": d.improved[:20], "regressed": d.regressed[:20]} for d in c.diffs]}
