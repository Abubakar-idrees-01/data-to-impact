"""Build the public report (site/index.html) from the saved runs in results/.

    python experiments/build_report.py

Sections appear as their results exist; missing ones say "pending". The page is self-contained (no scripts,
no external files), so it can be opened locally or served by GitHub Pages.
Example texts are short excerpts of RAGTruth (MIT) and of StudyRAG's public Wikipedia set (CC BY-SA).
"""

from __future__ import annotations

import html
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalkit.compare import compare  # noqa: E402
from evalkit.dataset import load_cases  # noqa: E402
from evalkit.metrics.agreement import Confusion  # noqa: E402
from evalkit.report import Bar, bar_chart, confusion_matrix, esc, fmt, page, table, tiles  # noqa: E402
from evalkit.runs import Run  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "site" / "index.html"
PRETTY = {"faithfulness-claims": "quote every problem", "faithfulness-direct": "verdict only",
          "faithfulness-severity": "major/minor problems"}


def label(run: Run) -> str:
    if run.name.startswith("baseline-"):
        return {"baseline-lexical": "Word overlap (no LLM)", "baseline-numbers": "New numbers (no LLM)",
                "baseline-always-hallucinated": "Always 'hallucinated'"}.get(run.name, run.name)
    rubric, model = run.name.split("__")[:2]
    model = model.split("_", 1)[-1]
    return f"{model} · {PRETTY.get(rubric, rubric)}"


def load_runs(split: str) -> list[Run]:
    d = RES / f"judge-ragtruth-{split}"
    return [Run.load(p) for p in sorted(d.glob("*.json")) if p.name != "selected.json"] if d.exists() else []


def val(run: Run, k: str) -> float | None:
    m = run.metrics.get(k)
    return m["value"] if m else None


def coverage(run: Run) -> float:
    return 1 - (val(run, "n_errors") or 0) / max(1, len(run.items))


def kappa_chart(runs: list[Run], title: str, selected: str | None = None) -> str:
    rows = sorted(runs, key=lambda r: -(val(r, "kappa") or 0))
    bars = [Bar(label(r) + (" ✓" if r.name == selected else ""), val(r, "kappa") or 0,
                val(r, "kappa_ci_low"), val(r, "kappa_ci_high"),
                "base" if r.name.startswith("baseline") else ("main" if r.name == selected else "alt"))
            for r in rows if coverage(r) >= 0.95]
    return bar_chart(bars, title, xmin=-0.2, xmax=1.0, ticks=6,
                     unit_note="Cohen's kappa with human annotators (0 = chance, 1 = perfect); whiskers: 95% "
                               "bootstrap CI. ✓ = configuration chosen by the selection rule.")


def config_table(runs: list[Run], selected: str | None) -> str:
    rows = []
    for r in sorted(runs, key=lambda r: -(val(r, "kappa") or 0)):
        cov = coverage(r)
        note = "chosen" if r.name == selected else ("excluded: answered only {:.0%}".format(cov) if cov < 0.95 else "")
        rows.append([esc(label(r)), fmt(val(r, "kappa")),
                     f'{fmt(val(r, "kappa_ci_low"))} – {fmt(val(r, "kappa_ci_high"))}',
                     fmt(val(r, "recall")), fmt(val(r, "precision")), fmt(val(r, "f1")), f"{cov:.0%}", note])
    return table(["detector", "kappa", "95% CI", "recall", "precision", "F1", "answered", ""], rows,
                 numeric=(1, 3, 4, 5, 6))


def slice_table(run: Run, key: str) -> str:
    groups = defaultdict(list)
    for it in run.items:
        if it.get("pred") is not None:
            groups[it[key]].append(it)
    rows = []
    for g, its in sorted(groups.items()):
        c = Confusion.from_labels([i["pred"] for i in its], [i["gold"] for i in its])
        rows.append([esc(g), c.n, fmt(c.kappa), fmt(c.recall), fmt(c.precision), fmt(c.accuracy)])
    return table([key, "n", "kappa", "recall", "precision", "accuracy"], rows, numeric=(1, 2, 3, 4, 5))


def examples(run: Run, cases: dict, kind: str, k: int = 3) -> str:
    want = (lambda it: it["gold"] and it["pred"] is False) if kind == "missed" else \
           (lambda it: not it["gold"] and it["pred"] is True)
    out = []
    for it in [i for i in run.items if i.get("pred") is not None and want(i)][:k]:
        c = cases.get(it["id"])
        if not c:
            continue
        resp = c.meta["response"]
        body = f"<p class='muted'>{esc(c.tags['task'])} · written by {esc(c.tags['generator'])}</p>"
        body += f"<blockquote>{esc(resp[:420])}{'…' if len(resp) > 420 else ''}</blockquote>"
        if kind == "missed":
            spans = "; ".join(f"“{esc(s[:160])}”" for s in c.meta["spans"][:2])
            body += f"<p><b>Humans marked:</b> {spans} ({esc(', '.join(c.meta['label_types']))})</p>"
        else:
            probs = it.get("problems") or []
            reason = "; ".join(f"“{esc(str(p.get('quote', ''))[:120])}”: {esc(str(p.get('why', ''))[:160])}"
                               for p in probs[:2]) or "(verdict-only rubric: no quote)"
            body += f"<p><b>Judge objected to:</b> {reason}</p>"
        out.append(f'<div class="card">{body}</div>')
    return "".join(out) or "<p class='muted'>None.</p>"


def judge_section() -> str:
    dev, test = load_runs("dev"), load_runs("test")
    sel_path = RES / "judge-ragtruth-dev" / "selected.json"
    selected = json.loads(sel_path.read_text())["name"] if sel_path.exists() else None
    parts = []
    test_judge = next((r for r in test if r.name == selected), None)
    if test_judge:
        c = Confusion(*(int(val(test_judge, k)) for k in ("tp", "fp", "fn", "tn")))
        lex = next((r for r in test if r.name == "baseline-lexical"), None)
        parts.append(tiles([
            (fmt(val(test_judge, "kappa")), "kappa with human annotators",
             f"95% CI {fmt(val(test_judge, 'kappa_ci_low'))}–{fmt(val(test_judge, 'kappa_ci_high'))} · {c.n} held-out answers"),
            (f"{c.recall:.0%}", "of human-marked hallucinations caught", f"{c.tp} of {c.tp + c.fn}"),
            (f"{c.precision:.0%}", "of judge alarms were real", f"{c.fp} false alarms on {c.fp + c.tn} clean answers"),
            (fmt(val(lex, "kappa")) if lex else "–", "kappa of the best non-LLM baseline", "word overlap"),
        ]))
        parts.append(kappa_chart(test, "Held-out test set: agreement with human annotators", selected))
        parts.append(confusion_matrix(c, f"Chosen judge on the held-out test set ({label(test_judge)})"))
        parts.append("<h3>Where does it agree less?</h3>" + '<div class="card">' + slice_table(test_judge, "task")
                     + "</div>" + '<div class="card">' + slice_table(test_judge, "generator") + "</div>")
        cases = {c.id: c for c in load_cases(ROOT / "data" / "ragtruth_test.jsonl")} \
            if (ROOT / "data" / "ragtruth_test.jsonl").exists() else {}
        if cases:
            parts.append("<h3>Hallucinations the judge missed</h3>" + examples(test_judge, cases, "missed"))
            parts.append("<h3>False alarms</h3>" + examples(test_judge, cases, "false"))
    else:
        parts.append('<div class="card"><b>Held-out test results: pending.</b> The chosen judge runs once on 240 '
                     "held-out answers; the free API tier allows ~200k tokens per model per day, so this takes "
                     "about two days. The tuning results below are final.</div>")
    if dev:
        parts.append("<h2>Tuning on the dev set (60 answers)</h2><p>Every judge configuration and baseline was "
                     "tried on a separate dev sample. The configuration for the test set was chosen by a rule "
                     "fixed in advance: highest kappa among configurations that answered ≥ 95% of cases.</p>")
        parts.append(kappa_chart(dev, "Dev set: agreement with human annotators", selected))
        parts.append('<div class="card">' + config_table(dev, selected) + "</div>")
    return "".join(parts)


def studyrag_section() -> str:
    d = RES / "studyrag-answers"
    if not (d / "v1.json").exists() or not (d / "v2.json").exists():
        return '<div class="card"><b>Pending.</b> StudyRAG answers are graded once the judge is chosen.</div>'
    v1, v2 = Run.load(d / "v1.json"), Run.load(d / "v2.json")
    cmp = compare(v1, v2)
    rows = []
    for x in cmp.diffs:
        verdict = ('<span class="good">real improvement</span>' if x.significant and x.delta * (1 if x.higher_is_better else -1) > 0
                   else '<span class="bad">real regression</span>' if x.significant else "within noise")
        rows.append([esc(x.metric), x.n, fmt(x.base, 3), fmt(x.new, 3), f"{x.delta:+.3f}",
                     f"{x.ci_low:+.3f} … {x.ci_high:+.3f}", fmt(x.p, 3), verdict])
    out = ['<div class="card">' + table(["metric", "n", "v1", "v2", "Δ", "95% CI of Δ", "p", "verdict"], rows,
                                        numeric=(1, 2, 3, 4, 6)) + "</div>"]
    scored = [it for it in v2.items if it.get("judge_correctness") is not None]
    if scored:
        c = Confusion.from_labels([it["fact_recall"] >= 0.5 for it in scored],
                                  [it["judge_correctness"] >= 0.5 for it in scored])
        out.append(f"<p>Project 1's word-matching <i>key-fact recall</i> and the LLM correctness judge agree on "
                   f"“mostly correct” for {c.accuracy:.0%} of answered questions (n = {c.n}).</p>")
    return "".join(out)


def main() -> None:
    body = f"""
<div class="kicker">evalkit · project 2 of Data to Impact</div>
<h1>Can an LLM judge spot hallucinations as well as a human?</h1>
<p class="lede">An LLM "judge" is only a metric once its agreement with people is measured. We measured it on
<a href="https://github.com/ParticleMedia/RAGTruth">RAGTruth</a>, where humans marked every hallucinated span in
answers written by six LLMs, then used the validated judge to grade a real RAG system.</p>
<h2>1. The judge vs human annotators</h2>
{judge_section()}
<h2>2. Using it: is StudyRAG v2 really better than v1?</h2>
<p>StudyRAG (project 1) answers questions from course material. Same 42 questions, two versions, compared item by
item with paired tests (McNemar for pass/fail metrics, paired bootstrap otherwise).</p>
{studyrag_section()}
<h2>How this was measured</h2>
<p>Judge: an LLM over an OpenAI-compatible API (Groq free tier), temperature 0, JSON output, every response cached.
Data: a balanced, seeded sample of RAGTruth (dev split for tuning, test split for the final number; ids in
<code>datasets/ragtruth_ids.json</code>). Agreement: Cohen's kappa, precision/recall/F1 for the "hallucinated"
class, 95% bootstrap intervals. Code: <a href="https://github.com/Abubakar-idrees-01/data-to-impact/tree/main/02-llm-evals">
02-llm-evals</a> (Python standard library only).</p>
<p class="muted">Generated {date.today():%d %B %Y} by experiments/build_report.py.</p>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page("evalkit: LLM judge validation", body,
                        "Measuring whether an LLM judge agrees with human hallucination labels, and using it "
                        "to evaluate a RAG system."), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
