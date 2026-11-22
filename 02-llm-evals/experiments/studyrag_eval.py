"""StudyRAG (project 1) as evalkit's first consumer.

1. Import StudyRAG's v1 and v2 answer runs (public set) and test whether v2 really improved.
2. Grade every answer with the validated LLM judge:
   - correctness vs the reference answer, compared with project 1's key-fact recall (a word-matching heuristic);
   - faithfulness vs the passages the answer was generated from, compared with project 1's sentence verifier.
   Where the judge and a heuristic disagree, the cases are listed for reading.

    python experiments/studyrag_eval.py [--model qwen/qwen3.8-27b]
Needs data/studyrag_public.jsonl (experiments/studyrag_export.py) and the EVALKIT_* API settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalkit.compare import compare  # noqa: E402
from evalkit.judge import Judge  # noqa: E402
from evalkit.llm import ChatClient  # noqa: E402
from evalkit.metrics.agreement import Confusion  # noqa: E402
from evalkit.runs import Run, load_jsonl_items  # noqa: E402

RAG_RESULTS = ROOT.parent / "01-rag-from-scratch" / "eval" / "results"
VERSIONS = {"v1": "answers_Qwen3.5-2B-Q4_K_M_v1_details.jsonl", "v2": "answers_Qwen3.5-2B-Q4_K_M_details.jsonl"}
SUITE = "studyrag-answers"
TASK = "Answer the student's question \"{q}\" using only the numbered sources, citing them like [1]."


def load_questions() -> dict[str, dict]:
    return {q["id"]: q for q in load_jsonl_items(ROOT / "data" / "studyrag_public.jsonl")}


def judge_version(name: str, rows: list[dict], questions: dict, correct: Judge, faithful: Judge | None) -> Run:
    items = []
    for r in rows:
        answered = r["status"] == "answered"
        item = {"id": r["id"], "answerable": r["answerable"], "seconds": r["seconds"],
                ("answered_answerable" if r["answerable"] else "refused_unanswerable"):
                    answered if r["answerable"] else not answered}
        if r["answerable"] and r["status"] == "answered":
            q = questions[r["id"]]
            c = correct(question=q["question"], reference=q["answer"], response=r["answer"])
            f = (faithful(task=TASK.format(q=q["question"]), source="\n\n".join(q["passages"]), response=r["answer"])
                 if faithful else None)
            item.update({
                "fact_recall": r["fact_recall"], "citation_precision": r["citation_precision"],
                "verifier_supported_ratio": r["supported_ratio"],
                "judge_correctness": c.label, "judge_verdict": c.details.get("verdict"),
                "judge_missing": c.details.get("missing", []),
                "judge_faithful": None if f is None or f.label is None else float(not f.label),
                "judge_problems": f.details.get("problems", []) if f else [],
                "errors": [e for e in (c.error, f.error if f else None) if e]})
        items.append(item)
        print(f"  {name} {r['id']}: correctness={item.get('judge_correctness')} "
              f"faithful={item.get('judge_faithful')} fact_recall={item.get('fact_recall')}", flush=True)
    metrics = ["fact_recall", "judge_correctness", "judge_faithful", "verifier_supported_ratio",
               "citation_precision", "answered_answerable", "refused_unanswerable", "seconds"]
    return Run.build(SUITE, name, items, metrics,
                     config={"source": VERSIONS[name], "correctness_judge": correct.id,
                             "faithfulness_judge": faithful.id if faithful else None})


def heuristic_vs_judge(run: Run) -> None:
    scored = [it for it in run.items if it.get("judge_correctness") is not None]
    # "mostly correct" by each method: heuristic recall >= 0.5, judge not "incorrect"
    h = [it["fact_recall"] >= 0.5 for it in scored]
    j = [it["judge_correctness"] >= 0.5 for it in scored]
    c = Confusion.from_labels(h, j)
    print(f"\n[{run.name}] key-fact heuristic vs judge on 'mostly correct': agree {c.accuracy:.2f}, "
          f"kappa {c.kappa:.2f} (n={c.n})")
    for it in scored:
        if (it["fact_recall"] >= 0.5) != (it["judge_correctness"] >= 0.5) or it["fact_recall"] == 0:
            print(f"   {it['id']}: fact_recall={it['fact_recall']} judge={it['judge_verdict']} "
                  f"missing={it['judge_missing']}")
    f = [it for it in run.items if it.get("judge_faithful") is not None]
    v = [it["verifier_supported_ratio"] >= 0.999 for it in f]
    jf = [it["judge_faithful"] == 1.0 for it in f]
    c = Confusion.from_labels(v, jf)
    print(f"[{run.name}] project-1 verifier (all sentences supported) vs judge faithful: agree {c.accuracy:.2f}, "
          f"kappa {c.kappa:.2f} (n={c.n}); judge flags {sum(not x for x in jf)}, verifier flags {sum(not x for x in v)}")
    for it in f:
        if (it["verifier_supported_ratio"] >= 0.999) != (it["judge_faithful"] == 1.0):
            print(f"   {it['id']}: verifier={it['verifier_supported_ratio']} judge_problems="
                  f"{[p.get('quote') for p in it['judge_problems']]}")


def patient_client(model: str | None, extra: dict | None = None) -> ChatClient:
    return ChatClient.from_env(dotenv=ROOT / ".env", cache_dir=ROOT / ".cache" / "llm", extra_body=extra,
                               max_wait=900, max_retries=1000, **({"model": model} if model else {}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--correctness-model", default="openai/gpt-oss-20b",
                   help="separate model = separate free-tier quota (the faithfulness judge needs its own)")
    p.add_argument("--correctness-effort", default="low", help="reasoning_effort for a reasoning model ('' = none)")
    p.add_argument("--correctness-max-tokens", type=int, default=2000)
    p.add_argument("--faithfulness", default=str(ROOT / "results" / "judge-ragtruth-dev" / "selected.json"),
                   help="the judge configuration selected on the RAGTruth dev set (experiments/select_judge.py)")
    p.add_argument("--correctness-only", action="store_true", help="only grade correctness (fills the cache)")
    a = p.parse_args()
    effort = {"reasoning_effort": a.correctness_effort} if a.correctness_effort else None
    correct = Judge(patient_client(a.correctness_model, effort), "correctness", max_tokens=a.correctness_max_tokens)
    faithful = None
    if not a.correctness_only:
        sel = json.loads(Path(a.faithfulness).read_text())
        faithful = Judge(patient_client(sel["model"], sel.get("extra_body")), sel["rubric"],
                         max_tokens=sel.get("max_tokens"))
    questions = load_questions()
    runs = {}
    for name, file in VERSIONS.items():
        rows = load_jsonl_items(RAG_RESULTS / file)
        runs[name] = judge_version(name, rows, questions, correct, faithful)
        if a.correctness_only:
            continue
        print(f"saved {runs[name].save(ROOT / 'results')}")
        heuristic_vs_judge(runs[name])
    if not a.correctness_only:
        print("\n" + compare(runs["v1"], runs["v2"]).to_markdown())
    print(f"\ncorrectness judge usage: {json.dumps(vars(correct.client.usage), default=str)[:300]}")

if __name__ == "__main__":
    main()
