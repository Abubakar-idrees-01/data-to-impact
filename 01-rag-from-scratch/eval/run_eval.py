"""Evaluation harness: measure how well StudyRAG actually works.

    python eval/run_eval.py retrieval --set sample      # chunking x search mode x reranker grid
    python eval/run_eval.py refusal   --set sample      # calibrate the "not in your documents" threshold
    python eval/run_eval.py verifier  --set sample      # precision/recall of the claim checker
    python eval/run_eval.py answers   --set sample      # end-to-end answers (needs the LLM; slow on CPU)
    python eval/run_eval.py mcq       --set sample -n 15
    python eval/run_eval.py speed                       # LLM speed + thread tuning

--set sample  -> eval/datasets/sample_qa.jsonl over data/sample_docs (public)
--set private -> eval/datasets/private/my_docs_qa.jsonl over data/my_docs (never pushed; only
                 aggregate numbers are published)

Dataset line format (JSONL):
{"id": "s01", "question": "...", "answer": "reference answer", "key_facts": ["...", "..."],
 "gold": [{"doc": "Email_marketing", "loc": [3]}], "answerable": true, "type": "definition"}
`gold` lists the document title + slide/page/section numbers that contain the answer.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from studyrag.config import Settings  # noqa: E402
from studyrag.embeddings import Embedder, Reranker  # noqa: E402
from studyrag.library import Library  # noqa: E402
from studyrag.retrieval import Retriever  # noqa: E402
from studyrag.study.generate import support_score  # noqa: E402

SETS = {
    "sample": (ROOT / "eval/datasets/sample_qa.jsonl", ROOT / "data/sample_docs", ROOT / "eval/results"),
    "private": (ROOT / "eval/datasets/private/my_docs_qa.jsonl", ROOT / "data/my_docs",
                ROOT / "eval/results/private"),
}
CACHE = ROOT / "eval/.cache"


# ------------------------------------------------------------------ shared helpers
def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_relevant(chunk, gold: list[dict]) -> bool:
    for g in gold:
        if chunk.doc_title.lower() == g["doc"].lower() and any(
                chunk.loc_start <= loc <= chunk.loc_end for loc in g["loc"]):
            return True
    return False


def first_relevant_rank(hits, gold) -> int | None:
    for r, h in enumerate(hits, 1):
        if is_relevant(h.chunk, gold):
            return r
    return None


def build_library(settings: Settings, embedder, docs_dir: Path, tag: str) -> Library:
    s = replace(settings, storage_dir=CACHE / tag)
    lib = Library(s, embedder)
    lib.sync_folder(docs_dir)
    return lib


def save(out_dir: Path, name: str, summary: dict, rows: list[dict] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
    if rows is not None:
        with open(out_dir / f"{name}_details.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved -> {out_dir / name}.json")


def md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def models(settings: Settings, rerank: bool = True):
    cache = settings.models_dir / "fastembed"
    emb = Embedder(settings.embedding_model, cache)
    rr = Reranker(settings.reranker_model, cache) if rerank else None
    return emb, rr


# ------------------------------------------------------------------ 1. retrieval grid
def eval_retrieval(a, settings, data, docs_dir, out_dir):
    emb, rr = models(settings)
    qs = [q for q in data if q["answerable"]]
    rows = []
    for strategy in ("fixed", "structure", "semantic"):
        s = replace(settings, chunk_strategy=strategy)
        lib = build_library(s, emb, docs_dir, f"{a.set}-{strategy}")
        avg_words = statistics.mean(c.word_count for c in lib.chunks)
        for mode in ("bm25", "vector", "hybrid"):
            for use_rr in ((False, True) if mode == "hybrid" else (False,)):
                ret = Retriever(lib, emb, rr, replace(s, use_reranker=use_rr))
                ranks, lat = [], []
                for q in qs:
                    t = time.perf_counter()
                    res = ret.retrieve(q["question"], top_k=10, mode=mode, rerank=use_rr)
                    lat.append((time.perf_counter() - t) * 1000)
                    ranks.append(first_relevant_rank(res.hits, q["gold"]))
                hit = lambda k: sum(1 for r in ranks if r and r <= k) / len(ranks)  # noqa: E731
                rows.append({"chunking": strategy, "search": mode + (" + rerank" if use_rr else ""),
                             "chunks": len(lib.chunks), "avg words": round(avg_words),
                             "hit@1": f"{hit(1):.2f}", "hit@3": f"{hit(3):.2f}", "hit@5": f"{hit(5):.2f}",
                             "MRR@10": f"{statistics.mean(1 / r if r else 0 for r in ranks):.3f}",
                             "ms/query": f"{statistics.median(lat):.0f}"})
                print(rows[-1])
    cols = ["chunking", "search", "chunks", "avg words", "hit@1", "hit@3", "hit@5", "MRR@10", "ms/query"]
    table = md_table(rows, cols)
    print("\n" + table)
    save(out_dir, "retrieval", {"n_questions": len(qs), "rows": rows, "table": table})


# ------------------------------------------------------------------ 2. refusal threshold
def eval_refusal(a, settings, data, docs_dir, out_dir):
    emb, rr = models(settings)
    lib = build_library(settings, emb, docs_dir, f"{a.set}-{settings.chunk_strategy}")
    ret = Retriever(lib, emb, rr, settings)
    pos, neg = [], []
    for q in data:
        top = ret.retrieve(q["question"], top_k=3).top_relevance
        (pos if q["answerable"] else neg).append(top)
    rows = []
    for th in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        kept = sum(p >= th for p in pos) / len(pos)
        refused = sum(n < th for n in neg) / len(neg) if neg else 0
        rows.append({"threshold": th, "answerable kept": f"{kept:.2f}", "unanswerable refused": f"{refused:.2f}",
                     "balanced acc": f"{(kept + refused) / 2:.2f}"})
    table = md_table(rows, ["threshold", "answerable kept", "unanswerable refused", "balanced acc"])
    print(table)
    print(f"\nanswerable top-rerank: median {statistics.median(pos):.3f}, min {min(pos):.3f}")
    if neg:
        print(f"unanswerable top-rerank: median {statistics.median(neg):.3f}, max {max(neg):.3f}")
    save(out_dir, "refusal", {"rows": rows, "table": table, "answerable_scores": pos, "unanswerable_scores": neg})


# ------------------------------------------------------------------ 3. verifier calibration
def eval_verifier(a, settings, data, docs_dir, out_dir):
    """Positives: a reference answer vs its own gold passage. Negatives: the same answer vs a
    *different* question's gold passage from the same document (topically close: a hard case)."""
    from studyrag.verify import verify_answer
    emb, _ = models(settings, rerank=False)
    lib = build_library(settings, emb, docs_dir, f"{a.set}-{settings.chunk_strategy}")
    qs = [q for q in data if q["answerable"]]
    gold_chunk = {}
    for q in qs:
        c = next((c for c in lib.chunks if is_relevant(c, q["gold"])), None)
        if c:
            gold_chunk[q["id"]] = c
    pairs = []
    for q in qs:
        if q["id"] not in gold_chunk:
            continue
        own = gold_chunk[q["id"]]
        pairs.append((q["answer"], own, True))
        others = [gold_chunk[o["id"]] for o in qs if o["id"] in gold_chunk and o["id"] != q["id"]
                  and gold_chunk[o["id"]].chunk_id != own.chunk_id]
        same_doc = [c for c in others if c.doc_id == own.doc_id] or others
        if same_doc:
            pairs.append((q["answer"], random.Random(q["id"]).choice(same_doc), False))
    scores = []
    for text, chunk, label in pairs:
        ver = verify_answer(text, [chunk], emb, threshold=0.0)
        s = min(c.support for c in ver.checks) if ver.checks else 1.0  # weakest sentence decides
        scores.append((s, label))
    rows = []
    for th in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
        tp = sum(s >= th and l for s, l in scores); fp = sum(s >= th and not l for s, l in scores)
        fn = sum(s < th and l for s, l in scores); tn = sum(s < th and not l for s, l in scores)
        prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
        rows.append({"threshold": th, "precision": f"{prec:.2f}", "recall": f"{rec:.2f}",
                     "flags wrong (TNR)": f"{tn / (tn + fp) if tn + fp else 0:.2f}",
                     "F1": f"{2 * prec * rec / (prec + rec) if prec + rec else 0:.2f}"})
    table = md_table(rows, ["threshold", "precision", "recall", "flags wrong (TNR)", "F1"])
    print(f"{len(scores)} pairs\n{table}")
    save(out_dir, "verifier", {"n_pairs": len(scores), "rows": rows, "table": table})


# ------------------------------------------------------------------ 4. end-to-end answers
def fact_found(fact: str, answer: str, emb) -> bool:
    if support_score(fact, answer) >= 0.6:
        return True
    v = emb.embed_documents([fact, answer])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return float(v[0] @ v[1]) >= 0.85


def eval_answers(a, settings, data, docs_dir, out_dir):
    from studyrag.answer import QAEngine
    from studyrag.llm import load_llm
    emb, rr = models(settings)
    lib = build_library(settings, emb, docs_dir, f"{a.set}-{settings.chunk_strategy}")
    llm = load_llm(settings)
    qa = QAEngine(Retriever(lib, emb, rr, settings), llm, emb, settings)
    items = data[: a.limit] if a.limit else data
    rows = []
    for i, q in enumerate(items, 1):
        ans = qa.ask(q["question"])
        row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"], "status": ans.status,
               "answer": ans.text, "confidence": ans.confidence, "seconds": round(ans.seconds, 1),
               "supported_ratio": round(ans.verification.supported_ratio, 2), "cited": ans.cited,
               "auto_cited": ans.auto_cited}
        if q["answerable"] and ans.status == "answered":
            facts = q.get("key_facts") or [q["answer"]]
            found = [fact_found(f, ans.text, emb) for f in facts]
            row["fact_recall"] = round(sum(found) / len(found), 2)
            cited_chunks = [ans.sources[n - 1].chunk for n in ans.cited]
            row["citation_precision"] = (round(sum(is_relevant(c, q["gold"]) for c in cited_chunks)
                                               / len(cited_chunks), 2) if cited_chunks else 0.0)
            row["retrieved_gold"] = any(is_relevant(h.chunk, q["gold"]) for h in ans.sources)
            # reference-free: does the passage the model cited actually support the sentence citing it?
            own = [c for c in ans.verification.checks if c.cited and not c.auto]
            row["model_citations"] = len(own)
            row["model_citations_supported"] = sum(c.supported for c in own)
        rows.append(row)
        print(f"[{i}/{len(items)}] {q['id']} {ans.status:<9} {ans.seconds:5.1f}s  "
              f"facts={row.get('fact_recall', '-')}  cite_prec={row.get('citation_precision', '-')}")

    ans_rows = [r for r in rows if r["answerable"]]
    una_rows = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans_rows if r["status"] == "answered"]
    mean = lambda xs: round(statistics.mean(xs), 3) if xs else None  # noqa: E731
    summary = {
        "model": llm.name, "n": len(rows), "n_answerable": len(ans_rows), "n_unanswerable": len(una_rows),
        "answer_rate_on_answerable": mean([r["status"] == "answered" for r in ans_rows]),
        "key_fact_recall": mean([r["fact_recall"] for r in answered]),
        "fully_correct_rate": mean([r["fact_recall"] >= 0.99 for r in answered]),
        "mostly_correct_rate (>=50% facts)": mean([r["fact_recall"] >= 0.5 for r in answered]),
        "citation_precision": mean([r["citation_precision"] for r in answered]),
        "citation_support (cited passage supports the sentence)": (
            round(sum(r["model_citations_supported"] for r in answered)
                  / max(1, sum(r["model_citations"] for r in answered)), 3)),
        "answers_with_any_citation": mean([bool(r["cited"]) for r in answered]),
        "answers_needing_auto_citation": mean([bool(r.get("auto_cited")) for r in answered]),
        "faithfulness (verified sentence ratio)": mean([r["supported_ratio"] for r in answered]),
        "gold_passage_retrieved": mean([r["retrieved_gold"] for r in answered]),
        "refusal_rate_on_unanswerable": mean([r["status"] == "not_found" for r in una_rows]),
        "median_seconds": statistics.median(r["seconds"] for r in rows),
    }
    print(json.dumps(summary, indent=2))
    save(out_dir, f"answers_{llm.name}", summary, rows)


# ------------------------------------------------------------------ 5. MCQ quality
MCQ_CHECK_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": list("ABCD")}},
                    "required": ["answer"]}


def eval_mcq(a, settings, data, docs_dir, out_dir):
    from studyrag.llm import load_llm
    from studyrag.study.generate import generate_mcqs, select_passages
    emb, _ = models(settings, rerank=False)
    lib = build_library(settings, emb, docs_dir, f"{a.set}-{settings.chunk_strategy}")
    llm = load_llm(settings)
    passages = select_passages(lib, None, a.n, min_words=40, seed=7)
    t = time.perf_counter()
    mcqs = list(generate_mcqs(llm, passages, "medium", emb, seed=7))
    gen_s = time.perf_counter() - t
    rows, consistent = [], 0
    for q in mcqs:
        opts = "\n".join(f"{'ABCD'[j]}) {o}" for j, o in enumerate(q.options))
        msgs = [{"role": "user", "content": f'Passage:\n"""\n{q.source.text}\n"""\n\nUsing only the passage, '
                 f"answer this question.\n{q.question}\n{opts}"}]
        try:
            pick = llm.generate_json(msgs, MCQ_CHECK_SCHEMA, max_tokens=20, temperature=0.0).get("answer")
        except Exception:
            pick = None
        ok = pick == q.answer_letter
        consistent += ok
        rows.append({**q.to_dict(), "self_check_pick": pick, "self_check_ok": ok})
    n = len(mcqs)
    summary = {
        "model": llm.name, "passages": len(passages), "mcqs_produced": n,
        "yield_rate": round(n / len(passages), 3) if passages else 0,
        "grounded_rate": round(sum(q.grounded for q in mcqs) / n, 3) if n else 0,
        "flagged_rate": round(sum(bool(q.issues) for q in mcqs) / n, 3) if n else 0,
        "answer_key_consistency": round(consistent / n, 3) if n else 0,
        "raw_answer_position_counts": {k: sum(q.raw_answer == k for q in mcqs) for k in "ABCD"},
        "seconds_per_mcq": round(gen_s / n, 1) if n else None,
    }
    print(json.dumps(summary, indent=2))
    save(out_dir, f"mcq_{llm.name}", summary, rows)


# ------------------------------------------------------------------ 6. speed
def eval_speed(a, settings, data, docs_dir, out_dir):
    from studyrag.llm.llamacpp import LlamaCppLLM
    prompt = [{"role": "user", "content": ("Marketing is the process of promoting products. " * 60)
               + "\nSummarise the text above in two sentences."}]
    rows = []
    for threads in (2, 4):
        llm = LlamaCppLLM(settings.llm_model_path, n_ctx=2048, n_threads=threads)
        n_prompt = llm.count_tokens(prompt[0]["content"])
        llm.generate([{"role": "user", "content": "Hi"}], max_tokens=4)       # warm-up
        t0 = time.perf_counter()
        first, n_out = None, 0
        for _ in llm.generate(prompt, max_tokens=96, temperature=0.0, stream=True):
            if first is None:
                first = time.perf_counter()
            n_out += 1
        t1 = time.perf_counter()
        rows.append({"threads": threads, "prompt tokens": n_prompt,
                     "prompt tok/s": round(n_prompt / (first - t0), 1),
                     "generate tok/s": round((n_out - 1) / (t1 - first), 1) if n_out > 1 else None,
                     "time to first token (s)": round(first - t0, 1)})
        print(rows[-1])
        del llm
    table = md_table(rows, list(rows[0]))
    print(table)
    save(out_dir, f"speed_{settings.llm_model_path.stem}", {"rows": rows, "table": table})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["retrieval", "refusal", "verifier", "answers", "mcq", "speed"])
    p.add_argument("--set", default="sample", choices=list(SETS))
    p.add_argument("-n", type=int, default=15, help="MCQs to generate")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (answers)")
    a = p.parse_args()
    settings = Settings.from_env()
    ds, docs_dir, out_dir = SETS[a.set]
    data = load_dataset(ds) if ds.exists() else []
    if not data and a.task not in ("mcq", "speed"):
        sys.exit(f"Dataset not found or empty: {ds}")
    fn = {"retrieval": eval_retrieval, "refusal": eval_refusal, "verifier": eval_verifier,
          "answers": eval_answers, "mcq": eval_mcq, "speed": eval_speed}[a.task]
    fn(a, settings, data, docs_dir, out_dir)


if __name__ == "__main__":
    main()
