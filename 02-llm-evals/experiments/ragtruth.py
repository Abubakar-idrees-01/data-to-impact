"""RAGTruth -> evalkit cases, and the fixed samples used to validate the judge.

RAGTruth (Niu et al., 2024; MIT licence) has ~18k responses written by 6 LLMs for 3 RAG-style tasks
(question answering over passages, news summarization, data-to-text), each annotated by humans with the exact
spans that are hallucinated (contradicting the source, or adding information that isn't in it).
A response is a positive case when humans marked at least one span.

    python experiments/ragtruth.py            # writes datasets/ragtruth_{dev,test}.jsonl and the id lists

The raw files are downloaded to data/raw/ragtruth/ (see docs/SETUP.md) and never committed.
Only the sampled ids are committed (datasets/ragtruth_ids.json), so the sample can be rebuilt exactly.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalkit.dataset import Case, save_cases  # noqa: E402

RAW = ROOT / "data" / "raw" / "ragtruth"
OUT = ROOT / "datasets"
SEED = 13
# (split in RAGTruth, task, per label)  -> balanced: same number with and without hallucination
PLAN = {
    "dev": [("train", "QA", 10), ("train", "Summary", 10), ("train", "Data2txt", 10)],
    "test": [("test", "QA", 60), ("test", "Summary", 30), ("test", "Data2txt", 30)],
}


def task_and_source(src: dict) -> tuple[str, str]:
    """The instruction the generator got, and the source text it had to stay faithful to."""
    info, kind = src["source_info"], src["task_type"]
    if kind == "QA":
        return (f"Briefly answer the question \"{info['question']}\" based strictly on the passages.",
                info["passages"])
    if kind == "Summary":
        return src["prompt"].split("\n", 1)[0].strip(), info
    # Data2txt: the instruction precedes "Structured data:"
    instruction = src["prompt"].split("Structured data:")[0].replace("Instruction:", "").strip()
    return instruction, json.dumps(info, ensure_ascii=False)


def to_case(resp: dict, src: dict) -> Case:
    task, source = task_and_source(src)
    types = sorted({lab["label_type"] for lab in resp["labels"]})
    return Case(id=f"rt{resp['id']}", input=task, context=[source],
                tags={"task": src["task_type"], "generator": resp["model"], "split": resp["split"]},
                meta={"response": resp["response"], "hallucinated": bool(resp["labels"]),
                      "label_types": types, "spans": [lab["text"] for lab in resp["labels"]]})


def load_raw():
    sources = {}
    with open(RAW / "source_info.jsonl", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            sources[s["source_id"]] = s
    with open(RAW / "response.jsonl", encoding="utf-8") as f:
        responses = [json.loads(line) for line in f]
    return responses, sources


def build() -> dict[str, list[Case]]:
    responses, sources = load_raw()
    rng = random.Random(SEED)
    out, ids = {}, {}
    for name, strata in PLAN.items():
        cases = []
        for split, task, k in strata:
            pool = [r for r in responses if r["split"] == split and r["quality"] == "good"
                    and sources[r["source_id"]]["task_type"] == task]
            for positive in (True, False):
                group = sorted((r for r in pool if bool(r["labels"]) == positive), key=lambda r: int(r["id"]))
                cases += [to_case(r, sources[r["source_id"]]) for r in rng.sample(group, k)]
        rng.shuffle(cases)
        out[name], ids[name] = cases, [c.id for c in cases]
    OUT.mkdir(exist_ok=True)
    (OUT / "ragtruth_ids.json").write_text(json.dumps({"seed": SEED, "plan": PLAN, **ids}, indent=1))
    for name, cases in out.items():
        save_cases(cases, ROOT / "data" / f"ragtruth_{name}.jsonl")
    return out


if __name__ == "__main__":
    for name, cases in build().items():
        print(name, len(cases), Counter((c.tags["task"], c.meta["hallucinated"]) for c in cases),
              Counter(c.tags["generator"] for c in cases))
