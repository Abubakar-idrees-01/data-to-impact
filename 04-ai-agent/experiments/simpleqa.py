"""E1. Does web research help? SimpleQA questions answered closed-book vs by the agent with web tools.

SimpleQA (OpenAI, MIT licence) is 4,326 short factual questions with one indisputable answer, written so that strong
models often get them wrong from memory. We sample 40 exactly the way the official code does
(`random.Random(0).sample(rows, n)`) and grade with the official grader prompt (CORRECT / INCORRECT / NOT_ATTEMPTED),
run on a different model (qwen3.8-27b) through evalkit, cached.

    python experiments/simpleqa.py run --condition closed      # model alone, no tools
    python experiments/simpleqa.py run --condition agent       # Scout with web_search + web_read
    python experiments/simpleqa.py grade                       # grade every answer file, write evalkit runs

Contamination: SimpleQA's questions and answers are published (Hugging Face, GitHub, Kaggle mirrors). Those sites
are excluded from search and reading, and every URL the agent saw is checked afterwards for the dataset's name.

Long runs: progress is saved after every question (rerunning continues where it stopped); when the daily quota
runs out the script waits for it instead of switching models, so every answer in a run comes from one model.
Data: data/raw/ (gitignored), fetched by scripts/get_data.sh.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scout.agent import Agent                        # noqa: E402
from scout.config import Settings                    # noqa: E402
from scout.llm import LLM, LLMError                   # noqa: E402
from scout.tools import Registry                     # noqa: E402
from scout.web import Web                            # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "experiments" / "out"
RESULTS = ROOT / "results"
EXCLUDE = ["huggingface.co", "kaggle.com", "github.com/openai/simple-evals", "openaipublic.blob.core.windows.net",
           "paperswithcode.com", "github.com/openai", "raw.githubusercontent.com"]
LEAK = re.compile(r"simple[-_ ]?qa|simple-evals", re.I)
CLOSED_SYSTEM = "Answer the question as accurately as you can, in one short sentence. If you don't know, say so."
TASK = "{q}\n\nGive a short, direct final answer (one sentence), with the source number."


def load_sample(n: int) -> list[dict]:
    with (RAW / "simple_qa_test_set.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["id"] = f"sq{i:04d}"
    return random.Random(0).sample(rows, n)


def grader_template() -> str:
    src = (RAW / "simpleqa_eval.py").read_text(encoding="utf-8")
    return src.split('GRADER_TEMPLATE = """', 1)[1].split('""".strip()', 1)[0].strip()


def short(model: str) -> str:
    return model.split("/")[-1]


async def wait_for_quota(e: Exception) -> bool:
    m = re.search(r"frees up in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(e))
    if "daily token limit" not in str(e):
        return False
    secs = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))) if m else 600
    print(f"  daily quota reached, waiting {secs / 60:.0f} min", flush=True)
    await asyncio.sleep(secs + 30)
    return True


async def answer_closed(s: Settings, q: str) -> dict:
    llm = LLM.from_settings(s)
    llm.fallback_model = None
    t0 = time.time()
    r = await llm.chat([{"role": "system", "content": CLOSED_SYSTEM}, {"role": "user", "content": q}], max_tokens=1024)
    return {"answer": r.content, "steps": 1, "tool_calls": 0, "searches": 0, "reads": 0,
            "tokens": r.prompt_tokens + r.completion_tokens, "seconds": round(time.time() - t0, 2), "urls": []}


async def answer_agent(s: Settings, q: str, max_steps: int) -> dict:
    llm = LLM.from_settings(s)
    llm.fallback_model = None
    web = Web(s.tavily_key, page_chars=s.page_chars, exclude_domains=EXCLUDE)
    agent = Agent(llm, Registry().add(*web.tools()), max_steps=max_steps, token_budget=30_000,
                  max_tokens_per_call=s.max_tokens_per_call, context_limit=s.context_limit,
                  traces_dir=s.state_dir / "traces" / "simpleqa")
    r = await agent.run(TASK.format(q=q))
    if r.stopped == "error":
        raise LLMError(r.answer)
    return {"answer": r.answer, "steps": r.steps, "tool_calls": r.tool_calls, "searches": web.searches,
            "reads": web.reads, "tokens": r.total_tokens, "seconds": r.seconds, "stopped": r.stopped,
            "urls": [x.url for x in web.book.sources], "trace": r.trace_path}


async def run(args) -> None:
    s = Settings.from_env()
    if args.model:
        s.api_model = args.model
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"simpleqa-{args.condition}-{short(s.api_model)}.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
    sample = load_sample(args.n)
    print(f"{args.condition} with {s.api_model}: {len(done)}/{len(sample)} already done -> {path.name}", flush=True)
    for row in sample:
        if row["id"] in done:
            continue
        while True:
            try:
                out = await (answer_closed(s, row["problem"]) if args.condition == "closed"
                             else answer_agent(s, row["problem"], args.max_steps))
                break
            except LLMError as e:
                if not await wait_for_quota(e):
                    print(f"  {row['id']}: error {e}; retrying in 60 s", flush=True)
                    await asyncio.sleep(60)
        rec = {"id": row["id"], "question": row["problem"], "gold": row["answer"], "model": s.api_model, **out}
        rec["leak"] = any(LEAK.search(u) for u in out["urls"])
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {row['id']} {out['tokens']:>6} tok {out['seconds']:>5.0f}s  {out['answer'][:70]!r}", flush=True)


def grade(args) -> None:
    sys.path.insert(0, str(ROOT.parent / "02-llm-evals" / "src"))
    from evalkit.llm import ChatClient
    from evalkit.runs import Run
    client = ChatClient.from_env(dotenv=ROOT.parent / "02-llm-evals" / ".env", cache_dir=ROOT / ".cache" / "grader",
                                 model=args.grader, max_wait=900, max_retries=200)
    template = grader_template()
    RESULTS.mkdir(parents=True, exist_ok=True)
    for path in sorted(OUT.glob("simpleqa-*.jsonl")):
        recs = [json.loads(l) for l in path.read_text().splitlines()]
        items = []
        for r in recs:
            answer = re.sub(r"\s*\[\d+\](\[\d+\])*", "", r["answer"]).strip()      # citations aren't part of the answer
            reply = client.chat([{"role": "user", "content": template.format(question=r["question"], target=r["gold"],
                                                                             predicted_answer=answer)}],
                                temperature=0.0, max_tokens=600).text
            m = re.search(r"\b(A|B|C)\b", reply) or re.search(r"(A|B|C)", reply)
            g = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}[m.group(1) if m else "C"]
            items.append({"id": r["id"], "correct": g == "CORRECT", "attempted": g != "NOT_ATTEMPTED",
                          "incorrect": g == "INCORRECT", "tokens": r["tokens"], "seconds": r["seconds"],
                          "steps": r["steps"], "tool_calls": r["tool_calls"], "searches": r["searches"],
                          "leak": r["leak"], "grade": g, "answer": r["answer"], "gold": r["gold"],
                          "question": r["question"]})
        name = path.stem.removeprefix("simpleqa-")
        attempted = [i for i in items if i["attempted"]]
        extra = {"correct_given_attempted": sum(i["correct"] for i in attempted) / len(attempted)} if attempted else {}
        run_ = Run.build("scout-simpleqa", name, items,
                         metrics=["correct", "attempted", "incorrect", "tokens", "seconds", "steps", "tool_calls",
                                  "searches", "leak"],
                         config={"n": len(items), "grader": args.grader, "model": recs[0]["model"] if recs else "",
                                 "sample": "random.Random(0).sample(rows, n)"}, extra_metrics=extra)
        out = run_.save(RESULTS)
        print(f"{name}: correct {run_.value('correct'):.2f}  attempted {run_.value('attempted'):.2f}  "
              f"tokens {run_.value('tokens'):.0f}  -> {out.relative_to(ROOT)}")
    print(f"grader: {client.usage.calls} calls, {client.usage.cache_hits} cached")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--condition", choices=["closed", "agent"], required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--model", help="default: SCOUT_API_MODEL")
    p.add_argument("--max-steps", type=int, default=6)
    p = sub.add_parser("grade")
    p.add_argument("--grader", default="qwen/qwen3.8-27b")
    a = ap.parse_args()
    asyncio.run(run(a)) if a.cmd == "run" else grade(a)


if __name__ == "__main__":
    main()
