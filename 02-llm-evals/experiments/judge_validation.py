"""Does the LLM judge agree with human annotators? Measured on RAGTruth.

    python experiments/judge_validation.py --split dev  --rubric faithfulness/claims --model qwen/qwen3.8-27b
    python experiments/judge_validation.py --split test --baselines        # non-LLM detectors, no API

Each configuration is saved as an evalkit run (results/judge-ragtruth-<split>/<name>.json) whose items record
the judge's label, the human label and whether they agree. Agreement metrics (precision, recall, F1, Cohen's
kappa, balanced accuracy) get bootstrap 95% confidence intervals, and two judges can be compared item by item
with `evalkit compare` on the `agree` metric.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalkit import baselines  # noqa: E402
from evalkit.dataset import load_cases  # noqa: E402
from evalkit.judge import Judge  # noqa: E402
from evalkit.llm import ChatClient  # noqa: E402
from evalkit.metrics.agreement import Confusion  # noqa: E402
from evalkit.metrics.stats import bootstrap_ci  # noqa: E402
from evalkit.runs import Run  # noqa: E402

RESULTS = ROOT / "results"
AGREEMENT = ["kappa", "f1", "precision", "recall", "balanced_accuracy", "accuracy"]


def agreement_metrics(items: list[dict]) -> dict[str, float]:
    """Agreement over items that got a verdict, with bootstrap CIs; errors are counted separately."""
    scored = [(it["pred"], it["gold"]) for it in items if it["pred"] is not None]
    out = {"n_errors": sum(it["pred"] is None for it in items)}
    if not scored:
        return out
    conf = lambda pairs: Confusion.from_labels([p for p, _ in pairs], [g for _, g in pairs])  # noqa: E731
    full = conf(scored).summary()
    for m in AGREEMENT:
        lo, hi = bootstrap_ci(scored, lambda ps, m=m: conf(ps).summary()[m], n_resamples=1000)
        out[m], out[f"{m}_ci_low"], out[f"{m}_ci_high"] = full[m], lo, hi
    out.update({k: full[k] for k in ("tp", "fp", "fn", "tn")})
    return out


def slices(items: list[dict], key) -> dict[str, dict]:
    groups = defaultdict(list)
    for it in items:
        if it["pred"] is not None:
            groups[key(it)].append(it)
    return {g: Confusion.from_labels([i["pred"] for i in its], [i["gold"] for i in its]).summary()
            for g, its in sorted(groups.items())}


def print_report(name: str, items: list[dict]) -> None:
    m = agreement_metrics(items)
    print(f"\n== {name}: n={len(items)} errors={m['n_errors']}")
    for k in AGREEMENT:
        if k in m:
            print(f"   {k:<18} {m[k]:.3f}  [{m[k + '_ci_low']:.3f}, {m[k + '_ci_high']:.3f}]")
    for title, key in [("task", lambda i: i["task"]), ("generator", lambda i: i["generator"])]:
        print(f"   by {title}:")
        for g, s in slices(items, key).items():
            print(f"     {g:<22} n={s['n']:<4} kappa={s['kappa']:.2f} f1={s['f1']:.2f} "
                  f"recall={s['recall']:.2f} precision={s['precision']:.2f}")
    pos = [i for i in items if i["gold"] and i["pred"] is not None]
    subtle = [i for i in pos if all("Subtle" in t for t in i["label_types"])]
    evident = [i for i in pos if any("Evident" in t for t in i["label_types"])]
    for label, group in (("evident", evident), ("subtle-only", subtle)):
        if group:
            print(f"   recall on {label} hallucinations: {sum(i['pred'] for i in group) / len(group):.2f} "
                  f"(n={len(group)})")


def base_item(case) -> dict:
    return {"id": case.id, "gold": case.meta["hallucinated"], "task": case.tags["task"],
            "generator": case.tags["generator"], "label_types": case.meta["label_types"],
            "response_words": len(case.meta["response"].split())}


def finish(item: dict, pred) -> dict:
    item["pred"] = pred
    item["agree"] = None if pred is None else float(pred == item["gold"])
    return item


def save(split: str, name: str, items: list[dict], config: dict) -> Path:
    run = Run.build(f"judge-ragtruth-{split}", name, items, metrics=["agree"], config=config,
                    extra_metrics=agreement_metrics(items))
    return run.save(RESULTS)


def run_baselines(split: str, cases, lexical_threshold: float) -> None:
    detectors = {
        "baseline-lexical": lambda c: baselines.lexical_hallucination(c.meta["response"], c.context[0],
                                                                      lexical_threshold),
        "baseline-numbers": lambda c: baselines.numbers_hallucination(c.meta["response"], c.context[0]),
        "baseline-always-hallucinated": lambda c: True,
    }
    for name, fn in detectors.items():
        items = [finish(base_item(c), fn(c)) for c in cases]
        print_report(name, items)
        save(split, name, items, {"detector": name, "lexical_threshold": lexical_threshold})


def tune_lexical(cases) -> float:
    """Pick the lexical threshold with the best kappa (only ever on the dev split)."""
    best = max((Confusion.from_labels([baselines.lexical_score(c.meta["response"], c.context[0]) < t
                                       for c in cases], [c.meta["hallucinated"] for c in cases]).kappa, t)
               for t in [x / 20 for x in range(4, 20)])
    print(f"lexical threshold tuned on dev: {best[1]:.2f} (kappa {best[0]:.3f})")
    return best[1]


def run_judge(split: str, cases, rubric: str, model: str | None, effort: str | None = None,
              max_tokens: int | None = None) -> None:
    extra = {"reasoning_effort": effort} if effort else None
    client = ChatClient.from_env(dotenv=ROOT / ".env", cache_dir=ROOT / ".cache" / "llm", extra_body=extra,
                                 max_wait=900, max_retries=1000,     # free tier: wait out the daily token cap
                                 **({"model": model} if model else {}))
    judge = Judge(client, rubric, max_tokens=max_tokens)
    name = f"{rubric.replace('/', '-')}__{client.model.replace('/', '_')}" + (f"__effort-{effort}" if effort else "")
    items, t0 = [], time.time()
    for i, c in enumerate(cases, 1):
        v = judge(task=c.input, source=c.context[0], response=c.meta["response"])
        item = finish(base_item(c), v.label)
        item.update({"n_problems": len(v.details.get("problems", [])), "error": v.error,
                     "problems": v.details.get("problems", [])})
        items.append(item)
        if i % 10 == 0 or i == len(cases):
            u = client.usage
            print(f"[{i}/{len(cases)}] {time.time() - t0:.0f}s  calls={u.calls} cached={u.cache_hits} "
                  f"tokens={u.prompt_tokens}+{u.completion_tokens} waited={sum(u.waits):.0f}s", flush=True)
    print_report(name, items)
    path = save(split, name, items, {"rubric": judge.rubric.id, "model": client.model, "extra_body": extra,
                                     "max_tokens": judge.max_tokens,
                                     "usage": vars(client.usage) | {"waits": len(client.usage.waits)}})
    print(f"saved {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--rubric", default="faithfulness/claims")
    p.add_argument("--model", help="override EVALKIT_API_MODEL")
    p.add_argument("--baselines", action="store_true", help="run the non-LLM detectors instead")
    p.add_argument("--effort", choices=["low", "medium", "high"], help="reasoning_effort for reasoning models")
    p.add_argument("--max-tokens", type=int, help="override the rubric's output budget")
    p.add_argument("--selected", help="take rubric/model/effort/max_tokens from a select_judge.py file")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    cases = load_cases(ROOT / "data" / f"ragtruth_{a.split}.jsonl")[: a.limit]
    if a.selected:
        sel = json.loads(Path(a.selected).read_text())
        a.rubric, a.model, a.max_tokens = sel["rubric"], sel["model"], sel.get("max_tokens")
        a.effort = (sel.get("extra_body") or {}).get("reasoning_effort")
    if a.baselines:
        threshold = tune_lexical(load_cases(ROOT / "data" / "ragtruth_dev.jsonl"))
        run_baselines(a.split, cases, threshold)
    else:
        run_judge(a.split, cases, a.rubric, a.model, a.effort, a.max_tokens)


if __name__ == "__main__":
    main()
