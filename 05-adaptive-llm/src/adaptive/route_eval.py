"""Evaluate routing: every model answers every question once, then every policy is simulated from those answers.

    python -m adaptive.route_eval run          # all local models × 128 questions (~1 h on the i5; resumable)
    python -m adaptive.route_eval summary      # policies: fixed model, rule router, cascades, router + cascade, oracle

Per (model, question) we store the reply, whether it is correct, the probability of every generated token, and the
seconds it took (load time excluded: models stay loaded in a real engine). Greedy decoding (temperature 0), so a
rerun gives the same answers.

**Confidence** for a reply = geometric mean probability of its *answer* tokens: the whole reply for capitals and
arithmetic; for word problems only the final "Answer: …" line (a long, fluent explanation is confident about the
words, not the number).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .hardware import profile, slug
from .router import evaluate_policy, features, oracle, pick_threshold, policy_items, rule_route
from .tasks import all_tasks, final_answer, grade

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "routing"
EVALKIT = ROOT.parent / "02-llm-evals" / "src"          # project 2's evaluation toolkit (standard library only)
MAX_TOKENS = {"capital": 24, "arith": 24, "gsm8k": 320}
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def runs_path(machine: str) -> Path:
    return OUT / f"runs-{machine}.jsonl"


def answer_confidence(probs: list[float], answer_tokens: int) -> float:
    span = probs[-(answer_tokens + 1):] if answer_tokens else probs       # + the end-of-turn token
    span = span or [0.0]
    return math.exp(sum(math.log(max(p, 1e-12)) for p in span) / len(span))


def run(models_dir: Path, threads: int, machine: str, limit: int | None = None, only: str | None = None) -> None:
    from .runtime import LocalModel
    tasks = all_tasks()
    if limit:                                            # the first `limit` questions of each family (smoke tests)
        tasks = [t for t in tasks if int(t["id"].rsplit("-", 1)[1]) < limit]
    path = runs_path(machine)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {(r["model"], r["task"]) for r in map(json.loads, path.read_text().splitlines())}
    for gguf in sorted(models_dir.glob(only or "*.gguf")):
        m = None
        for t in tasks:
            if m is None:
                m = LocalModel(gguf, n_ctx=2048, threads=threads)
            if (m.name, t["id"]) in done:
                continue
            g = m.generate(t["messages"], max_tokens=MAX_TOKENS[t["family"]], temperature=0, reuse=False,
                           confidence=True)
            k = len(m.tokens(final_answer(g.text))) if t["family"] == "gsm8k" else 0
            rec = {"model": m.name, "params": m.info["params"], "task": t["id"], "family": t["family"],
                   "reply": g.text, "correct": grade(t, g.text), "seconds": g.total_s, "ttft_s": g.ttft_s,
                   "gen_tokens": g.gen_tokens, "answer_tokens": k, "probs": [round(p, 4) for p in g.token_probs]}
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{m.name:<22} {t['id']:<11} {'✓' if rec['correct'] else '✗'} "
                  f"conf {answer_confidence(rec['probs'], k):.2f} {g.total_s:5.1f}s  {g.text[-60:]!r}", flush=True)


def load(machine: str) -> tuple[list[str], list[dict], dict]:
    rows = [json.loads(l) for l in runs_path(machine).read_text().splitlines()]
    models = sorted({r["model"] for r in rows}, key=lambda n: next(r["params"] for r in rows if r["model"] == n))
    results = [{} for _ in models]
    family = {}
    for r in rows:
        results[models.index(r["model"])][r["task"]] = {
            "correct": r["correct"], "seconds": r["seconds"],
            "confidence": answer_confidence(r["probs"], r["answer_tokens"])}
        family[r["task"]] = r["family"]
    return models, results, family


def summary(machine: str) -> dict:
    models, results, family = load(machine)
    tasks = [t["id"] for t in all_tasks() if all(t["id"] in res for res in results)]
    msgs = {t["id"]: t["messages"] for t in all_tasks()}
    routed = {t: rule_route(features(msgs[t]), len(models)) for t in tasks}
    fams = sorted(set(family[t] for t in tasks))
    rows = []

    def add(name, f, tau=0.0, rungs=None):
        res = {"policy": name, "tau": tau, **evaluate_policy(results, tasks, f, tau, rungs)}
        res["by_family"] = {fam: evaluate_policy(results, [t for t in tasks if family[t] == fam], f, tau, rungs)["accuracy"]
                            for fam in fams}
        rows.append(res)

    for i, m in enumerate(models):
        add(f"always {m}", lambda t, i=i: i, rungs=[i])
    add("rule router", lambda t: routed[t])                     # tau = 0: never escalates
    n = len(models)
    for small, big in sorted({(0, n - 1), (0, 1), (1, n - 1)}):
        if not 0 <= small < big < n:
            continue
        for tau in TAUS:
            add(f"cascade {models[small]} -> {models[big]}", lambda t, s=small: s, tau, rungs=[small, big])
    for tau in TAUS:
        add("cascade through all", lambda t: 0, tau)
        add("rule router + cascade", lambda t: routed[t], tau)
    rows.append({"policy": "oracle (smallest correct model)", "tau": None, **oracle(results, tasks)})
    out = {"machine": machine, "models": models, "n_tasks": len(tasks), "families": fams, "policies": rows}
    (OUT / f"summary-{machine}.json").write_text(json.dumps(out, indent=1))
    return out


def export(machine: str) -> list[dict]:
    """Headline policies on held-out questions, saved as evalkit runs (results/evalkit/adaptive-routing/*.json).

    Questions alternate between a dev half (thresholds are chosen there) and a test half (reported), so a
    cascade's τ is never tuned on the questions it is scored on. τ* = the cheapest threshold whose dev accuracy is
    within 2 points of always using the largest model.
    """
    import sys
    sys.path.insert(0, str(EVALKIT))
    from evalkit.runs import Run
    models, results, family = load(machine)
    ids = [t["id"] for t in all_tasks() if all(t["id"] in res for res in results)]
    dev, test = ids[0::2], ids[1::2]
    msgs = {t["id"]: t["messages"] for t in all_tasks()}
    routed = {t: rule_route(features(msgs[t]), len(models)) for t in ids}
    n, top = len(models), len(models) - 1
    target = evaluate_policy(results, dev, lambda t: top, 0.0)["accuracy"]
    policies = [(f"always-{models[i]}", lambda t, i=i: i, 0.0, [i]) for i in range(n)]
    policies.append(("rule-router", lambda t: routed[t], 0.0, None))
    for name, start, rungs in [(f"cascade-{models[0]}-{models[top]}", lambda t: 0, [0, top]),
                               ("cascade-all", lambda t: 0, None), ("router-cascade", lambda t: routed[t], None)]:
        policies.append((name, start, pick_threshold(results, dev, start, TAUS, target, rungs=rungs), rungs))
    out = []
    for name, start, tau, rungs in policies:
        items = [{**it, "family": family[it["id"]], "model": models[it["rung"]]}
                 for it in policy_items(results, test, start, tau, rungs)]
        run = Run.build("adaptive-routing", name.lower(), items, metrics=["correct", "seconds"],
                        config={"machine": machine, "tau": tau, "models": models, "split": "test (odd positions)",
                                "dev_target_accuracy": target})
        run.save(ROOT / "results" / "evalkit")
        out.append({"policy": name, "tau": tau, "accuracy": run.value("correct"), "seconds": run.value("seconds"),
                    "ci": [run.metrics["correct"]["ci_low"], run.metrics["correct"]["ci_high"]]})
    (OUT / f"headline-{machine}.json").write_text(json.dumps(out, indent=1))
    print(f"\nHeld-out test half ({len(test)} questions); τ chosen on the dev half:")
    for r in out:
        print(f"  {r['policy']:<44} τ={r['tau']:<5} accuracy {r['accuracy']:.3f} "
              f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]  {r['seconds']:.2f} s/question")
    return out


def print_summary(s: dict) -> None:
    print(f"{s['n_tasks']} questions, models: {', '.join(s['models'])}\n")
    print(f"{'policy':<52} {'τ':>5} {'acc':>6} {'s/q':>6}  " + " ".join(f"{f:>7}" for f in s["families"]))
    for r in s["policies"]:
        fam = " ".join(f"{r['by_family'][f]:>7.2f}" for f in s["families"]) if "by_family" in r else ""
        tau = "" if not r["tau"] else r["tau"]
        print(f"{r['policy']:<52} {tau:>5} {r['accuracy']:>6.3f} {r['seconds']:>6.2f}  {fam}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "summary", "export"])
    ap.add_argument("--models", default=str(ROOT.parent / "01-rag-from-scratch" / "models" / "llm"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions of each family")
    ap.add_argument("--only", default=None, help="glob of model files to run, e.g. 'Qwen3.5-0.8B*'")
    ap.add_argument("--tag", default="", help="suffix for the results files (e.g. 'smoke')")
    a = ap.parse_args()
    hw = profile(bandwidth=False)
    machine = slug(hw.cpu) + (f"-{a.tag}" if a.tag else "")
    if a.action == "run":
        run(Path(a.models), a.threads or hw.physical_cores, machine, a.limit, a.only)
    if a.action == "export":
        export(machine)
        return
    print_summary(summary(machine))


if __name__ == "__main__":
    main()
