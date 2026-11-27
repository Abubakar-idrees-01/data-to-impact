"""Send each request to the smallest model that can handle it.

Two ideas, usable together:

**1. Route before answering (cheap, from the prompt alone).** A few text features say how hard a request looks:
numbers and maths words, reasoning words ("why", "explain", "prove", "step by step"), code, length. The rules were
fixed *before* looking at evaluation results, so they are a fair test, not a fit to the data:
- code, a multi-step word problem, reasoning words, or a long prompt -> the top rung;
- plain arithmetic -> the middle rung;
- anything else (short factual questions, chit-chat) -> the bottom rung.

**2. Cascade (escalate when unsure).** Answer with a small model; if its confidence is below a threshold τ, ask the
next model up. Confidence = geometric mean probability of the answer's tokens (`runtime.Generation.confidence`).
Small models are often *unsure* when they are wrong (0.8B on "37 × 43": 0.39, wrong; on "capital of France": 0.88,
right), so a low score is a cheap signal to get a second opinion. The price: an escalated question pays for both
models.

The policies are pure functions of per-model results (correct?, confidence, seconds), so `route_eval` runs every
model on every question once and then simulates every policy exactly, for any threshold, without re-running models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REASONING = re.compile(r"\b(why|explain|prove|derive|compare|analy[sz]e|step[- ]by[- ]step|reason|justify|"
                       r"evaluate|design|plan|trade-?offs?)\b", re.I)
WORD_PROBLEM = re.compile(r"\b(how (many|much|long|far|old)|total|each|per|remaining|left|percent|twice|half|"
                          r"times (as|more)|average)\b", re.I)
CODE = re.compile(r"```|\bdef |\bclass |\bfunction\b|\bSELECT\b|\bregex\b|\bpython\b|\bjavascript\b|\bbug\b", re.I)
ARITH = re.compile(r"\d\s*[-+*/×x÷−]\s*\d")


@dataclass
class Features:
    words: int
    numbers: int
    arithmetic: bool
    word_problem: bool
    reasoning: bool
    code: bool


def features(messages: list[dict]) -> Features:
    text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return Features(words=len(text.split()), numbers=len(re.findall(r"\d+(?:\.\d+)?", text)),
                    arithmetic=bool(ARITH.search(text)), word_problem=bool(WORD_PROBLEM.search(text)),
                    reasoning=bool(REASONING.search(text)), code=bool(CODE.search(text)))


def rule_route(f: Features, n_rungs: int) -> int:
    top, mid = n_rungs - 1, min(1, n_rungs - 1)
    if f.code or f.reasoning or f.words > 80 or (f.word_problem and f.numbers >= 2):
        return top
    if f.arithmetic or f.numbers >= 2:
        return mid
    return 0


# ---------- policies over measured results
# results[rung][task_id] = {"correct": bool, "confidence": float, "seconds": float}

def run_cascade(results: list[dict], task: str, start: int, tau: float, rungs: list[int] | None = None) -> tuple[bool, float, int]:
    """(correct, seconds spent, rung that answered). Escalates through `rungs` from `start` while confidence < tau."""
    rungs = rungs if rungs is not None else list(range(len(results)))
    seconds, i = 0.0, rungs.index(start) if start in rungs else 0
    while True:
        r = results[rungs[i]][task]
        seconds += r["seconds"]
        if r["confidence"] >= tau or i == len(rungs) - 1:
            return r["correct"], seconds, rungs[i]
        i += 1


def evaluate_policy(results: list[dict], tasks: list[str], choose_start, tau: float = 0.0,
                    rungs: list[int] | None = None) -> dict:
    """Accuracy and mean seconds per question for a start-rung chooser + cascade threshold (tau=0: no cascade)."""
    correct, seconds, used = 0, 0.0, []
    for t in tasks:
        ok, s, rung = run_cascade(results, t, choose_start(t), tau, rungs)
        correct += ok
        seconds += s
        used.append(rung)
    n = len(tasks)
    return {"accuracy": round(correct / n, 4), "seconds": round(seconds / n, 3),
            "answered_by": [round(used.count(r) / n, 3) for r in range(len(results))]}


def oracle(results: list[dict], tasks: list[str]) -> dict:
    """Upper bound: each question goes straight to the smallest model that gets it right (if any does)."""
    correct, seconds = 0, 0.0
    for t in tasks:
        ok = [r for r in range(len(results)) if results[r][t]["correct"]]
        rung = ok[0] if ok else 0
        correct += bool(ok)
        seconds += results[rung][t]["seconds"]
    return {"accuracy": round(correct / len(tasks), 4), "seconds": round(seconds / len(tasks), 3)}


def policy_items(results: list[dict], tasks: list[str], choose_start, tau: float = 0.0,
                 rungs: list[int] | None = None) -> list[dict]:
    """Per-question outcome of a policy (for evalkit: compare policies question by question)."""
    out = []
    for t in tasks:
        ok, s, rung = run_cascade(results, t, choose_start(t), tau, rungs)
        out.append({"id": t, "correct": ok, "seconds": round(s, 3), "rung": rung})
    return out


def pick_threshold(results: list[dict], dev: list[str], choose_start, taus: list[float], target_accuracy: float,
                   slack: float = 0.02, rungs: list[int] | None = None) -> float:
    """The cheapest τ whose accuracy on the dev questions is within `slack` of `target_accuracy` (else the best τ)."""
    scored = [(tau, evaluate_policy(results, dev, choose_start, tau, rungs)) for tau in taus]
    good = [(r["seconds"], tau) for tau, r in scored if r["accuracy"] >= target_accuracy - slack]
    if good:
        return min(good)[1]
    return max(scored, key=lambda x: (x[1]["accuracy"], -x[1]["seconds"]))[0]
