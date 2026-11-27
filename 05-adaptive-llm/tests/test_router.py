"""B3: routing rules, cascade arithmetic, graders, and the token-probability recorder."""

import numpy as np
import pytest

from adaptive.route_eval import answer_confidence
from adaptive.router import evaluate_policy, features, oracle, rule_route, run_cascade
from adaptive.runtime import Generation, ProbRecorder
from adaptive.tasks import CAPITALS, all_tasks, arith_tasks, final_answer, grade


def user(text):
    return [{"role": "system", "content": "x"}, {"role": "user", "content": text}]


@pytest.mark.parametrize("text, rung", [
    ("What is the capital of Peru?", 0),
    ("hi! how are you?", 0),
    ("What is 37 × 43?", 1),
    ("Tom has 3 apples and buys 5 more each day. How many apples does he have after 4 days?", 2),
    ("Why is the sky blue?", 2),
    ("Fix the bug in this python function: def f(x): return x +", 2),
])
def test_rule_router(text, rung):
    assert rule_route(features(user(text)), 3) == rung


def test_router_with_fewer_rungs_stays_in_range():
    assert rule_route(features(user("Why?")), 1) == 0
    assert rule_route(features(user("What is 2 + 2?")), 2) == 1


R = [  # results[rung][task]
    {"a": {"correct": True, "confidence": 0.9, "seconds": 1.0}, "b": {"correct": False, "confidence": 0.3, "seconds": 1.0}},
    {"a": {"correct": True, "confidence": 0.9, "seconds": 2.0}, "b": {"correct": False, "confidence": 0.4, "seconds": 2.0}},
    {"a": {"correct": True, "confidence": 0.9, "seconds": 5.0}, "b": {"correct": True, "confidence": 0.8, "seconds": 5.0}},
]


def test_cascade_escalates_only_when_unsure_and_pays_for_every_model_tried():
    assert run_cascade(R, "a", 0, 0.5) == (True, 1.0, 0)            # confident: the small model answers
    assert run_cascade(R, "b", 0, 0.5) == (True, 8.0, 2)            # 0.3, 0.4 < 0.5: all three models ran
    assert run_cascade(R, "b", 0, 0.5, rungs=[0, 2]) == (True, 6.0, 2)
    assert run_cascade(R, "b", 0, 0.0) == (False, 1.0, 0)           # τ = 0 never escalates
    assert run_cascade(R, "b", 0, 0.99) == (True, 8.0, 2)           # the last rung always answers


def test_policy_summary_and_oracle():
    p = evaluate_policy(R, ["a", "b"], lambda t: 0, 0.5)
    assert p == {"accuracy": 1.0, "seconds": 4.5, "answered_by": [0.5, 0.0, 0.5]}
    assert oracle(R, ["a", "b"]) == {"accuracy": 1.0, "seconds": 3.0}           # a on rung 0 (1 s), b on rung 2 (5 s)


def test_graders():
    t = all_tasks()
    au = next(x for x in t if x["family"] == "capital" and x["accept"] == CAPITALS["Australia"])
    assert grade(au, "Canberra.") and grade(au, "The capital is Canberra") and not grade(au, "Sydney")
    br = next(x for x in t if x["family"] == "capital" and x["accept"] == CAPITALS["Brazil"])
    assert grade(br, "Brasilia") and grade(br, "Brasília")
    mul = arith_tasks()[0]
    assert grade(mul, f"{mul['answer']}") and grade(mul, f"= {mul['answer']:,}") and not grade(mul, "1")
    gsm = {"family": "gsm8k", "answer": 1234.0}
    assert grade(gsm, "3 boxes cost $12, so...\nAnswer: $1,234") and not grade(gsm, "Answer: 1235\nwait 1234")
    assert final_answer("blah\nANSWER: 7 cats") == "7 cats"


def test_arith_tasks_are_reproducible_and_correct():
    a, b = arith_tasks(), arith_tasks()
    assert a == b and len(a) == 40
    for x in a:
        q = x["messages"][1]["content"].split("?")[0].replace("What is ", "")
        assert eval(q.replace("×", "*").replace("−", "-")) == x["answer"]


def test_answer_confidence_uses_the_answer_span():
    probs = [0.99] * 50 + [0.2, 0.3, 0.9]                  # fluent reasoning, unsure answer, sure end-of-turn
    assert answer_confidence(probs, 2) == pytest.approx((0.2 * 0.3 * 0.9) ** (1 / 3))
    assert answer_confidence(probs, 0) > 0.9               # the whole reply looks confident


def test_prob_recorder_looks_up_the_chosen_token_one_step_late():
    rec = ProbRecorder()
    logits = np.log(np.array([0.7, 0.2, 0.1], dtype=np.float32))
    rec([5, 6], logits.copy())                   # step 1: prompt of 2 tokens
    rec([5, 6, 1], logits.copy())                # token 1 was sampled (p = 0.2)
    rec([5, 6, 1, 0], logits.copy())             # token 0 was sampled (p = 0.7)
    assert rec.finish() == pytest.approx([0.2, 0.7, 0.7])      # last step: top probability
    g = Generation("x", "m", 1, 0, 3, 0.1, 1.0, 1.0, token_probs=[0.5, 0.5])
    assert g.confidence == pytest.approx(0.5)
