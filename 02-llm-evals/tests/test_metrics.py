"""Metrics are checked against hand-computed or textbook values, not just "it runs"."""

import math

import pytest

from evalkit.dataset import Case, load_cases, save_cases
from evalkit.metrics import agreement, retrieval, stats, text


# ---------------------------------------------------------------- text
def test_normalize_and_exact_match():
    assert text.normalize("The  Eiffel-Tower!") == "eiffel tower"
    assert text.exact_match("the Eiffel Tower.", "Eiffel tower") == 1.0
    assert text.exact_match("Paris", "London") == 0.0


def test_token_f1_squad_example():
    # pred tokens: [eiffel, tower, in, paris] ; ref: [paris]  -> P=1/4, R=1 -> F1=0.4
    assert text.token_f1("The Eiffel tower in Paris", "Paris") == pytest.approx(0.4)
    assert text.token_f1("", "") == 1.0 and text.token_f1("x", "") == 0.0


def test_rouge_l():
    # LCS of [a b c d] and [a c d e] = [a c d] = 3 -> P = R = 3/4
    assert text.rouge_l("a1 b1 c1 d1", "a1 c1 d1 e1") == pytest.approx(0.75)


def test_key_fact_recall_ignores_stopwords_and_order():
    answer = "Affiliates are usually paid per sale, and some programs pay per click."
    assert text.fact_found("paid per sale", answer)
    assert not text.fact_found("cost per mille", answer)
    assert text.key_fact_recall(answer, ["paid per sale", "cost per mille"]) == 0.5


# ---------------------------------------------------------------- retrieval
def test_retrieval_metrics():
    ranked = ["d3", "d1", "d7", "d2"]
    rel = {"d1", "d2"}
    assert retrieval.first_relevant_rank(ranked, rel) == 2
    assert retrieval.hit_at_k(ranked, rel, 1) == 0.0 and retrieval.hit_at_k(ranked, rel, 2) == 1.0
    assert retrieval.reciprocal_rank(ranked, rel) == 0.5
    assert retrieval.precision_at_k(ranked, rel, 4) == 0.5
    assert retrieval.recall_at_k(ranked, rel, 2) == 0.5
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1 + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert retrieval.ndcg_at_k(ranked, rel, 4) == pytest.approx(expected)


def test_retrieval_with_predicate():
    ranked = [{"page": 4}, {"page": 9}]
    assert retrieval.reciprocal_rank(ranked, lambda h: h["page"] == 9) == 0.5


# ---------------------------------------------------------------- agreement
def test_confusion_and_kappa_textbook_example():
    # Classic example: 50 items, both say yes 20, both no 15, A-yes/B-no 5, A-no/B-yes 10
    # po = 0.7; pe = 0.5*0.6 + 0.5*0.4 = 0.5 -> kappa = 0.4
    c = agreement.Confusion(tp=20, fp=5, fn=10, tn=15)
    assert c.accuracy == pytest.approx(0.7)
    assert c.kappa == pytest.approx(0.4)
    assert c.precision == pytest.approx(0.8) and c.recall == pytest.approx(20 / 30)
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert agreement.cohen_kappa(a, b) == pytest.approx(0.4)


def test_always_negative_judge_is_exposed():
    gold = [True] * 2 + [False] * 8
    c = agreement.Confusion.from_labels([False] * 10, gold)
    assert c.accuracy == 0.8               # looks good...
    assert c.kappa == 0.0 and c.recall == 0.0 and c.balanced_accuracy == 0.5   # ...but is useless


# ---------------------------------------------------------------- stats
def test_wilson_ci_contains_p_and_stays_in_range():
    lo, hi = stats.wilson_ci(10, 10)
    assert hi == 1.0 and 0.6 < lo < 0.75     # 95% CI for 10/10 is about [0.72, 1]
    lo, hi = stats.wilson_ci(5, 10)
    assert lo < 0.5 < hi


def test_mcnemar_exact():
    # b=1, c=9: p = 2 * (C(10,0) + C(10,1)) / 2^10 = 22/1024
    assert stats.mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert stats.mcnemar_exact(5, 5) == 1.0


def test_paired_bootstrap_detects_consistent_improvement():
    a = [0.0] * 30 + [1.0] * 10
    b = [1.0] * 25 + [0.0] * 5 + [1.0] * 10        # b fixes 25 items, breaks none
    res = stats.paired_bootstrap(a, b)
    assert res["delta"] == pytest.approx(25 / 40)
    assert res["ci_low"] > 0 and res["p"] < 0.01


def test_bootstrap_ci_over_pairs():
    pairs = [(True, True)] * 30 + [(False, False)] * 30 + [(True, False)] * 5
    kappa = lambda ps: agreement.Confusion.from_labels([p for p, _ in ps], [g for _, g in ps]).kappa  # noqa: E731
    lo, hi = stats.bootstrap_ci(pairs, kappa, n_resamples=500)
    assert lo < kappa(pairs) < hi and hi <= 1.0


def test_correlations_and_percentile():
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 1000]) == pytest.approx(1.0)
    assert stats.ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    assert stats.percentile([1, 2, 3, 4], 50) == 2.5


# ---------------------------------------------------------------- dataset
def test_cases_roundtrip_with_aliases(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"id": 1, "question": "Q?", "answer": "A", "passages": "ctx", "difficulty": "hard"}\n')
    [c] = load_cases(p)
    assert (c.id, c.input, c.expected, c.context, c.meta) == ("1", "Q?", "A", ["ctx"], {"difficulty": "hard"})
    save_cases([c], tmp_path / "out.jsonl")
    assert load_cases(tmp_path / "out.jsonl")[0] == c


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"id": "a", "input": "x"}\n{"id": "a", "input": "y"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(p)
    with pytest.raises(ValueError):
        Case.from_dict({"input": "no id"})
