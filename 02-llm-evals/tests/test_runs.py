"""Runs, comparisons, the regression gate and the CLI."""

import json

import pytest

from evalkit.cli import main
from evalkit.compare import compare
from evalkit.gate import check
from evalkit.runs import Run, aggregate


def make_run(name, correct, seconds=None):
    items = [{"id": f"q{i}", "correct": c, "seconds": (seconds or [10.0] * len(correct))[i], "note": "text"}
             for i, c in enumerate(correct)]
    return Run.build("demo", name, items)


def test_build_detects_numeric_metrics_and_cis():
    run = make_run("v1", [True] * 8 + [False] * 2)
    assert set(run.metrics) == {"correct", "seconds"}           # "note" is text, "id" is skipped
    m = run.metrics["correct"]
    assert m["value"] == 0.8 and m["ci_low"] < 0.8 < m["ci_high"] and m["n"] == 10
    with pytest.raises(ValueError, match="id"):
        Run.build("demo", "x", [{"correct": True}])


def test_aggregate_skips_missing_values():
    agg = aggregate([{"a": 1.0}, {"a": None}, {"b": 2}], ["a", "b", "c"])
    assert agg["a"]["n"] == 1 and agg["b"]["value"] == 2.0 and "c" not in agg


def test_save_load_roundtrip(tmp_path):
    run = make_run("v1", [True, False])
    path = run.save(tmp_path)
    assert path == tmp_path / "demo" / "v1.json"
    loaded = Run.load(path)
    assert loaded.items == run.items and loaded.metrics == run.metrics and "python" in loaded.env


def test_compare_finds_flips_and_significance():
    base = make_run("v1", [False] * 12 + [True] * 28)
    new = make_run("v2", [True] * 11 + [False] + [True] * 28)       # fixes 11 items, breaks none
    d = compare(base, new).get("correct")
    assert d.test == "McNemar" and d.improved == sorted(f"q{i}" for i in range(11)) and not d.regressed
    assert d.delta == pytest.approx(11 / 40) and d.significant
    noisy = make_run("v3", [True] + [False] * 11 + [False] + [True] * 27)   # fixes 1, breaks 1
    assert not compare(base, noisy).get("correct").significant


def test_gate_absolute_and_baseline_rules():
    base = make_run("v1", [True] * 36 + [False] * 4)
    worse = make_run("v2", [True] * 30 + [False] * 10)
    rules = {"correct": {"min": 0.8, "max_drop": 0.05},
             "seconds": {"higher_is_better": False, "max": 20}}
    results = {r.metric: r for r in check(worse, rules, base)}
    assert not results["correct"].ok and "baseline" in results["correct"].message
    assert results["seconds"].ok
    # the same drop is tolerated when significance is required and the test can't confirm it
    slightly = make_run("v3", [True] * 33 + [False] * 7)
    soft = {"correct": {"max_drop": 0.05, "require_significance": True}}
    assert check(slightly, soft, base)[0].ok


def test_cli_import_compare_gate(tmp_path, capsys):
    details = tmp_path / "d.jsonl"
    details.write_text("\n".join(json.dumps({"id": i, "correct": i % 4 != 0}) for i in range(20)))
    assert main(["import", str(details), "--suite", "s", "--name", "v1", "--out", str(tmp_path)]) == 0
    run_path = tmp_path / "s" / "v1.json"
    assert main(["compare", str(run_path), str(run_path)]) == 0
    assert "| correct | 20 |" in capsys.readouterr().out
    rules = tmp_path / "gate.toml"
    rules.write_text("[metrics.correct]\nmin = 0.9\n")
    assert main(["gate", str(run_path), "--rules", str(rules)]) == 1
    assert "FAIL  correct: 0.750 < min 0.9" in capsys.readouterr().out


def test_lower_is_better_metrics_flip_direction():
    base = make_run("v1", [True] * 4, seconds=[10.0, 10.0, 10.0, 10.0])
    new = make_run("v2", [True] * 4, seconds=[5.0, 5.0, 5.0, 12.0])
    d = compare(base, new).get("seconds")
    assert not d.higher_is_better and d.improved == ["q0", "q1", "q2"] and d.regressed == ["q3"]
    d = compare(base, new, lower_is_better=set()).get("seconds")         # explicit override
    assert d.higher_is_better and d.regressed == ["q0", "q1", "q2"]
