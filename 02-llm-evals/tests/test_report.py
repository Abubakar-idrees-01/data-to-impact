"""The HTML report renders, escapes text and prints every charted number."""

from evalkit.metrics.agreement import Confusion
from evalkit.report import Bar, bar_chart, confusion_matrix, page, table, tiles


def test_report_blocks():
    chart = bar_chart([Bar("judge <A>", 0.62, 0.5, 0.71), Bar("baseline", 0.3, kind="base")], "Kappa")
    assert "judge &lt;A&gt;" in chart and ">0.62<" in chart and ">0.30<" in chart and chart.count("<rect") == 2
    cm = confusion_matrix(Confusion(tp=50, fp=10, fn=20, tn=40), "Judge vs humans")
    assert all(f">{n}<" in cm for n in (50, 10, 20, 40)) and "missed" in cm and "false alarm" in cm
    html = page("T", tiles([("0.6", "kappa", "95% CI")]) + table(["a", "b"], [["<x>", "1"]], numeric=(1,)))
    assert ">-0.2<" in bar_chart([Bar("x", 0.5)], "t", xmin=-0.2, ticks=6) and ">0<" in bar_chart([Bar("x", 0.5)], "t", xmin=-0.2, ticks=6)
    assert html.startswith("<!doctype html>") and "prefers-color-scheme:dark" in html and "<x>" in html
