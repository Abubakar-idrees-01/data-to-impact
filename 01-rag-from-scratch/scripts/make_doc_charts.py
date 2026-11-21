"""Draw the charts used in the docs and PDFs (docs/images/chart_*.png).

The numbers are the published ones from docs/RESULTS.md and the git history; this script only draws them.
Needs matplotlib (a dev-only dependency, not in requirements.txt):  pip install matplotlib
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated palette)
GRAY, INK, MUTED, GRID = "#c9c8c2", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _title(ax, text, sub=None):
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=MUTED)


def results():
    metrics = ["Right passage in top 3", "Unanswerable questions refused", "Key facts in the answer",
               "Answers mostly correct", "Sentences backed by sources", "Citations that back the sentence"]
    public = [1.00, 1.00, 0.90, 0.97, 0.95, 0.96]
    private = [1.00, 1.00, 0.94, 0.96, 0.95, 0.98]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    h = 0.36
    for i, (p, q) in enumerate(zip(public, private)):
        y = len(metrics) - 1 - i
        ax.barh(y + h / 2 + 0.01, p, h, color=BLUE, label="Public set (Wikipedia, 42 q)" if i == 0 else None)
        ax.barh(y - h / 2 - 0.01, q, h, color=ORANGE, label="Your lectures (OCR'd slides, 35 q)" if i == 0 else None)
        ax.text(p + 0.01, y + h / 2, f"{p:.2f}", va="center", fontsize=8.5, color=MUTED)
        ax.text(q + 0.01, y - h / 2, f"{q:.2f}", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(metrics)), metrics[::-1])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    _title(ax, "StudyRAG v1: measured quality", "Share of questions (higher is better). Qwen3.5-2B on a 2015 i5 CPU.")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, -0.22), ncol=2, frameon=False, fontsize=9)
    fig.savefig(OUT / "chart_results.png")
    plt.close(fig)


def models():
    names = ["Qwen3.5-0.8B", "Qwen2.5-1.5B", "Qwen3.5-2B\n(default)"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"wspace": 0.45})
    secs = [14.6, 22.6, 31.9]
    a.barh(range(3), secs, color=[GRAY, GRAY, BLUE], height=0.55)
    for i, s in enumerate(secs):
        a.text(s + 0.6, i, f"{s:.0f} s", va="center", fontsize=9, color=MUTED)
    a.set_yticks(range(3), names)
    a.set_xlim(0, 38)
    _title(a, "Time per answer", "median seconds, public set (lower is faster)")
    sup = [None, 0.80, 0.98]
    b.barh([1, 2], sup[1:], color=[GRAY, BLUE], height=0.55)
    b.text(0.02, 0, "not measured", va="center", fontsize=9, color=MUTED, style="italic")
    for i in (1, 2):
        b.text(sup[i] + 0.02, i, f"{sup[i]:.2f}", va="center", fontsize=9, color=MUTED)
    b.set_yticks(range(3), names)
    b.set_xlim(0, 1.15)
    _title(b, "Trustworthy citations", "on your OCR'd slides (higher is better)")
    for ax in (a, b):
        ax.set_ylim(2.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.savefig(OUT / "chart_models.png")
    plt.close(fig)


def timeline():
    day = "2026-11-21 "
    t = lambda s: datetime.strptime(day + s, "%Y-%m-%d %H:%M")  # noqa: E731
    phases = [  # (label, start, end) from the commit history
        ("1  Repo + plan", "14:30", "16:00"),
        ("2  Build the whole system", "16:00", "16:40"),
        ("3  First measurements", "16:40", "17:35"),
        ("4  Real slides: OCR fixes", "17:35", "18:02"),
        ("5  Your course outline: tables", "18:02", "19:14"),
        ("6  Regression + model comparison", "19:14", "21:32"),
        ("7  MCQ self-check, screenshots, release", "21:32", "22:17"),
    ]
    found = [("17:08", "rerank + refusal"),
             ("18:01", "OCR garble 3.4%→0%"), ("19:13", "table refusal"),
             ("20:43", "'Week 5' regression"), ("21:45", "MCQ self-check")]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    for i, (label, s, e) in enumerate(phases):
        ax.barh(i, t(e) - t(s), left=t(s), height=0.55, color=BLUE)
        ax.text(t(e) + (t("14:40") - t("14:30")) * 0.3, i, label, va="center", fontsize=8.8, color=INK)
    for k, (s, label) in enumerate(found):
        ax.axvline(t(s), color=ORANGE, lw=1, ls=(0, (2, 2)), zorder=0)
        ax.text(t(s), -0.85 - 0.45 * (k % 2), " " + label, fontsize=7.8, color=ORANGE, va="center")
    ax.set_yticks([])
    ax.set_ylim(len(phases) - 0.4, -1.6)
    ax.set_xlim(t("14:20"), t("23:45"))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    _title(ax, "How StudyRAG was built: 21 Nov 2026",
           "Blue bars: phases (from the commit times). Dashed orange lines: a problem the evaluation found and fixed.")
    fig.savefig(OUT / "chart_timeline.png")
    plt.close(fig)


if __name__ == "__main__":
    results()
    models()
    timeline()
    print("wrote", *sorted(p.name for p in OUT.glob("chart_*.png")))
