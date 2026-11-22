"""Self-contained HTML reports: inline SVG charts, light/dark themes, no JavaScript libraries, no network.

Building blocks return HTML strings; `page()` wraps them. Charts follow two rules:
every number shown as a bar also has its confidence interval, and every chart has a text/table equivalent.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Iterable, Sequence

from .metrics.agreement import Confusion
from .runs import Run

CSS = """
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e6e5e0;--grid:#eeede8;
--accent:#2a78d6;--accent2:#eb6834;--base:#b9b8b2;--good:#0b7a3e;--bad:#c2372f;--soft:#f3f6fb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--panel:#1c1c1b;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#8f8e88;--line:#2e2e2c;--grid:#262625;--accent:#3987e5;--accent2:#d95926;--base:#5f5e59;
--good:#3fb871;--bad:#e66767;--soft:#1f2630}}
:root[data-theme="dark"]{--bg:#141413;--panel:#1c1c1b;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e88;--line:#2e2e2c;
--grid:#262625;--accent:#3987e5;--accent2:#d95926;--base:#5f5e59;--good:#3fb871;--bad:#e66767;--soft:#1f2630}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,
"Noto Sans",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:30px;line-height:1.2;margin:4px 0 8px}h2{font-size:21px;margin:40px 0 8px}h3{font-size:16px;margin:20px 0 6px}
p{margin:8px 0}.lede{color:var(--ink2);font-size:17px;max-width:760px}.kicker{color:var(--accent);font-weight:700;
letter-spacing:.06em;text-transform:uppercase;font-size:12px}.muted{color:var(--muted);font-size:13px}
a{color:var(--accent)}code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--soft);
padding:1px 5px;border-radius:4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .v{font-size:28px;font-weight:700;font-variant-numeric:tabular-nums}.tile .l{color:var(--ink2);font-size:13px}
.tile .s{color:var(--muted);font-size:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600}td.num,th.num{text-align:right}
.pill{display:inline-block;padding:0 7px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
.good{color:var(--good)}.bad{color:var(--bad)}
svg text{fill:var(--ink2);font:12px system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
svg .lab{fill:var(--ink)}svg .grid{stroke:var(--grid)}svg .axis{stroke:var(--line)}
details{margin:6px 0}summary{cursor:pointer;color:var(--ink2)}
blockquote{margin:6px 0;padding:6px 10px;border-left:3px solid var(--accent2);background:var(--soft);font-size:13.5px}
"""


def esc(x) -> str:
    return html.escape(str(x))


def fmt(v: float | None, digits: int = 2) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def page(title: str, body: str, description: str = "") -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><meta name="description" content="{esc(description)}">'
            f"<style>{CSS}</style></head><body><main>{body}</main></body></html>")


def tiles(items: Iterable[tuple[str, str, str]]) -> str:
    """(value, label, small print) stat tiles."""
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="v">{esc(v)}</div><div class="l">{esc(l)}</div>'
        f'<div class="s">{esc(s)}</div></div>' for v, l, s in items) + "</div>"


def table(headers: Sequence[str], rows: Iterable[Sequence], numeric: Sequence[int] = ()) -> str:
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{c}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


@dataclass
class Bar:
    label: str
    value: float
    low: float | None = None
    high: float | None = None
    kind: str = "main"          # main | alt | base  -> accent / accent2 / gray


def bar_chart(bars: Sequence[Bar], title: str, xmin: float = 0.0, xmax: float = 1.0, ticks: int = 5,
              unit_note: str = "", width: int = 860) -> str:
    """Horizontal bars with 95% CI whiskers. Every value is also printed, so color is never the only cue."""
    left, right, row, top = 250, 70, 30, 28
    plot_w = width - left - right
    height = top + row * len(bars) + 30
    x = lambda v: left + (max(xmin, min(xmax, v)) - xmin) / (xmax - xmin) * plot_w  # noqa: E731
    colors = {"main": "var(--accent)", "alt": "var(--accent2)", "base": "var(--base)"}
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{esc(title)}">']
    for i in range(ticks + 1):
        v = round(xmin + (xmax - xmin) * i / ticks, 6) + 0.0      # no "-2.8e-17" or "-0" tick labels
        parts.append(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 6}" y2="{height - 24}"/>'
                     f'<text x="{x(v):.1f}" y="{height - 8}" text-anchor="middle">{v:g}</text>')
    if xmin < 0 < xmax:
        parts.append(f'<line class="axis" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top - 6}" y2="{height - 24}" '
                     f'style="stroke:var(--muted)"/>')
    for i, b in enumerate(bars):
        y = top + i * row
        x0 = x(max(xmin, 0)) if xmin < 0 else x(xmin)
        parts.append(f'<text class="lab" x="{left - 10}" y="{y + 15}" text-anchor="end">{esc(b.label)}</text>')
        x1 = x(b.value)
        parts.append(f'<rect x="{min(x0, x1):.1f}" y="{y + 4}" width="{abs(x1 - x0):.1f}" height="16" rx="3" '
                     f'fill="{colors[b.kind]}"><title>{esc(b.label)}: {b.value:.3f}</title></rect>')
        if b.low is not None and b.high is not None:
            lo, hi = x(b.low), x(b.high)
            parts.append(f'<line x1="{lo:.1f}" x2="{hi:.1f}" y1="{y + 12}" y2="{y + 12}" stroke="var(--ink)" '
                         f'stroke-width="1.5"/><line x1="{lo:.1f}" x2="{lo:.1f}" y1="{y + 7}" y2="{y + 17}" '
                         f'stroke="var(--ink)" stroke-width="1.5"/><line x1="{hi:.1f}" x2="{hi:.1f}" y1="{y + 7}" '
                         f'y2="{y + 17}" stroke="var(--ink)" stroke-width="1.5"/>')
        tx = max(x1, x(b.high) if b.high is not None else x1) + 6
        parts.append(f'<text x="{tx:.1f}" y="{y + 16}">{b.value:.2f}</text>')
    parts.append("</svg>")
    note = f'<p class="muted">{esc(unit_note)}</p>' if unit_note else ""
    return f'<div class="card"><h3>{esc(title)}</h3>{"".join(parts)}{note}</div>'


def confusion_matrix(c: Confusion, title: str, pos: str = "hallucinated", neg: str = "supported") -> str:
    cells = [(c.tp, f"judge: {pos}", f"human: {pos}", True), (c.fp, f"judge: {pos}", f"human: {neg}", False),
             (c.fn, f"judge: {neg}", f"human: {pos}", False), (c.tn, f"judge: {neg}", f"human: {neg}", True)]
    peak = max(1, *(v for v, *_ in cells))
    size, pad, lw = 120, 6, 150
    svg = [f'<svg viewBox="0 0 {lw + 2 * size + 20} {2 * size + 60}" width="100%" style="max-width:440px" '
           f'role="img" aria-label="{esc(title)}">',
           f'<text x="{lw + size / 2}" y="16" text-anchor="middle">human: {esc(pos)}</text>',
           f'<text x="{lw + 1.5 * size}" y="16" text-anchor="middle">human: {esc(neg)}</text>',
           f'<text x="{lw - 10}" y="{26 + size / 2}" text-anchor="end">judge: {esc(pos)}</text>',
           f'<text x="{lw - 10}" y="{26 + 1.5 * size}" text-anchor="end">judge: {esc(neg)}</text>']
    for k, (v, _, _, agree) in enumerate(cells):
        cx, cy = lw + (k % 2) * size, 26 + (k // 2) * size
        color = "var(--accent)" if agree else "var(--accent2)"
        opacity = 0.15 + 0.75 * v / peak
        svg.append(f'<rect x="{cx + pad / 2}" y="{cy + pad / 2}" width="{size - pad}" height="{size - pad}" rx="8" '
                   f'fill="{color}" fill-opacity="{opacity:.2f}"/>'
                   f'<text class="lab" x="{cx + size / 2}" y="{cy + size / 2 + 2}" text-anchor="middle" '
                   f'style="font-size:22px;font-weight:700">{v}</text>'
                   f'<text x="{cx + size / 2}" y="{cy + size / 2 + 22}" text-anchor="middle">'
                   f'{"agree" if agree else ("false alarm" if k == 1 else "missed")}</text>')
    svg.append("</svg>")
    return f'<div class="card"><h3>{esc(title)}</h3>{"".join(svg)}</div>'


def run_summary(run: Run, metrics: Sequence[str] | None = None) -> str:
    rows = []
    for k, m in run.metrics.items():
        if metrics and k not in metrics:
            continue
        ci = f'{fmt(m["ci_low"])} – {fmt(m["ci_high"])}' if "ci_low" in m else "–"
        rows.append([esc(k), fmt(m["value"], 3), ci, m.get("n", "")])
    return table(["metric", "value", "95% CI", "n"], rows, numeric=(1, 3))
