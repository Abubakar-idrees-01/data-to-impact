"""Build the PDF versions of the docs (docs/pdf/*.pdf) from the Markdown sources.

Markdown is the source of truth (it renders on GitHub); the PDFs are for reading and sharing.
Pipeline: pandoc (Markdown -> HTML) -> our print stylesheet -> headless Chrome (HTML -> PDF).

Needs: pandoc on PATH, Google Chrome/Chromium, and Playwright. Scout's venv has no Playwright, so run this with
any environment that has it, e.g. project 1's:  ../01-rag-from-scratch/.venv/bin/python scripts/build_pdfs.py
Mermaid diagrams are drawn by mermaid.js from a CDN, so a build with diagrams needs internet.

    python scripts/build_pdfs.py            # all PDFs
    python scripts/build_pdfs.py timeline   # only the ones whose file name contains "timeline"
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import shutil

from playwright.sync_api import sync_playwright


def chrome_path() -> str | None:
    return next((p for n in ("google-chrome", "chromium", "chromium-browser", "chrome") if (p := shutil.which(n))), None)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "pdf"
REPO_URL = "https://github.com/Abubakar-idrees-01/data-to-impact"
REPO_ROOT = ROOT.parent                      # the monorepo root, for GitHub links
MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"

# (output file, kicker, Markdown sources). Several sources become chapters of one PDF.
BOOKS = [
    ("Scout-Overview.pdf", "Project overview", [DOCS / "OVERVIEW.md"]),
    ("Scout-Timeline.pdf", "Project timeline", [DOCS / "TIMELINE.md"]),
    ("Scout-Setup-Guide.pdf", "Setup guide", [DOCS / "SETUP.md"]),
    ("Scout-How-It-Works.pdf", "Architecture", [DOCS / "HOW_IT_WORKS.md"]),
    ("Scout-Results.pdf", "Evaluation results", [DOCS / "RESULTS.md"]),
    ("Scout-Design-Decisions.pdf", "Design decisions", [DOCS / "DECISIONS.md"]),
    ("Scout-Concepts-Handbook.pdf", "Concepts handbook", sorted((DOCS / "concepts").glob("*.md"))),
]
# Markdown links between docs point at the matching PDF instead.
PDF_FOR = {src.resolve(): name for name, _, srcs in BOOKS if len(srcs) == 1 for src in srcs}
PDF_FOR.update({src.resolve(): "Scout-Concepts-Handbook.pdf" for src in BOOKS[-1][2]})

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
:root { --ink:#1b1b1a; --muted:#5c5b57; --line:#e3e2dc; --accent:#2a78d6; --soft:#f4f6fa; --code:#f5f5f3; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Noto Sans", "DejaVu Sans", "Liberation Sans", sans-serif, "Noto Color Emoji";
       color: var(--ink); background: #fff; font-size: 10.4pt; line-height: 1.55; margin: 0; }
.kicker { color: var(--accent); font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: 8.5pt; }
.meta { color: var(--muted); font-size: 8.5pt; margin: 2px 0 18px; padding-bottom: 10px; border-bottom: 2px solid var(--accent); }
h1 { font-size: 21pt; line-height: 1.2; margin: 4px 0 4px; }
h2 { font-size: 14pt; margin: 22px 0 8px; padding-bottom: 3px; border-bottom: 1px solid var(--line); }
h3 { font-size: 11.5pt; margin: 16px 0 6px; }
h2, h3, h4 { break-after: avoid; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
a { color: var(--accent); text-decoration: none; }
code { font-family: "DejaVu Sans Mono", "Liberation Mono", monospace; font-size: 8.6pt;
       background: var(--code); padding: 1px 4px; border-radius: 3px; }
pre { background: var(--code); border: 1px solid var(--line); border-radius: 6px; padding: 9px 11px;
      white-space: pre-wrap; word-break: break-word; break-inside: avoid; font-size: 8.4pt; line-height: 1.4; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 14px; font-size: 8.8pt; line-height: 1.4; }
th { background: var(--soft); text-align: left; font-weight: 700; }
th, td { border: 1px solid var(--line); padding: 5px 7px; vertical-align: top; }
tr { break-inside: avoid; }
img { max-width: 100%; max-height: 105mm; width: auto; border: 1px solid var(--line); border-radius: 6px; display: block; margin: 10px auto; break-inside: avoid; }
td img { margin: 4px auto; }
blockquote { border-left: 3px solid var(--accent); margin: 10px 0; padding: 2px 12px; color: var(--muted); background: var(--soft); }
hr { border: none; border-top: 1px solid var(--line); margin: 18px 0; }
details { margin: 8px 0; } summary { font-weight: 700; margin-bottom: 4px; list-style: none; }
.mermaid { text-align: center; margin: 12px 0; break-inside: avoid; }
.mermaid svg { width: 100% !important; max-width: 100% !important; height: auto; }
.chapter { break-before: page; }
.toc { list-style: none; padding: 0; } .toc li { padding: 5px 0; border-bottom: 1px dotted var(--line); }
"""

HEADER = """<div style="font-size:7.5pt;color:#8a8984;width:100%;padding:0 16mm;font-family:sans-serif;">
Scout &middot; {kicker}</div>"""
FOOTER = """<div style="font-size:7.5pt;color:#8a8984;width:100%;padding:0 16mm;font-family:sans-serif;
display:flex;justify-content:space-between;"><span>data-to-impact / 04-ai-agent</span>
<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>"""


def md_to_html(path: Path) -> tuple[str, str]:
    """Convert one Markdown file; return (title, body html) with links and images fixed for a PDF."""
    body = subprocess.run(["pandoc", "-f", "gfm", "-t", "html5", str(path)], capture_output=True,
                          text=True, check=True).stdout
    m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    title = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else path.stem
    if m:
        body = body.replace(m.group(0), "", 1)       # the title is rendered in our own header block

    def fix_src(mt):
        return f'src="{(path.parent / html.unescape(mt.group(1))).resolve().as_uri()}"'

    def fix_href(mt):
        href = html.unescape(mt.group(1))
        if re.match(r"^(https?:|mailto:|#)", href):
            return mt.group(0)
        target = (path.parent / href.split("#")[0]).resolve()
        if target in PDF_FOR:                        # another doc: link to its PDF, next to this one
            return f'href="{PDF_FOR[target]}"'
        if target.parent == OUT.resolve():           # already a link to one of the PDFs
            return f'href="{target.name}"'
        try:
            rel = target.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return mt.group(0)
        kind = "tree" if target.is_dir() else "blob"
        return f'href="{REPO_URL}/{kind}/main/{rel}"'

    body = re.sub(r'src="([^"]+)"', lambda mt: mt.group(0) if mt.group(1).startswith("http") else fix_src(mt), body)
    body = re.sub(r'href="([^"]+)"', fix_href, body)
    body = re.sub(r"<details(?![^>]*\bopen\b)", "<details open", body)   # paper can't be clicked open
    body = re.sub(r'<pre class="mermaid"><code>(.*?)</code></pre>', r'<div class="mermaid">\1</div>', body, flags=re.S)
    return title, body


def build_html(kicker: str, sources: list[Path]) -> str:
    today = date.today().strftime("%d %B %Y")
    head = f'<div class="kicker">Scout &middot; {kicker}</div>'
    if len(sources) == 1:
        title, body = md_to_html(sources[0])
        content = f'{head}<h1>{title}</h1><div class="meta">Generated {today} from {sources[0].name}</div>{body}'
    else:
        parts = [md_to_html(s) for s in sources]
        toc = "".join(f'<li><a href="#ch{i}">{t}</a></li>' for i, (t, _) in enumerate(parts, 1))
        content = (f'{head}<h1>Concepts Handbook</h1><div class="meta">Generated {today}. One learning note per core '
                   f'idea: intuition, then the algorithm, then our code and results.</div><ul class="toc">{toc}</ul>')
        content += "".join(f'<section class="chapter" id="ch{i}"><h1>{t}</h1>{b}</section>'
                           for i, (t, b) in enumerate(parts, 1))
    mermaid = ""
    if 'class="mermaid"' in content:
        mermaid = (f'<script src="{MERMAID_JS}"></script><script>mermaid.initialize({{startOnLoad:false,'
                   f'theme:"neutral",flowchart:{{htmlLabels:true}}}});</script>')
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{kicker}</title>'
            f'<style>{CSS}</style>{mermaid}</head><body>{content}</body></html>')


def main(filters: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    books = [b for b in BOOKS if not filters or any(f.lower() in b[0].lower() for f in filters)]
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        browser = p.chromium.launch(executable_path=chrome_path(), headless=True)
        page = browser.new_page()
        for name, kicker, sources in books:
            page_file = Path(tmp) / (name + ".html")
            page_file.write_text(build_html(kicker, sources), encoding="utf-8")
            page.goto(page_file.as_uri(), wait_until="networkidle")
            if page.locator(".mermaid").count():
                page.evaluate("mermaid.run()")
                page.wait_for_selector(".mermaid svg", timeout=30_000)
            page.wait_for_function("Array.from(document.images).every(i => i.complete)")
            page.pdf(path=str(OUT / name), format="A4", print_background=True, display_header_footer=True,
                     header_template=HEADER.format(kicker=kicker), footer_template=FOOTER,
                     margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
            print(f"{name}: {(OUT / name).stat().st_size / 1e6:.1f} MB")
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1:])
