"""Text normalisation applied to everything we extract.

Slides and PDFs are messy: odd bullet glyphs, non-breaking spaces, words split across lines
with hyphens, runs of blank lines. Cleaning here means every later step (BM25 tokens,
embeddings, the LLM prompt) sees consistent text.
"""

from __future__ import annotations

import re
import unicodedata

_BULLETS = "•◦▪▫●○■□►▶➢➤✓✔-–—*·"
_BULLET_LINE = re.compile(rf"^\s*[{re.escape(_BULLETS)}]+\s*")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t  -​]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0b", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # "mark-\neting" -> "marketing"
    lines = []
    for line in text.split("\n"):
        line = _SPACES.sub(" ", line).strip()
        if _BULLET_LINE.match(line):
            line = "- " + _BULLET_LINE.sub("", line)  # unify every bullet style to "- "
        lines.append(line)
    text = "\n".join(lines)
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def linearize_table(rows: list[list[str]]) -> list[str]:
    """Turn table rows into sentences that search models can read.

    Rerankers and embedders are trained on prose, so "Final examination | 40% | 1-5" looks like
    noise to them. Repeating the column headers in every row ("Final examination: Weight 40%;
    ...") gives each row its meaning. Measured on our course outline: the reranker's relevance for
    "What is the weightage of the final exam?" went from 0.02 (pipe rows) to 0.80.

    - 2-column tables are usually key/value lists ("Instructor | Dr X") -> "Instructor: Dr X.";
      so are tables whose header repeats ("Item | Details | Item | Details").
    - wider tables use the first row as headers and the first column as the row label.
    """
    rows = [[c for c in r] for r in rows if any(c.strip() for c in r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    if width == 1:
        return [r[0] for r in rows]
    header, body = rows[0], rows[1:]
    if width == 2 or (len(set(header)) < len(header) and width % 2 == 0):
        # key/value layout; repeated header labels ("Item | Details | Item | Details") mean
        # several key/value pairs side by side, so the header row itself carries no meaning
        if width > 2:
            rows = body
        return [" ".join(f"{r[i]}: {r[i + 1]}." for i in range(0, len(r) - 1, 2) if r[i] and r[i + 1])
                or " ".join(c for c in r if c) for r in rows]
    if not body or not all(header):        # no usable header row: keep the plain cells
        return [" | ".join(r) for r in rows]
    out = []
    for r in body:
        pairs = [f"{header[i]}: {r[i]}" for i in range(1, min(len(r), len(header))) if r[i]]
        pairs += [r[i] for i in range(len(header), len(r)) if r[i]]
        out.append(f"{r[0]} — " + "; ".join(pairs) + "." if pairs else r[0])
    return out
