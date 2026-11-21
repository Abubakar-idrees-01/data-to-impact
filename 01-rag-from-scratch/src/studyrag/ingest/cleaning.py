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
