"""Data model for loaded documents.

A Document is a list of Blocks. A Block is the smallest unit that has a *citable location*:
a slide (PPTX), a page (PDF) or a heading-delimited section (DOCX/Markdown/TXT).
Keeping the location attached from the very first step is what makes precise citations possible later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Block:
    text: str
    location: int          # 1-based slide / page / section number
    kind: str              # "slide" | "page" | "section"
    heading: str = ""      # slide title or section heading, if any


@dataclass
class Document:
    doc_id: str            # short content hash: same file => same id
    title: str             # human-friendly name (file name without extension)
    source: str            # original file name
    doc_type: str          # "pdf" | "docx" | "pptx" | "md" | "txt"
    blocks: list[Block] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks)


def location_label(kind: str, start: int, end: int, heading: str = "") -> str:
    """Human-readable citation target, e.g. 'Slide 4', 'Slides 4-6', 'Page 2', 'Section 3: Pricing'."""
    noun = {"slide": "Slide", "page": "Page", "section": "Section"}.get(kind, "Part")
    label = f"{noun} {start}" if start == end else f"{noun}s {start}-{end}"
    if heading:
        label += f": {heading}"
    return label
