"""File loaders: turn PDF / DOCX / PPTX / TXT / MD (and legacy DOC / PPT) into Documents.

Each loader preserves *where* text came from (slide, page or section number) because
citations are only as precise as the locations recorded here.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .cleaning import clean_text
from .models import Block, Document

SUPPORTED = {".pdf", ".docx", ".pptx", ".txt", ".md", ".doc", ".ppt"}


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_document(path: str | Path) -> Document:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED))}")
    if ext in {".doc", ".ppt"}:
        return _load_legacy_office(path)
    loader = {".pdf": _load_pdf, ".docx": _load_docx, ".pptx": _load_pptx,
              ".txt": _load_text, ".md": _load_markdown}[ext]
    blocks = [b for b in loader(path) if b.text.strip()]
    return Document(doc_id=file_hash(path), title=path.stem, source=path.name,
                    doc_type=ext.lstrip("."), blocks=blocks)


# ---------------------------------------------------------------- PPTX
def _load_pptx(path: Path) -> list[Block]:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    def shape_texts(shape) -> list[str]:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            out = []
            for s in _reading_order(shape.shapes):
                out.extend(shape_texts(s))
            return out
        if getattr(shape, "has_table", False) and shape.has_table:
            rows = []
            for row in shape.table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            return ["\n".join(rows)]
        if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
            paras = []
            for p in shape.text_frame.paragraphs:
                t = "".join(r.text for r in p.runs).strip() or p.text.strip()
                if t:
                    # indented paragraphs on slides are sub-bullets
                    paras.append(("  " * p.level) + "- " + t if p.level or len(shape.text_frame.paragraphs) > 1 else t)
            return ["\n".join(paras)]
        return []

    prs = Presentation(str(path))
    blocks = []
    for i, slide in enumerate(prs.slides, start=1):
        title_shape = slide.shapes.title
        title = clean_text(title_shape.text) if title_shape is not None and title_shape.has_text_frame else ""
        parts = []
        for shape in _reading_order(slide.shapes):
            if title_shape is not None and shape.shape_id == title_shape.shape_id:
                continue
            parts.extend(t for t in shape_texts(shape) if t.strip())
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip() if slide.notes_slide.notes_text_frame else ""
            if notes:
                parts.append("Speaker notes: " + notes)
        body = clean_text("\n".join(parts))
        text = f"{title}\n{body}".strip() if title else body
        blocks.append(Block(text=text, location=i, kind="slide", heading=title.replace("\n", " ")))
    return blocks


def _reading_order(shapes):
    """Sort shapes top-to-bottom, then left-to-right (rounded rows), roughly how people read a slide."""
    def key(s):
        top = s.top if s.top is not None else 0
        left = s.left if s.left is not None else 0
        return (round(top / 300000), left)   # ~0.3 inch row buckets
    return sorted(shapes, key=key)


# ---------------------------------------------------------------- DOCX
def _load_docx(path: Path) -> list[Block]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    sections: list[tuple[str, list[str]]] = [("", [])]
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "").lower()
            if style.startswith("heading") or style == "title":
                sections.append((text, []))
            elif "list" in style:
                sections[-1][1].append("- " + text)
            else:
                sections[-1][1].append(text)
        elif tag == "tbl":
            t = Table(child, d)
            for row in t.rows:
                cells = []
                for c in row.cells:            # merged cells repeat; keep each value once per row
                    v = c.text.strip()
                    if v and (not cells or cells[-1] != v):
                        cells.append(v)
                if cells:
                    sections[-1][1].append(" | ".join(cells))
    blocks = []
    n = 0
    for heading, lines in sections:
        if not heading and not lines:
            continue
        n += 1
        text = clean_text((heading + "\n" if heading else "") + "\n".join(lines))
        blocks.append(Block(text=text, location=n, kind="section", heading=heading))
    return blocks


# ---------------------------------------------------------------- PDF
def _load_pdf(path: Path) -> list[Block]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [Block(text=clean_text(page.extract_text() or ""), location=i, kind="page")
            for i, page in enumerate(reader.pages, start=1)]


# ---------------------------------------------------------------- TXT / MD
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _load_markdown(path: Path) -> list[Block]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in raw.splitlines():
        m = _MD_HEADING.match(line)
        if m:
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    blocks, n = [], 0
    for heading, lines in sections:
        body = clean_text("\n".join(lines))
        if not body and not heading:
            continue
        if not body:          # heading with no text of its own: skip, its children carry content
            continue
        n += 1
        blocks.append(Block(text=clean_text(f"{heading}\n{body}" if heading else body),
                            location=n, kind="section", heading=heading))
    return blocks


def _load_text(path: Path) -> list[Block]:
    raw = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    return [Block(text=raw, location=1, kind="section")]


# ---------------------------------------------------------------- legacy Office
def _load_legacy_office(path: Path) -> Document:
    """Convert .doc/.ppt to .docx/.pptx with LibreOffice (headless), then load normally."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError(f"Loading {path.suffix} files needs LibreOffice installed (soffice not found).")
    target = "docx" if path.suffix.lower() == ".doc" else "pptx"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([soffice, "--headless", "--convert-to", target, "--outdir", tmp, str(path)],
                       check=True, capture_output=True, timeout=180)
        converted = Path(tmp) / f"{path.stem}.{target}"
        doc = load_document(converted)
    doc.doc_id, doc.title, doc.source = file_hash(path), path.stem, path.name
    return doc
