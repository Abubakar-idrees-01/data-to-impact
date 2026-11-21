"""Chunking: split Documents into retrieval units ("chunks").

Why chunk at all? An embedding of a whole 40-slide lecture is a blurry average of everything in it;
an embedding of one slide is specific. But chunks that are too small lose context ("it increases
conversion" — what does?). Chunking is the trade-off between precision and context.

Three strategies, compared with measurements in docs/DECISIONS.md:

1. fixed      — sliding window of N words with overlap. Ignores structure. The classic baseline.
2. structure  — respects the document's own units (slide / page / section). Oversized units are
                split recursively (paragraphs -> lines -> sentences -> words); tiny neighbouring
                units (e.g. a title-only slide) are merged so no chunk is too thin to be useful.
3. semantic   — splits where the *meaning* shifts: embeds each sentence and cuts where adjacent
                sentences are least similar.

Every chunk carries a contextual header (document title + location + heading) that is prepended
when embedding/indexing, so a chunk like "- 3 key benefits..." is still findable by its topic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

from .ingest.models import Block, Document, location_label


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    kind: str          # slide | page | section
    loc_start: int
    loc_end: int
    heading: str = ""

    @property
    def location(self) -> str:
        return location_label(self.kind, self.loc_start, self.loc_end, self.heading)

    @property
    def header(self) -> str:
        return f"{self.doc_title} — {self.location}"

    @property
    def index_text(self) -> str:
        """What BM25 and the embedding model see: contextual header + chunk text."""
        return f"{self.header}\n{self.text}"

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


# ------------------------------------------------------------------ helpers
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\-])|\n+")


def split_sentences(text: str) -> list[str]:
    """Sentence-ish units. On slides, each bullet line counts as a sentence."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def _words(text: str) -> int:
    return len(text.split())


def _recursive_split(text: str, max_words: int) -> list[str]:
    """Split text into pieces <= max_words, preferring the largest natural boundary available."""
    if _words(text) <= max_words:
        return [text]
    for sep in ("\n\n", "\n", "SENT", " "):
        parts = split_sentences(text) if sep == "SENT" else text.split(sep)
        parts = [p for p in parts if p.strip()]
        if len(parts) <= 1:
            continue
        joiner = " " if sep in ("SENT", " ") else sep
        pieces, current = [], ""
        for part in parts:
            candidate = f"{current}{joiner}{part}" if current else part
            if _words(candidate) <= max_words:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = part
        if current:
            pieces.append(current)
        out = []
        for p in pieces:                 # a single part may still be too long: recurse deeper
            out.extend(_recursive_split(p, max_words) if _words(p) > max_words else [p])
        return out
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _make(doc: Document, n: int, text: str, kind: str, start: int, end: int, heading: str) -> Chunk:
    return Chunk(chunk_id=f"{doc.doc_id}:{n}", doc_id=doc.doc_id, doc_title=doc.title,
                 text=text.strip(), kind=kind, loc_start=start, loc_end=end, heading=heading)


# ------------------------------------------------------------------ 1. fixed
def chunk_fixed(doc: Document, size: int = 180, overlap: int = 30) -> list[Chunk]:
    words: list[tuple[str, int]] = [(w, b.location) for b in doc.blocks for w in b.text.split()]
    if not words:
        return []
    kind = doc.blocks[0].kind
    step = max(1, size - overlap)
    chunks = []
    for n, i in enumerate(range(0, len(words), step)):
        window = words[i:i + size]
        chunks.append(_make(doc, n, " ".join(w for w, _ in window), kind,
                            window[0][1], window[-1][1], ""))
        if i + size >= len(words):
            break
    return chunks


# ------------------------------------------------------------------ 2. structure-aware
def chunk_structure(doc: Document, max_words: int = 180, min_words: int = 50) -> list[Chunk]:
    # a) split oversized blocks recursively
    units: list[tuple[str, Block]] = []
    for b in doc.blocks:
        pieces = _recursive_split(b.text, max_words)
        for j, piece in enumerate(pieces):
            # continuation pieces of a long section repeat its heading so they stay findable
            if j > 0 and b.heading and not piece.startswith(b.heading):
                piece = f"{b.heading} (cont.)\n{piece}"
            units.append((piece, b))
    # b) merge thin neighbours (e.g. a title-only slide + the next slide)
    chunks: list[Chunk] = []
    buf_text, buf_blocks = "", []
    for text, b in units:
        if buf_blocks and (_words(buf_text) >= min_words or _words(buf_text) + _words(text) > max_words):
            chunks.append(_flush(doc, len(chunks), buf_text, buf_blocks))
            buf_text, buf_blocks = "", []
        buf_text = f"{buf_text}\n\n{text}" if buf_text else text
        buf_blocks.append(b)
    if buf_blocks:
        if chunks and _words(buf_text) < min_words // 2 and \
                _words(chunks[-1].text) + _words(buf_text) <= max_words:
            last = chunks.pop()               # fold a tiny tail into the previous chunk
            chunks.append(_make(doc, len(chunks), f"{last.text}\n\n{buf_text}", last.kind,
                                last.loc_start, buf_blocks[-1].location, last.heading))
        else:
            chunks.append(_flush(doc, len(chunks), buf_text, buf_blocks))
    return chunks


def _flush(doc: Document, n: int, text: str, blocks: list[Block]) -> Chunk:
    heading = next((b.heading for b in blocks if b.heading), "")
    return _make(doc, n, text, blocks[0].kind, blocks[0].location, blocks[-1].location, heading)


# ------------------------------------------------------------------ 3. semantic
def chunk_semantic(doc: Document, embed: Callable[[list[str]], np.ndarray],
                   max_words: int = 180, min_words: int = 40, percentile: float = 25.0) -> list[Chunk]:
    sents: list[tuple[str, Block]] = [(s, b) for b in doc.blocks for s in split_sentences(b.text)]
    if not sents:
        return []
    vecs = np.asarray(embed([s for s, _ in sents]), dtype=np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1) if len(sents) > 1 else np.array([])
    cut_below = float(np.percentile(sims, percentile)) if len(sims) else 0.0

    chunks: list[Chunk] = []
    cur: list[tuple[str, Block]] = [sents[0]]
    for i in range(1, len(sents)):
        cur_words = sum(_words(s) for s, _ in cur)
        topic_shift = sims[i - 1] < cut_below and cur_words >= min_words
        if topic_shift or cur_words + _words(sents[i][0]) > max_words:
            chunks.append(_flush(doc, len(chunks), "\n".join(s for s, _ in cur), [b for _, b in cur]))
            cur = []
        cur.append(sents[i])
    chunks.append(_flush(doc, len(chunks), "\n".join(s for s, _ in cur), [b for _, b in cur]))
    return chunks


def chunk_document(doc: Document, strategy: str = "structure", *, size: int = 180, overlap: int = 30,
                   min_words: int = 50, embed: Callable | None = None) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(doc, size, overlap)
    if strategy == "structure":
        return chunk_structure(doc, size, min_words)
    if strategy == "semantic":
        if embed is None:
            raise ValueError("semantic chunking needs an embed function")
        return chunk_semantic(doc, embed, size)
    raise ValueError(f"unknown chunk strategy: {strategy}")
