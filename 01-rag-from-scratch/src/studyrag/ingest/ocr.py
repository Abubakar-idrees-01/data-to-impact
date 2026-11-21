"""OCR for image-only slides, scanned PDFs and photos of notes.

Real-world course material is often *pictures of text*: slides exported as images, scanned
handouts, phone photos. Without OCR, a RAG system silently indexes nothing for those pages.

We use RapidOCR (PaddleOCR's detection + recognition models running on ONNX Runtime): CPU-only,
pip-installable on Windows/macOS/Linux, no system packages. It returns one box per text line;
we rebuild reading order (top-to-bottom, then left-to-right), paragraph breaks (large vertical
gaps) and a heading guess (the tallest line near the top, i.e. the slide title).

OCR is slow-ish on old CPUs (~1-3 s per slide), so results are cached by image hash: re-indexing
the same file never runs OCR twice.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import threading
from dataclasses import dataclass
from pathlib import Path

_MIN_SCORE = 0.5
_engine = None
_lock = threading.Lock()


@dataclass
class OcrResult:
    text: str
    heading: str


class OcrUnavailable(RuntimeError):
    pass


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as e:
                raise OcrUnavailable("OCR needs the 'rapidocr' package: pip install rapidocr") from e
            _engine = RapidOCR()
        return _engine


def available() -> bool:
    try:
        import rapidocr  # noqa: F401
        return True
    except ImportError:
        return False


class OcrCache:
    """JSON file mapping sha1(image bytes) -> OCR result."""

    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict[str, dict] = {}
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text())
            except json.JSONDecodeError:
                self.data = {}

    def get(self, key: str) -> OcrResult | None:
        d = self.data.get(key)
        return OcrResult(**d) if d else None

    def put(self, key: str, res: OcrResult) -> None:
        self.data[key] = {"text": res.text, "heading": res.heading}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data))


_cache = OcrCache(None)


def set_cache_path(path: Path) -> None:
    global _cache
    if _cache.path != path:
        _cache = OcrCache(path)


def ocr_image(image: bytes) -> OcrResult:
    key = hashlib.sha1(image).hexdigest()
    cached = _cache.get(key)
    if cached is not None:
        return cached
    out = _get_engine()(image)
    boxes = getattr(out, "boxes", None)
    txts = getattr(out, "txts", None)
    scores = getattr(out, "scores", None)
    if boxes is None or txts is None or len(txts) == 0:
        res = OcrResult("", "")
    else:
        res = _layout(boxes, txts, scores)
    _cache.put(key, res)
    return res


def _layout(boxes, txts, scores) -> OcrResult:
    items = []
    for box, txt, sc in zip(boxes, txts, scores):
        if sc < _MIN_SCORE or not str(txt).strip():
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"text": str(txt).strip(), "top": min(ys), "bottom": max(ys), "left": min(xs),
                      "h": max(ys) - min(ys)})
    if not items:
        return OcrResult("", "")
    page_h = max(i["bottom"] for i in items) or 1.0
    med_h = statistics.median(i["h"] for i in items)

    # group boxes into lines: same line if vertical centres are within half a line height
    items.sort(key=lambda i: (i["top"] + i["bottom"]) / 2)
    lines: list[list[dict]] = []
    for it in items:
        c = (it["top"] + it["bottom"]) / 2
        if lines:
            last = lines[-1]
            lc = statistics.mean((x["top"] + x["bottom"]) / 2 for x in last)
            if abs(c - lc) < 0.5 * max(med_h, it["h"]):
                last.append(it)
                continue
        lines.append([it])

    out_lines, prev_bottom, heading, heading_h = [], None, "", 0.0
    for ln in lines:
        ln.sort(key=lambda x: x["left"])
        text = " ".join(x["text"] for x in ln)
        top, bottom, h = min(x["top"] for x in ln), max(x["bottom"] for x in ln), max(x["h"] for x in ln)
        if re.fullmatch(r"\d{1,3}", text) and top > 0.8 * page_h:      # slide number in the corner
            continue
        if top < 0.3 * page_h and h > 1.3 * med_h and h > heading_h:     # big text near the top = title
            heading, heading_h = text, h
        if prev_bottom is not None and top - prev_bottom > 1.2 * med_h:
            out_lines.append("")                                       # paragraph break
        out_lines.append(text)
        prev_bottom = bottom
    text = "\n".join(out_lines)
    text = re.sub(r"^[•·●▪◦]\s*", "- ", text, flags=re.M)
    return OcrResult(text.strip(), heading)
